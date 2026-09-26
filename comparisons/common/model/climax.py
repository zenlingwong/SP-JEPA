"""Minimal PyTorch ClimaX core and Ocean forecasting adapter.

The module preserves the computation pattern of the published ClimaX release
(variable patch embeddings, variable aggregation, ViT blocks, lead-time and
patch decoder) while avoiding the upstream timm/Lightning runtime.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size: tuple[int, int], patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, drop: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("embed dimension must be divisible by heads")
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def sincos_1d(dim: int, positions: Tensor) -> Tensor:
    if dim % 2:
        raise ValueError("sincos dimension must be even")
    omega = torch.arange(dim // 2, device=positions.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim / 2)))
    out = positions.reshape(-1, 1).float() * omega.reshape(1, -1)
    return torch.cat((out.sin(), out.cos()), dim=1)


def sincos_2d(dim: int, height: int, width: int, device: torch.device) -> Tensor:
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.cat((sincos_1d(dim // 2, yy.flatten()), sincos_1d(dim // 2, xx.flatten())), dim=1)


class ClimaX(nn.Module):
    """Official-source ClimaX architecture with a small Ocean-facing API."""

    def __init__(
        self,
        variables: int = 5,
        img_size: tuple[int, int] = (90, 180),
        patch_size: int = 2,
        embed_dim: int = 1024,
        depth: int = 8,
        decoder_depth: int = 2,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.variables = variables
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(self.img_size, patch_size, embed_dim) for _ in range(variables)]
        )
        self.num_patches = self.token_embeds[0].num_patches
        self.channel_embed = nn.Parameter(torch.zeros(1, variables, embed_dim))
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.var_agg = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.lead_time_embed = nn.Linear(1, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio, drop_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        head: list[nn.Module] = []
        for _ in range(decoder_depth):
            head.extend((nn.Linear(embed_dim, embed_dim), nn.GELU()))
        head.append(nn.Linear(embed_dim, variables * patch_size**2))
        self.head = nn.Sequential(*head)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.channel_embed, std=0.02)
        nn.init.normal_(self.channel_query, std=0.02)
        with torch.no_grad():
            self.pos_embed.copy_(sincos_2d(self.embed_dim, self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size, self.pos_embed.device).unsqueeze(0))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _aggregate(self, tokens: Tensor) -> Tensor:
        # tokens: B, variables, patches, dim
        b, _, patches, _ = tokens.shape
        tokens = tokens.permute(0, 2, 1, 3).reshape(b * patches, self.variables, self.embed_dim)
        query = self.channel_query.expand(b * patches, -1, -1)
        aggregated, _ = self.var_agg(query, tokens, tokens, need_weights=False)
        return aggregated[:, 0].reshape(b, patches, self.embed_dim)

    def forward_step(self, x: Tensor, lead_time: float | Tensor = 1.0) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.variables:
            raise ValueError(f"expected [B,{self.variables},H,W], got {tuple(x.shape)}")
        embeds = torch.stack([layer(x[:, i : i + 1]) for i, layer in enumerate(self.token_embeds)], dim=1)
        embeds = embeds + self.channel_embed.unsqueeze(2)
        hidden = self._aggregate(embeds) + self.pos_embed
        if not torch.is_tensor(lead_time):
            lead_time = torch.full((x.shape[0],), float(lead_time), device=x.device, dtype=x.dtype)
        hidden = self.pos_drop(hidden + self.lead_time_embed(lead_time.reshape(-1, 1).to(x.dtype)).unsqueeze(1))
        for block in self.blocks:
            hidden = block(hidden)
        patches = self.head(self.norm(hidden))
        p = self.patch_size
        h, w = self.img_size[0] // p, self.img_size[1] // p
        patches = patches.reshape(x.shape[0], h, w, p, p, self.variables).permute(0, 5, 1, 3, 2, 4)
        return patches.reshape(x.shape[0], self.variables, self.img_size[0], self.img_size[1])

    def rollout(self, history: Tensor, valid: Tensor | None = None) -> Tensor:
        current = history[:, -1]
        if valid is not None:
            current = current * valid[:, -1].to(current.dtype)
        if tuple(current.shape[-2:]) != self.img_size:
            current = F.interpolate(current, size=self.img_size, mode="bilinear", align_corners=False)
        outputs = []
        for lead in (1.0, 2.0, 3.0):
            current = self.forward_step(current, lead)
            outputs.append(current)
        return torch.stack(outputs, dim=1)

    def load_pretrained_checkpoint(self, path: str | Path) -> dict[str, object]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint = payload["state_dict"]
        mapped: dict[str, Tensor] = {}
        skipped: list[str] = []
        for key, value in checkpoint.items():
            if not key.startswith("net."):
                continue
            target = key[4:]
            if target == "pos_embed" and value.shape != self.pos_embed.shape:
                old_h, old_w = 16, 32
                new_h = self.img_size[0] // self.patch_size
                new_w = self.img_size[1] // self.patch_size
                value = F.interpolate(value.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2), size=(new_h, new_w), mode="bicubic", align_corners=False)
                value = value.permute(0, 2, 3, 1).reshape_as(self.pos_embed)
            if target in {"channel_embed"} and value.shape != self.channel_embed.shape:
                value = value.mean(dim=1, keepdim=True).expand_as(self.channel_embed).clone()
            if target == "head.4.weight" and value.shape != self.state_dict()[target].shape:
                value = value.reshape(48, -1, value.shape[-1]).mean(dim=0).repeat(self.variables, 1)
            if target == "head.4.bias" and value.shape != self.state_dict()[target].shape:
                value = value.reshape(48, -1).mean(dim=0).repeat(self.variables)
            if target.startswith("token_embeds."):
                parts = target.split(".")
                index = int(parts[1])
                if index >= self.variables:
                    continue
                target = ".".join(parts)
            if target in self.state_dict() and self.state_dict()[target].shape == value.shape:
                mapped[target] = value
            else:
                skipped.append(key)
        # The official checkpoint has variable-specific patch kernels. Average
        # them into each Ocean channel instead of silently selecting arbitrary vars.
        for suffix in ("proj.weight", "proj.bias"):
            keys = [f"token_embeds.{i}.{suffix}" for i in range(48) if f"token_embeds.{i}.{suffix}" in checkpoint]
            if keys:
                mean = torch.stack([checkpoint[f"net.{key}"] for key in keys]).mean(0)
                for i in range(self.variables):
                    mapped[f"token_embeds.{i}.{suffix}"] = mean.clone()
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        return {"loaded": len(mapped), "skipped": len(skipped), "missing": len(missing), "unexpected": len(unexpected), "strategy": "backbone_shape_match_channel_mean_head_reinit"}


def masked_area_mse(prediction: Tensor, target: Tensor, valid: Tensor, area: Tensor) -> Tensor:
    weight = valid.to(prediction.dtype) * area[:, None, None]
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1.0)
