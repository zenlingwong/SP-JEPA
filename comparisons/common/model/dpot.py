"""DPOT Ocean adapter.

The DPOT core is the pinned upstream implementation in ``vendor.dpot_upstream``.
Only the feature/channel adapters and rectangular Ocean padding live here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vendor.dpot_upstream import DPOTNet


class DPOTTransition(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 5, pretrained_checkpoint: str | None = None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.history = 10
        self.img_size = 256
        self.patch_size = 8
        self.in_project = nn.Conv2d(input_channels, 4, kernel_size=1)
        self.dpot = DPOTNet(
            img_size=self.img_size,
            patch_size=self.patch_size,
            mixing_type="afno",
            in_channels=4,
            in_timesteps=self.history,
            out_timesteps=1,
            out_channels=4,
            normalize=False,
            embed_dim=512,
            modes=32,
            depth=4,
            n_blocks=4,
            mlp_ratio=1,
            out_layer_dim=32,
            n_cls=12,
        )
        self.out_project = nn.Conv2d(4, output_channels, kernel_size=1)
        if pretrained_checkpoint is not None:
            self.load_pretrained(pretrained_checkpoint)

    @staticmethod
    def _safe_checkpoint(path: str | Path) -> dict:
        # DPOT checkpoints contain argparse.Namespace metadata.  Explicitly
        # allowlist that inert metadata type while keeping weights_only=True.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("DPOT checkpoint must contain a model state_dict")
        return checkpoint["model"]

    def load_pretrained(self, path: str | Path) -> None:
        state = self._safe_checkpoint(path)
        state = dict(state)
        position = state.get("pos_embed")
        if position is not None and tuple(position.shape) != tuple(self.dpot.pos_embed.shape):
            state["pos_embed"] = F.interpolate(position, size=self.dpot.pos_embed.shape[-2:], mode="bicubic", align_corners=False)
        missing, unexpected = self.dpot.load_state_dict(state, strict=False)
        # The official Tiny state must load completely apart from the resized
        # positional grid; silently missing backbone tensors would invalidate F2.
        unexpected = [key for key in unexpected if key != "args"]
        if missing or unexpected:
            raise RuntimeError(f"DPOT pretrained load mismatch: missing={missing}, unexpected={unexpected}")

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = sequence.shape
        if sequence.shape[1] < self.history:
            raise ValueError(f"DPOT requires at least {self.history} history frames")
        sequence = sequence[:, -self.history:]
        padded_height = max(self.img_size, ((height + self.patch_size - 1) // self.patch_size) * self.patch_size)
        padded_width = max(self.img_size, ((width + self.patch_size - 1) // self.patch_size) * self.patch_size)
        # DPOT's official implementation is square-grid and fixed-resolution.
        # Use a square 256 canvas for the 90x180 Ocean grid, then crop exactly.
        padded_width = padded_height = max(padded_height, padded_width)
        projected = self.in_project(sequence.reshape(batch * self.history, self.input_channels, height, width))
        projected = F.pad(projected, (0, padded_width - width, 0, padded_height - height))
        projected = projected.reshape(batch, self.history, 4, padded_height, padded_width).permute(0, 3, 4, 1, 2).contiguous()
        prediction, _ = self.dpot(projected)
        prediction = prediction[:, :, :, 0, :].permute(0, 3, 1, 2)
        prediction = self.out_project(prediction)
        return prediction[..., :height, :width]
