"""ENMA adapter for the monthly grid.

The VAE and masked flow-transformer cores are loaded from a verified external
cache. This file bridges the
project's [B,T,C,H,W] tensors, masks, and 90x180 grid to the official regular
2-D interfaces; it does not replace the flow model with a deterministic head.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from enma.encoding.regular._2d.encoder_decoder import RegularEncoderDecoder2D
from enma.process.model import ENMA


PAD_H, PAD_W = 96, 192
GRID_H, GRID_W = 90, 180
LATENT_H, LATENT_W = 6, 12
TOKEN_DIM = 8


def _cfg(**model_overrides):
    model = dict(
        name="regular-2d",
        image_size=[PAD_H, PAD_W],
        token_dim=TOKEN_DIM,
        layers=["residual", "compress_space", "residual", "compress_space", "residual", "compress_space", "residual", "compress_space", "residual"],
        residual_conv_kernel_size=3,
        channels=5,
        init_dim=16,
        max_dim=64,
        input_conv_kernel_size=3,
        output_conv_kernel_size=3,
        pad_mode="circular",
        num_groups=8,
    )
    model.update(model_overrides)
    return SimpleNamespace(model=SimpleNamespace(**model))


class OceanENMAVAE(nn.Module):
    """Official ENMA regular-grid VAE with Ocean channel/grid bridging."""

    def __init__(self):
        super().__init__()
        self.cfg = _cfg()
        self.model = RegularEncoderDecoder2D(self.cfg)
        self.token_dim = TOKEN_DIM
        self.mean_fc = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.logvar_fc = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.sample_posterior = True

    def _pad(self, values):
        # values: [B,T,5,90,180] -> official regular encoder [B,T,HW,5]
        padded = F.pad(values, (0, PAD_W - GRID_W, 0, PAD_H - GRID_H))
        return padded.permute(0, 1, 3, 4, 2).reshape(values.shape[0], values.shape[1], PAD_H * PAD_W, 5)

    def _coords(self, values):
        return values.new_zeros(values.shape[0], values.shape[1], PAD_H * PAD_W, 1)

    def encode(self, values, *, mode=True):
        raw = self.model.encode(self._pad(values), self._coords(values))
        mu = self.mean_fc(raw)
        logvar = self.logvar_fc(raw).clamp(-20.0, 20.0)
        if mode or not self.sample_posterior:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def encode_with_kl(self, values):
        raw = self.model.encode(self._pad(values), self._coords(values))
        mu = self.mean_fc(raw)
        logvar = self.logvar_fc(raw).clamp(-20.0, 20.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.sample_posterior else mu
        kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
        return z, kl

    def decode(self, latent, frames):
        coords = frames.new_zeros(frames.shape[0], latent.shape[1], PAD_H * PAD_W, 1)
        raw = self.model.decode(latent, coords)
        raw = raw.reshape(frames.shape[0], latent.shape[1], PAD_H, PAD_W, 5)
        return raw[:, :, :GRID_H, :GRID_W].permute(0, 1, 4, 2, 3)

    def forward(self, values):
        latent, kl = self.encode_with_kl(values)
        return self.decode(latent, values), kl


def enma_flow_config():
    model = SimpleNamespace(
        vae_embed_dim=TOKEN_DIM,
        num_tokens=[LATENT_H, LATENT_W],
        model_type="flow",
        depth=4,
        embed_dim=128,
        dim_ffn=512,
        dropout=0.0,
        num_heads=4,
        norm_first=True,
        qk_norm=False,
        norm="rms",
        activation="gelu",
        flex_attn=False,
        kv_cache=False,
        patch_size=[1, 1],
        pos_embed="sin",
        num_frames=15,
        rotary=False,
        theta=10000,
        axes_dim_st=[32, 32],
        axes_dim_s=[64, 64],
        use_liger_rope=False,
        # Keep the official temporal mixer active: upstream fwd expects it to
        # restore [B,time,tokens,dim] before spatial conditioning.
        temp_mixer_rank=24,
        diffloss_d=2,
        diffloss_w=128,
        diffusion_batch_mul=1,
        num_iter=8,
        num_sample_steps=32,
    )
    return SimpleNamespace(model=model, data=SimpleNamespace(icl_train=0))


class OceanENMAFlow(nn.Module):
    """Official ENMA masked autoregressive flow with Ocean latent I/O."""

    def __init__(self):
        super().__init__()
        self.cfg = enma_flow_config()
        self.model = ENMA(self.cfg)

    def loss(self, latent_sequence):
        return self.model("fwd", z=latent_sequence, gd_truth_z=latent_sequence)

    @torch.no_grad()
    def sample(self, history_latent, *, samples=64, seed=20260920):
        # ENMA's generate samples one trajectory at a time from the fixed seed.
        was_training = self.training
        self.eval()
        out = []
        for sample_id in range(samples):
            torch.manual_seed(seed + sample_id)
            generated = self.model(
                "generate", z=history_latent, t=history_latent.new_zeros(history_latent.shape[0], 15), input_len=history_latent.shape[1], temperature=1.0
            )
            out.append(generated[:, history_latent.shape[1] :])
        if was_training:
            self.train()
        return torch.stack(out, dim=1)
