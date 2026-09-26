"""Poseidon-T Ocean adapter."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vendor.poseidon_scot import ScOT, ScOTConfig


class PoseidonTransition(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 5, pretrained_directory: str | None = None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.img_size = 256
        self.in_project = nn.Conv2d(input_channels, 4, 1)
        config = ScOTConfig(
            image_size=self.img_size,
            patch_size=4,
            num_channels=4,
            num_out_channels=4,
            embed_dim=48,
            depths=[4, 4, 4, 4],
            num_heads=[3, 6, 12, 24],
            skip_connections=[2, 2, 2, 0],
            window_size=16,
            mlp_ratio=4.0,
            p=2,
            residual_model="convnext",
            use_conditioning=True,
        )
        if pretrained_directory is None:
            self.poseidon = ScOT(config)
        else:
            self.poseidon = ScOT.from_pretrained(pretrained_directory, config=config, ignore_mismatched_sizes=False)
        self.out_project = nn.Conv2d(4, output_channels, 1)

    def forward(self, sequence: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = sequence.shape
        visible = sequence[:, -1]
        padded = self.img_size
        projected = self.in_project(visible)
        projected = F.pad(projected, (0, padded - width, 0, padded - height))
        # Poseidon uses a scalar physical-time coordinate for conditional norms;
        # the T checkpoint is unconditional, so this is a deterministic metadata
        # input retained for the adapter contract and set to the latest calendar.
        time = calendar[:, -1, :1].to(projected)
        result = self.poseidon(pixel_values=projected, time=time, return_dict=True)
        prediction = result.output
        prediction = self.out_project(prediction)
        return prediction[..., :height, :width]
