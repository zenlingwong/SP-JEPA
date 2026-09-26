"""F4 pretrained PDE-Transformer mc-s Ocean adapter."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from safetensors.torch import load_file

from vendor.pde_transformer_mixed import PDE_S


class PDETransformerPretrainedTransition(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 5, checkpoint: str | Path | None = None):
        super().__init__()
        self.input_channels = input_channels
        self.in_project = nn.Conv2d(input_channels, 2, 1)
        self.model = PDE_S(
            in_channels=2,
            out_channels=2,
            patch_size=4,
            periodic=True,
            carrier_token_active=False,
            window_size=8,
        )
        if checkpoint is not None:
            state = load_file(str(checkpoint), device="cpu")
            state = {key[6:] if key.startswith("model.") else key: value for key, value in state.items()}
            missing, unexpected = self.model.load_state_dict(state, strict=True)
            if missing or unexpected:
                raise RuntimeError(f"PDE-Transformer checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.out_project = nn.Conv2d(2, output_channels, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        visible = sequence[:, -1]
        height, width = visible.shape[-2:]
        pad_height = (-height) % 128
        pad_width = (-width) % 128
        projected = self.in_project(visible)
        projected = F.pad(projected, (0, pad_width, 0, pad_height))
        timestep = projected.new_zeros(projected.shape[0])
        prediction = self.model(projected, timestep, None)
        prediction = self.out_project(prediction)
        return prediction[..., :height, :width]
