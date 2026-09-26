"""Ocean baseline adapters.

U-Net, ConvLSTM and FNO use task-specific implementations. PDE-Transformer-MSE
loads the external PDE-S mixed-channel core. All use the same monthly interface.
"""

import torch
from torch import nn
from torch.nn import functional as F

def _features(values, valid, geometry, calendar):
    batch, _, _, height, width = values.shape
    cal = calendar[:, :, :, None, None].expand(-1, -1, -1, height, width)
    return torch.cat((values, valid.float(), geometry[:, None].expand(batch, values.shape[1], -1, -1, -1), cal), 2)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetStep(nn.Module):
    def __init__(self, input_channels, output_channels=5, width=32):
        super().__init__()
        self.down1 = DoubleConv(input_channels, width)
        self.down2 = DoubleConv(width, width * 2)
        self.down3 = DoubleConv(width * 2, width * 4)
        self.mid = DoubleConv(width * 4, width * 8)
        self.up3 = DoubleConv(width * 8 + width * 4, width * 4)
        self.up2 = DoubleConv(width * 4 + width * 2, width * 2)
        self.up1 = DoubleConv(width * 2 + width, width)
        self.out = nn.Conv2d(width, output_channels, 1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(F.avg_pool2d(a, 2))
        c = self.down3(F.avg_pool2d(b, 2))
        d = self.mid(F.avg_pool2d(c, 2))
        u = F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False)
        u = self.up3(torch.cat((u, c), 1))
        u = F.interpolate(u, size=b.shape[-2:], mode="bilinear", align_corners=False)
        u = self.up2(torch.cat((u, b), 1))
        u = F.interpolate(u, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.up1(torch.cat((u, a), 1)))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        gates = self.gates(torch.cat((x, h), 1))
        i, f, g, o = gates.chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMTransition(nn.Module):
    def __init__(self, input_channels, output_channels=5, hidden_channels=48):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels)
        self.readout = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_channels, output_channels, 1))

    def forward(self, sequence):
        batch, _, _, height, width = sequence.shape
        h = sequence.new_zeros(batch, self.cell.hidden_channels, height, width)
        c = h.clone()
        for frame in sequence.unbind(1):
            h, c = self.cell(frame, (h, c))
        return self.readout(h), (h, c)


class SpectralConv2d(nn.Module):
    """Fourier layer following neuraloperator's 2-D spectral convolution."""

    def __init__(self, in_channels, out_channels, modes_x=16, modes_y=16):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes_x, self.modes_y = modes_x, modes_y
        scale = 1 / (in_channels * out_channels)
        self.weight_pos = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    @staticmethod
    def _mul(input, weight):
        return torch.einsum("bixy,ioxy->boxy", input, weight)

    def forward(self, x):
        batch, _, height, width = x.shape
        spectrum = torch.fft.rfft2(x, norm="ortho")
        output = torch.zeros(batch, self.out_channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)
        mx, my = min(self.modes_x, height), min(self.modes_y, width // 2 + 1)
        output[:, :, :mx, :my] = self._mul(spectrum[:, :, :mx, :my], self.weight_pos[:, :, :mx, :my])
        output[:, :, -mx:, :my] = self._mul(spectrum[:, :, -mx:, :my], self.weight_neg[:, :, :mx, :my])
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, width, modes_x=16, modes_y=16):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class FNOTransition(nn.Module):
    def __init__(self, input_channels, output_channels=5, width=48, modes_x=16, modes_y=16, depth=4):
        super().__init__()
        self.lift = nn.Conv2d(input_channels, width, 1)
        self.blocks = nn.Sequential(*[FNOBlock(width, modes_x, modes_y) for _ in range(depth)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, output_channels, 1))

    def forward(self, x):
        return self.project(self.blocks(self.lift(x)))


class PhysicsGuidedGraphBlock(nn.Module):
    """PyTorch-only Ocean adaptation of NeuralOM's graph message passing.

    The official NeuralOM graph is DGL-based and tied to a 361x720 daily grid.
    For the project's 90x180 monthly grid, messages use the four spherical-grid
    neighbors and geometry-conditioned edge gates while keeping the same
    physics-guided interaction pattern.
    """

    def __init__(self, width, geometry_channels=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.edge_gate = nn.Conv2d(geometry_channels, 4, 1)
        self.message = nn.Conv2d(width * 4, width, 1)
        self.update = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 1))

    def forward(self, hidden, geometry):
        # Longitude is periodic; latitude uses replicated polar boundaries.
        north = torch.cat((hidden[:, :, 1:], hidden[:, :, -1:]), dim=2)
        south = torch.cat((hidden[:, :, :1], hidden[:, :, :-1]), dim=2)
        east = torch.roll(hidden, 1, -1)
        west = torch.roll(hidden, -1, -1)
        neighbors = torch.cat((north, south, east, west), dim=1)
        gates = torch.sigmoid(self.edge_gate(geometry))
        gated = torch.cat(tuple(neighbors[:, i * hidden.shape[1] : (i + 1) * hidden.shape[1]] * gates[:, i : i + 1] for i in range(4)), dim=1)
        return hidden + self.update(self.norm(self.message(gated)))


class NeuralOMTransition(nn.Module):
    """NeuralOM adaptation with progressive residual correction."""

    def __init__(self, input_channels, output_channels=5, width=48, corrections=3, geometry_channels=4):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv2d(input_channels, width, 1), nn.GELU())
        self.base = nn.Conv2d(width, output_channels, 1)
        self.blocks = nn.ModuleList([PhysicsGuidedGraphBlock(width, geometry_channels) for _ in range(corrections)])
        self.corrections = nn.ModuleList([nn.Conv2d(width, output_channels, 1) for _ in range(corrections)])

    def forward(self, x, geometry):
        hidden = self.lift(x)
        prediction = x[:, :5] + self.base(hidden)
        for block, correction in zip(self.blocks, self.corrections):
            hidden = block(hidden, geometry)
            prediction = prediction + correction(hidden)
        return prediction


class PDETransformerTransition(nn.Module):
    """Official-source PDE-S core with Ocean-grid padding and cropping."""

    def __init__(self, input_channels, output_channels=5):
        super().__init__()
        from vendor.pde_transformer_mixed import PDE_S

        # patch size 4, two 2x downsamplers, and window size 8 must all divide
        # every multiscale token grid: 4 * 2**2 * 8 = 128.
        self.multiple = 128
        self.model = PDE_S(
            in_channels=input_channels,
            out_channels=output_channels,
            patch_size=4,
            periodic=False,
            carrier_token_active=False,
            window_size=8,
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_height = (-height) % self.multiple
        pad_width = (-width) % self.multiple
        padded = F.pad(x, (0, pad_width, 0, pad_height))
        timestep = padded.new_zeros(padded.shape[0])
        prediction = self.model(padded, timestep, None)
        return prediction[..., :height, :width]


class OceanBaseline(nn.Module):
    def __init__(self, method, channels=5, history=12, geometry_channels=4, calendar_channels=2, pretrained_checkpoint=None, load_pretrained=True):
        super().__init__()
        self.method = method
        per_frame = channels * 2 + geometry_channels + calendar_channels
        if method == "unet":
            self.transition = UNetStep(history * per_frame, channels)
        elif method == "convlstm":
            self.transition = ConvLSTMTransition(per_frame, channels)
        elif method == "fno":
            self.transition = FNOTransition(per_frame, channels)
        elif method == "pde_transformer_mse":
            self.transition = PDETransformerTransition(per_frame, channels)
        elif method == "neuralom":
            self.transition = NeuralOMTransition(per_frame, channels, geometry_channels=geometry_channels)
        elif method in ("dpot_pretrained", "dpot_scratch"):
            from model.dpot import DPOTTransition

            checkpoint = pretrained_checkpoint if method == "dpot_pretrained" else None
            if method == "dpot_pretrained" and checkpoint is None and load_pretrained:
                raise ValueError("dpot_pretrained requires --pretrained-checkpoint")
            self.transition = DPOTTransition(per_frame, channels, pretrained_checkpoint=checkpoint)
        elif method in ("poseidon_pretrained", "poseidon_scratch"):
            from model.poseidon import PoseidonTransition

            checkpoint = pretrained_checkpoint if method == "poseidon_pretrained" else None
            if method == "poseidon_pretrained" and checkpoint is None and load_pretrained:
                raise ValueError("poseidon_pretrained requires --pretrained-checkpoint")
            self.transition = PoseidonTransition(per_frame, channels, pretrained_directory=checkpoint)
        elif method == "pde_transformer_pretrained":
            from model.pde_transformer_pretrained import PDETransformerPretrainedTransition

            if pretrained_checkpoint is None and load_pretrained:
                raise ValueError("pde_transformer_pretrained requires --pretrained-checkpoint")
            self.transition = PDETransformerPretrainedTransition(per_frame, channels, pretrained_checkpoint)
        else:
            raise ValueError(f"unknown baseline method: {method}")

    def step(self, values, valid, geometry, calendar):
        features = _features(values, valid, geometry, calendar)
        if self.method == "convlstm":
            prediction, _ = self.transition(features)
        elif self.method == "unet":
            prediction = self.transition(features.flatten(1, 2))
        elif self.method == "neuralom":
            prediction = self.transition(features[:, -1], geometry)
        elif self.method in ("dpot_pretrained", "dpot_scratch"):
            prediction = self.transition(features)
        elif self.method in ("poseidon_pretrained", "poseidon_scratch"):
            prediction = self.transition(features, calendar)
        elif self.method == "pde_transformer_pretrained":
            prediction = self.transition(features)
        else:
            prediction = self.transition(features[:, -1])
        return prediction

    def rollout(self, x, valid, geometry, calendar, future_calendar):
        values, masks, predictions = x, valid, []
        for index in range(future_calendar.shape[1]):
            prediction = self.step(values, masks, geometry, calendar)
            predictions.append(prediction)
            values = torch.cat((values[:, 1:], prediction[:, None]), 1)
            masks = torch.cat((masks[:, 1:], torch.ones_like(prediction, dtype=torch.bool)[:, None]), 1)
            calendar = torch.cat((calendar[:, 1:], future_calendar[:, index:index + 1]), 1)
        return torch.stack(predictions, 1)
