"""Global spatial LeWM for factual forecast comparison."""

import torch
from torch import nn
from torch.nn import functional as F

from .events import EventDecoder, eawm_event_loss, event_targets
from .lewm import ARPredictor, Attention, Embedder, FeedForward, MLP


class SpatialBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = Attention(dim, heads, dim // heads)
        self.mlp = FeedForward(dim, 4 * dim)

    def forward(self, x):
        x = x + self.attn(x, causal=False)
        return x + self.mlp(x)


class NumericalEncoder(nn.Module):
    def __init__(self, channels, height, width, patch, dim, depth=2, heads=4):
        super().__init__()
        if height % patch or width % patch:
            raise ValueError("height and width must be divisible by patch")
        self.channels, self.height, self.width, self.patch, self.dim = channels, height, width, patch, dim
        self.n_patches = height // patch * (width // patch)
        self.patch_embed = nn.Linear(2 * channels * patch * patch, dim)
        self.geometry_embed = nn.Linear(4, dim, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([SpatialBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, valid, geometry):
        b, t, c, h, w = x.shape
        values = torch.where(valid, x, 0)
        packed = torch.cat((values, valid.float()), 2).reshape(b * t, 2 * c, h, w)
        tokens = self.patch_embed(F.unfold(packed, self.patch, stride=self.patch).transpose(1, 2))
        if geometry.ndim == 3:
            geometry = geometry[None].expand(b, -1, -1, -1)
        geo = F.avg_pool2d(geometry, self.patch, stride=self.patch).flatten(2).transpose(1, 2)
        tokens = tokens + self.geometry_embed(geo)[:, None].expand(b, t, self.n_patches, self.dim).reshape(b * t, self.n_patches, self.dim)
        tokens = torch.cat((self.global_token.expand(b * t, -1, -1), tokens), 1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens).reshape(b, t, self.n_patches + 1, self.dim)


class StructuredDecoder(nn.Module):
    def __init__(self, dim, channels, height, width, patch, write_basis, weight):
        super().__init__()
        basis = torch.as_tensor(write_basis, dtype=torch.float32)
        weight = torch.as_tensor(weight, dtype=torch.float32).expand(channels, height, width).clone()
        if basis.ndim != 4 or basis.shape[1:] != (channels, height, width) or not 0 < basis.shape[0] < dim:
            raise ValueError("write_basis must be [K,C,H,W], with 0 < K < global-token dimension")
        A, w = basis.double().flatten(1).T, weight.double().flatten()
        column_scale = (A.square() * w[:, None]).sum(0).sqrt()
        if (column_scale == 0).any():
            raise ValueError("write_basis contains a direction outside the weighted support")
        normalized = A / column_scale
        gram = normalized.T @ (w[:, None] * normalized)
        if torch.linalg.matrix_rank(gram) != basis.shape[0]:
            raise ValueError("write_basis is not full rank on the declared weighted support")
        C = (torch.linalg.solve(gram, normalized.T * w[None]) / column_scale[:, None]).float()
        self.channels, self.height, self.width, self.patch = channels, height, width, patch
        self.write_dim = basis.shape[0]
        self.register_buffer("basis", basis)
        self.register_buffer("coefficient_map", C)
        metric = gram * column_scale[:, None] * column_scale[None, :] / w.sum()
        self.register_buffer("coefficient_metric", metric.float(), persistent=False)
        self.free_patch = nn.Sequential(nn.Linear(2 * dim - self.write_dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, channels * patch * patch))

    def coefficients(self, field):
        return field.flatten(-3) @ self.coefficient_map.T

    def field_increment(self, coefficients):
        return torch.einsum("...k,kcyx->...cyx", coefficients, self.basis)

    def coefficient_energy(self, error):
        """Area-weighted standardized-field MSE of the declared field A @ error."""
        return torch.einsum("...i,ij,...j->...", error, self.coefficient_metric, error)

    def forward(self, z):
        single = z.ndim == 3
        if single:
            z = z[:, None]
        b, t, n, d = z.shape
        eta = z[:, :, 0, :self.write_dim]
        global_h = z[:, :, 0, self.write_dim:][:, :, None].expand(-1, -1, n - 1, -1)
        patches = self.free_patch(torch.cat((z[:, :, 1:], global_h), -1)).reshape(b * t, n - 1, -1).transpose(1, 2)
        free = F.fold(patches, (self.height, self.width), self.patch, stride=self.patch).reshape(b, t, self.channels, self.height, self.width)
        free = free - self.field_increment(self.coefficients(free))
        field = free + self.field_increment(eta)
        return field[:, 0] if single else field


class OceanLeWM(nn.Module):
    def __init__(self, channels, height, width, patch, write_basis, weight, dim=128, history_size=12, predictor_depth=2, predictor_heads=4, predictor_mlp_dim=512, predictor_dim_head=32, spatial_depth=2, spatial_heads=4, projector_hidden=256, variant="structured", allow_writes=True):
        super().__init__()
        if variant != "structured" or allow_writes:
            raise ValueError("comparison model requires structured factual forecasting")
        self.dim, self.history_size = dim, history_size
        self.encoder = NumericalEncoder(channels, height, width, patch, dim, spatial_depth, spatial_heads)
        self.n_tokens = self.encoder.n_patches + 1
        self.projector = MLP(dim, projector_hidden, dim, norm_fn=nn.LayerNorm)
        self.predictor = ARPredictor(history_size * self.n_tokens, predictor_depth, predictor_heads, predictor_mlp_dim, dim, dim, dim, predictor_dim_head, dropout=0.0)
        self.action_encoder = Embedder(write_basis.shape[0] + 2, smoothed_dim=write_basis.shape[0] + 2, emb_dim=dim)
        self.pred_proj = MLP(dim, projector_hidden, dim, norm_fn=nn.LayerNorm)
        self.decoder = StructuredDecoder(dim, channels, height, width, patch, write_basis, weight)
        self.event_decoder = EventDecoder(dim, channels, height, width, patch)
        months = torch.arange(history_size).repeat_interleave(self.n_tokens)
        self.register_buffer("temporal_mask", months[:, None] >= months[None, :], persistent=False)

    @property
    def write_dim(self):
        return self.decoder.write_dim

    def encode(self, x, valid, geometry):
        return self.projector(self.encoder(x, valid, geometry))

    def predict(self, emb, calendar):
        b, t, n, d = emb.shape
        actions = emb.new_zeros(b, t, self.write_dim)
        condition = self.action_encoder(torch.cat((actions, calendar), -1))[:, :, None].expand(-1, -1, n, -1).reshape(b, t * n, d)
        pred = self.predictor(emb.reshape(b, t * n, d), condition, self.temporal_mask[:t * n, :t * n])
        return self.pred_proj(pred).reshape(b, t, n, d)

    def decode(self, z):
        return self.decoder(z)

    def rollout_state(self, state, calendar, future_calendar):
        predictions = []
        clock = calendar
        for step in range(future_calendar.shape[1]):
            pred = self.predict(state[:, -self.history_size:], clock[:, -self.history_size:])[:, -1]
            predictions.append(pred)
            state = torch.cat((state, pred[:, None]), 1)
            clock = torch.cat((clock, future_calendar[:, step:step + 1]), 1)
        return torch.stack(predictions, 1)

    def rollout(self, x, valid, geometry, horizon, calendar, future_calendar):
        return self.rollout_state(self.encode(x, valid, geometry), calendar, future_calendar[:, :horizon])


def masked_mse(pred, target, valid, area):
    while area.ndim < pred.ndim:
        area = area.unsqueeze(1)
    weight = valid * area
    return ((pred - target).square() * weight).sum() / weight.sum().clamp_min(1)


def temporal_centered_residual(emb, window):
    """Center consecutive latent frames within non-overlapping local windows."""
    if window is None:
        return emb
    if window < 2:
        raise ValueError("temporal centering window must be at least 2")
    chunks = torch.split(emb, window, dim=1)
    return torch.cat([chunk - chunk.mean(dim=1, keepdim=True) for chunk in chunks], dim=1)


def hidden_regularizer(model, sigreg, emb, temporal_window=None):
    emb = temporal_centered_residual(emb, temporal_window)
    b, t, n, d = emb.shape
    global_h = emb[:, :, 0, model.write_dim:].transpose(0, 1)
    spatial_h = emb[:, :, 1:].permute(1, 2, 0, 3).reshape(t * (n - 1), b, d)
    return (sigreg(global_h) + (n - 1) * sigreg(spatial_h)) / n


def latent_prediction_loss(model, predicted, target):
    error = predicted - target
    hidden_sum = error[..., 0, model.write_dim:].square().sum(-1) + error[..., 1:, :].square().sum((-1, -2))
    coefficient_energy = model.decoder.coefficient_energy(error[..., 0, :model.write_dim])
    return ((hidden_sum + model.write_dim * coefficient_energy) / (error.shape[-2] * error.shape[-1])).mean()


def lewm_loss(model, sigreg, batch, event_threshold=None, event_density_threshold=None, sigreg_weight=0.09, physical_weight=1.0, observed_weight=0.1, coefficient_weight=1.0, event_weight=0.1, sigreg_temporal_window=None):
    x, target = batch["x"], batch["target"]
    valid, target_valid = batch["valid"], batch["target_valid"]
    history_emb = model.encode(x, valid, batch["geometry"])
    predicted = model.rollout_state(history_emb, batch["calendar"], batch["future_calendar"])
    target_emb = model.encode(target, target_valid, batch["geometry"])
    prediction_loss = latent_prediction_loss(model, predicted, target_emb)
    prediction_native_mse = (predicted - target_emb).square().mean()
    emb = torch.cat((history_emb, target_emb), 1)
    fields, masks = torch.cat((x, target), 1), torch.cat((valid, target_valid), 1)
    sigreg_loss = hidden_regularizer(model, sigreg, emb, sigreg_temporal_window)
    physical_loss = masked_mse(model.decode(predicted), target, target_valid, batch["area"])
    observed_loss = masked_mse(model.decode(emb), fields, masks, batch["area"])
    coefficient_error = emb[:, :, 0, :model.write_dim] - model.decoder.coefficients(fields)
    coefficient_loss = model.decoder.coefficient_energy(coefficient_error).mean()
    coefficient_native_mse = coefficient_error.square().mean()
    event_loss = prediction_loss.new_zeros(())
    if event_threshold is not None and event_weight:
        previous = torch.cat((history_emb[:, -1:], predicted[:, :-1]), 1)
        labels, event_valid = event_targets(x[:, -1], target, valid[:, -1], target_valid, event_threshold)
        event_loss = eawm_event_loss(model.event_decoder(previous, predicted), labels, event_valid, event_density_threshold)
    loss = prediction_loss + sigreg_weight * sigreg_loss + physical_weight * physical_loss + observed_weight * observed_loss + coefficient_weight * coefficient_loss + event_weight * event_loss
    return loss, {"loss": loss, "prediction": prediction_loss, "sigreg": sigreg_loss, "physical": physical_loss, "observed": observed_loss, "coefficient": coefficient_loss, "event": event_loss, "coefficient_native_mse": coefficient_native_mse, "prediction_native_mse": prediction_native_mse}
