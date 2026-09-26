"""Paired-environment batches; mechanism and root IDs stay outside forward inputs."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .paired_pulse import event_bounds


ACTION_COLUMNS = torch.tensor([0, 8, 2, 10], dtype=torch.long)


def lattice_dictionary(std: torch.Tensor) -> torch.Tensor:
    """32 fixed width-0.6 Gaussian templates in normalized field units."""
    y = 2 * math.pi * torch.arange(16, dtype=torch.float64) / 16
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    columns = []
    for field in range(2):
        for cy in (0., math.pi / 2, math.pi, 3 * math.pi / 2):
            for cx in (0., math.pi / 2, math.pi, 3 * math.pi / 2):
                dy = torch.minimum((yy - cy).abs(), 2 * math.pi - (yy - cy).abs())
                dx = torch.minimum((xx - cx).abs(), 2 * math.pi - (xx - cx).abs())
                template = torch.zeros(2, 16, 16, dtype=torch.float64)
                template[field] = torch.exp(-(dx.square() + dy.square()) / (2 * .6 ** 2)) / std[field]
                columns.append(template.flatten())
    return torch.stack(columns, 1).float()


def geometry_grid() -> torch.Tensor:
    y = 2 * math.pi * torch.arange(16) / 16
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    return torch.stack((yy.sin(), yy.cos(), xx.sin(), xx.cos())).float()


def calendar_grid(length: int, start: int) -> torch.Tensor:
    t = torch.arange(start, start + length)
    a = 2 * math.pi * t / 12
    return torch.stack((a.sin(), a.cos()), -1).float()


def event_labels(history: torch.Tensor, factual: torch.Tensor, changed: torch.Tensor,
                 delta: torch.Tensor, threshold: torch.Tensor, epsilon: torch.Tensor):
    """Per-branch event bounds from storage error and numerical tolerance."""
    current = history[-1]
    factual_seq = torch.cat((current[None], factual), 0)
    changed_seq = torch.cat(((current + delta)[None], changed), 0)
    labels, known = [], []
    for values in (factual_seq, changed_seq):
        storage = (values - values.float().double()).abs().amax((0, 2, 3))
        eps = torch.maximum(torch.full_like(epsilon, 1e-7), epsilon + storage)
        bound, mask = event_bounds(values, threshold, eps)
        labels.append(bound.reshape(7, 4).float())
        known.append(mask)
    return torch.stack(labels), torch.stack(known)


def load_dataset(path: str | Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["schema"] != "paired-transport-exchange-v1":
        raise ValueError("unsupported paired-environment data schema")
    return data


def model_batch(data: dict, rows: torch.Tensor | list[int], *, paired: bool,
                sig_rows: torch.Tensor | list[int] | None = None, histories: torch.Tensor | None = None,
                device: str | torch.device = "cpu") -> dict:
    """Build exactly the factual or paired world_losses contract."""
    rows = torch.as_tensor(rows, dtype=torch.long)
    sig_rows = rows if sig_rows is None else torch.as_tensor(sig_rows, dtype=torch.long)
    records = data["pairs"] if paired else data["roots"]
    sig_records = data["roots"]
    mean = data["statistics"]["mean"].view(1, 1, 2, 1, 1)
    std = data["statistics"]["std"].view(1, 1, 2, 1, 1)

    def normalized(source, indices, key):
        return ((source[key][indices] - mean) / std).float()

    x = normalized(records, rows, "history") if histories is None else ((histories - mean) / std).float()
    y = normalized(records, rows, "factual")
    sx = normalized(sig_records, sig_rows, "history")
    sy = normalized(sig_records, sig_rows, "factual")
    if not paired and histories is not None:
        sx = x
    b = len(rows)
    sb = len(sig_rows)
    geom = geometry_grid()
    cal = calendar_grid(6, -5)
    future = calendar_grid(6, 1)
    events = records["events"][rows]
    known = records["known"][rows]
    batch = {
        "x": x, "valid": torch.ones_like(x, dtype=torch.bool),
        "target": y, "target_valid": torch.ones_like(y, dtype=torch.bool),
        "geometry": geom.expand(b, -1, -1, -1), "area": torch.ones(b, 1, 16, 16),
        "calendar": cal.expand(b, -1, -1), "future_calendar": future.expand(b, -1, -1),
        "current_event_truth": events[:, 0, 0], "current_event_known": known[:, 0, 0],
        "current_event_boundary_valid": torch.ones_like(known[:, 0, 0]),
        "future_event_truth": events[:, 0, 1:], "future_event_known": known[:, 0, 1:],
        "future_event_boundary_valid": torch.ones_like(known[:, 0, 1:]),
        "sigreg_x": sx, "sigreg_valid": torch.ones_like(sx, dtype=torch.bool),
        "sigreg_target": sy, "sigreg_target_valid": torch.ones_like(sy, dtype=torch.bool),
        "sigreg_geometry": geom.expand(sb, -1, -1, -1),
        "sigreg_calendar": cal.expand(sb, -1, -1),
        "sigreg_future_calendar": future.expand(sb, -1, -1),
    }
    if paired:
        u = normalized(records, rows, "changed")
        action = records["action"][rows]
        a = torch.zeros(b, 32)
        a[torch.arange(b), ACTION_COLUMNS[action // 4]] = torch.where(action % 2 == 0, -1., 1.) * torch.where(action % 4 < 2, .05, .10)
        batch.update({
            "changed_target": u, "changed_target_valid": torch.ones_like(u, dtype=torch.bool),
            "action": a, "event_scale": data["statistics"]["event_scale"].float(),
            "changed_current_event_truth": events[:, 1, 0],
            "changed_current_event_known": known[:, 1, 0],
            "changed_current_event_boundary_valid": torch.ones_like(known[:, 1, 0]),
            "changed_event_truth": events[:, 1, 1:], "changed_event_known": known[:, 1, 1:],
            "changed_event_boundary_valid": torch.ones_like(known[:, 1, 1:]),
        })
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
