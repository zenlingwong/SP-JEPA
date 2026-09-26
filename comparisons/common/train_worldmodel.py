#!/usr/bin/env python3
"""Factual LeWM-style ocean training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from model.lewm import SIGReg
from model.ocean import OceanLeWM, lewm_loss
from src.data import GlobalStateDataset
from src.metrics import add_field_metrics, empty_metric_totals, finalize_field_metrics
from train_baseline import clip_grad_norm_finite
from run_metadata import data_contract, public_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="lewm", choices=["lewm"])
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--validation-every", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--basis-y", type=int, default=3)
    parser.add_argument("--basis-x", type=int, default=6)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--event-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-temporal-window", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_basis(train_data, basis_y, basis_x):
    sample = train_data[0]
    channels, height, width = sample["x"].shape[1:]
    valid = sample["valid"][-1]
    basis = []
    for channel in range(channels):
        for row in range(basis_y):
            for column in range(basis_x):
                mode = torch.zeros(channels, height, width)
                y0, y1 = row * height // basis_y, (row + 1) * height // basis_y
                x0, x1 = column * width // basis_x, (column + 1) * width // basis_x
                mode[channel, y0:y1, x0:x1] = valid[channel, y0:y1, x0:x1] / float(train_data.std[channel, 0, 0])
                if mode.any():
                    basis.append(mode)
    return torch.stack(basis), sample["area"].expand(channels, height, width) * valid


def make_event_thresholds(train_data):
    """Calibrate EAWM-inspired auxiliary labels using training windows only."""
    changes = []
    masks = []
    for index in torch.linspace(0, len(train_data) - 1, min(64, len(train_data))).long().tolist():
        row = train_data[index]
        values = torch.cat((row["x"], row["target"]))
        valid = torch.cat((row["valid"], row["target_valid"]))
        changes.append(values[1:] - values[:-1])
        masks.append(valid[1:] & valid[:-1])
    changes = torch.cat(changes)
    masks = torch.cat(masks)
    threshold = torch.stack(
        [changes[:, channel][masks[:, channel]].abs().quantile(0.75).clamp_min(1e-6) for channel in range(changes.shape[1])]
    )
    density = (
        ((changes.abs() > threshold[None, :, None, None]) & masks).sum((1, 2, 3)).float()
        / masks.sum((1, 2, 3)).clamp_min(1)
    ).quantile(0.90).clamp_min(1e-6)
    return threshold, density


def validate(model, loader, dataset, device, max_batches=None):
    model.eval()
    model_totals = empty_metric_totals(dataset.horizon, 5)
    persistence_totals = empty_metric_totals(dataset.horizon, 5)
    batches = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            latent = model.rollout(batch["x"], batch["valid"], batch["geometry"], dataset.horizon, batch["calendar"], batch["future_calendar"])
            prediction = model.decode(latent)
            persistence = batch["x"][:, -1:].expand_as(batch["target"])
            std = torch.as_tensor(dataset.std.reshape(-1), device=device)
            add_field_metrics(model_totals, prediction, batch["target"], batch["target_valid"], batch["area"], std, unit_ids=batch["origin_index"])
            add_field_metrics(persistence_totals, persistence, batch["target"], batch["target_valid"], batch["area"], std, unit_ids=batch["origin_index"])
            batches += 1
    complete = max_batches is None or batches == len(loader)
    summary = finalize_field_metrics(model_totals, persistence_totals)
    values = [value / float(dataset.std.reshape(-1)[channel] ** 2) for row in summary["mse"] for channel, value in enumerate(row) if value is not None]
    selector = sum(values) / len(values) if complete and values else None
    return {"complete": complete, "batches": batches, "selector": selector, "metrics": summary}


def checkpoint(model, optimizer, step, validation, path, run_config, basis, weight, model_init, model_kwargs):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "validation": validation, "run_config": run_config, "basis": basis, "weight": weight, "model_init": model_init, "model_kwargs": model_kwargs}, path)


def recover_best(output):
    best_score, best_step = None, None
    path = output / "validation.jsonl"
    if not path.is_file():
        return best_score, best_step
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("selector") is not None and (best_score is None or row["selector"] < best_score):
            best_score, best_step = row["selector"], int(row["step"])
    return best_score, best_step


def main():
    args = parse_args()
    if args.sigreg_temporal_window is not None and args.sigreg_temporal_window < 2:
        raise ValueError("sigreg-temporal-window must be at least 2")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train_data = GlobalStateDataset(args.release_root, "train", "main", False)
    validation_data = GlobalStateDataset(args.release_root, "validation", "main", False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    basis, weight = make_basis(train_data, args.basis_y, args.basis_x)
    event_threshold, event_density_threshold = None, None
    if args.event_weight:
        event_threshold, event_density_threshold = make_event_thresholds(train_data)
        event_threshold = event_threshold.to(device)
        event_density_threshold = event_density_threshold.to(device)
    channels, height, width = train_data[0]["x"].shape[1:]
    model_init = {"channels": channels, "height": height, "width": width, "patch": 6}
    model_kwargs = {"dim": args.dim, "history_size": train_data.history, "predictor_depth": args.depth, "predictor_heads": args.heads, "predictor_mlp_dim": 4 * args.dim, "predictor_dim_head": args.head_dim, "spatial_depth": args.depth, "spatial_heads": args.heads, "projector_hidden": 4 * args.dim, "variant": "structured", "allow_writes": False}
    model = OceanLeWM(**model_init, write_basis=basis, weight=weight, **model_kwargs).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_config = {"args": public_args(args), "method": args.method, "data_contract": data_contract(train_data), "model_parameters": sum(parameter.numel() for parameter in model.parameters()), "model_init": model_init, "model_kwargs": model_kwargs, "loss_scheme": "lewm_factual_v1", "event_threshold": event_threshold.detach().cpu().tolist() if event_threshold is not None else None, "event_density_threshold": float(event_density_threshold.detach().cpu()) if event_density_threshold is not None else None, "test_evaluated": False}
    manifest_path = args.output / "manifest.json"
    if args.resume:
        saved = torch.load(args.output / "checkpoint_last.pt", map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
        best_score, best_step = recover_best(args.output)
    else:
        manifest_path.write_text(json.dumps(run_config, indent=2, default=str) + "\n")
        start_step, best_score, best_step = 0, None, None
    stream = iter(train_loader)
    started = time.perf_counter()
    with (args.output / "training.jsonl").open("a" if args.resume else "w") as log, (args.output / "validation.jsonl").open("a" if args.resume else "w") as validation_log:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iter(train_loader)
                batch = next(stream)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            model.train()
            loss, terms = lewm_loss(model, sigreg, batch, event_threshold=event_threshold, event_density_threshold=event_density_threshold, sigreg_weight=args.sigreg_weight, event_weight=args.event_weight, sigreg_temporal_window=args.sigreg_temporal_window)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = clip_grad_norm_finite(model.parameters(), 1.0)
            optimizer.step()
            row = {"step": step, **{key: float(value.detach()) for key, value in terms.items()}, "gradient_norm": float(gradient_norm), "elapsed_seconds": time.perf_counter() - started}
            if step % args.validation_every == 0 or step == args.steps:
                validation = validate(model, validation_loader, validation_data, device, args.max_validation_batches)
                row["validation"] = validation
                validation_log.write(json.dumps({"step": step, **validation}, allow_nan=False) + "\n")
                validation_log.flush()
                if validation["selector"] is not None and (best_score is None or validation["selector"] < best_score):
                    best_score, best_step = validation["selector"], step
                    checkpoint(model, optimizer, step, validation, args.output / "checkpoint_best.pt", run_config, basis, weight, model_init, model_kwargs)
                checkpoint(model, optimizer, step, validation, args.output / "checkpoint_last.pt", run_config, basis, weight, model_init, model_kwargs)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 10 == 0:
                log.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
    report = {"status": "completed", "method": args.method, "steps": args.steps, "best_step": best_step, "best_selector": best_score, "elapsed_seconds": time.perf_counter() - started, "validation_complete": args.max_validation_batches is None, "test_evaluated": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
