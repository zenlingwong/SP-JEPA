"""ClimaX adaptation for ocean forecasting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.climax import ClimaX, masked_area_mse
from src.data import GlobalStateDataset
from src.metrics import add_field_metrics, empty_metric_totals, finalize_field_metrics
from train_baseline import clip_grad_norm_finite
from run_metadata import data_contract, public_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--pretrained-path", type=Path)
    parser.add_argument("--internal-height", type=int, default=32)
    parser.add_argument("--internal-width", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate(model, loader, dataset, device):
    model.eval()
    totals = empty_metric_totals(dataset.horizon, 5)
    persistence_totals = empty_metric_totals(dataset.horizon, 5)
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            prediction = model.rollout(batch["x"], batch["valid"])
            prediction = torch.nn.functional.interpolate(
                prediction.flatten(0, 1), size=batch["target"].shape[-2:], mode="bilinear", align_corners=False
            ).reshape_as(batch["target"])
            persistence = batch["x"][:, -1:].expand_as(batch["target"])
            std = torch.as_tensor(dataset.std.reshape(-1), device=device)
            add_field_metrics(totals, prediction, batch["target"], batch["target_valid"], batch["area"], std, unit_ids=batch["origin_index"])
            add_field_metrics(persistence_totals, persistence, batch["target"], batch["target_valid"], batch["area"], std, unit_ids=batch["origin_index"])
    summary = finalize_field_metrics(totals, persistence_totals)
    values = [value / float(dataset.std.reshape(-1)[channel] ** 2) for row in summary["mse"] for channel, value in enumerate(row) if value is not None]
    return {"complete": True, "batches": len(loader), "selector": sum(values) / len(values), "metrics": summary}


def save_checkpoint(model, optimizer, step, validation, path, run_config, model_init):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "validation": validation, "run_config": run_config, "model_init": model_init, "torch_rng_state": torch.get_rng_state()}, path)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train_data = GlobalStateDataset(args.release_root, "train", "main", False)
    validation_data = GlobalStateDataset(args.release_root, "validation", "main", False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    channels, height, width = train_data[0]["x"].shape[1:]
    model_init = {"variables": channels, "img_size": (args.internal_height, args.internal_width)}
    model = ClimaX(**model_init).to(device)
    load_info = None
    if args.pretrained_path is not None and not args.resume:
        load_info = model.load_pretrained_checkpoint(args.pretrained_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    load_counts = {key: int(load_info[key]) for key in ("loaded", "skipped", "missing", "unexpected")} if load_info is not None else None
    run_config = {"args": public_args(args), "method": "climax_ocean_adapter", "data_contract": data_contract(train_data), "model_init": model_init, "model_parameters": sum(p.numel() for p in model.parameters()), "test_evaluated": False, "pretrained_load_counts": load_counts}
    manifest = args.output / "manifest.json"
    if args.resume:
        saved = torch.load(args.output / "checkpoint_last.pt", map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
        best_score = None
        best_step = None
        if (args.output / "validation.jsonl").exists():
            for line in (args.output / "validation.jsonl").read_text().splitlines():
                row = json.loads(line)
                if best_score is None or row["selector"] < best_score:
                    best_score, best_step = row["selector"], row["step"]
    else:
        manifest.write_text(json.dumps(run_config, indent=2) + "\n")
        start_step, best_score, best_step = 0, None, None
    stream = iter(train_loader)
    started = time.perf_counter()
    with (args.output / "training.jsonl").open("a" if args.resume else "w") as log, (args.output / "validation.jsonl").open("a" if args.resume else "w") as vlog:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iter(train_loader)
                batch = next(stream)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            model.train()
            current = torch.nn.functional.interpolate(
                batch["x"][:, -1], size=(args.internal_height, args.internal_width), mode="bilinear", align_corners=False
            )
            predictions = []
            for lead in (1.0, 2.0, 3.0):
                current = model.forward_step(current, lead)
                predictions.append(current)
            prediction = torch.stack(predictions, dim=1)
            prediction = torch.nn.functional.interpolate(
                prediction.flatten(0, 1), size=batch["target"].shape[-2:], mode="bilinear", align_corners=False
            ).reshape_as(batch["target"])
            loss = masked_area_mse(prediction, batch["target"], batch["target_valid"], batch["area"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = clip_grad_norm_finite(model.parameters(), 1.0)
            optimizer.step()
            row = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm), "elapsed_seconds": time.perf_counter() - started}
            if step % args.validation_every == 0 or step == args.steps:
                validation = validate(model, validation_loader, validation_data, device)
                vlog.write(json.dumps({"step": step, **validation}, allow_nan=False) + "\n")
                vlog.flush()
                row["validation"] = validation
                if best_score is None or validation["selector"] < best_score:
                    best_score, best_step = float(validation["selector"]), step
                    save_checkpoint(model, optimizer, step, validation, args.output / "checkpoint_best.pt", run_config, model_init)
                save_checkpoint(model, optimizer, step, validation, args.output / "checkpoint_last.pt", run_config, model_init)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 10 == 0:
                log.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
    report = {"status": "completed", "method": "climax_ocean_adapter", "steps": args.steps, "best_step": best_step, "best_selector": best_score, "elapsed_seconds": time.perf_counter() - started, "validation_complete": True, "test_evaluated": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
