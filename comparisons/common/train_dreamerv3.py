#!/usr/bin/env python3
"""Auditable W1 DreamerV3-RSSM Ocean forecasting adaptation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.dreamerv3 import OceanDreamerV3
from src.data import GlobalStateDataset
from src.metrics import add_field_metrics, empty_metric_totals, finalize_field_metrics
from train_baseline import clip_grad_norm_finite
from run_metadata import data_contract, public_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--deter", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--stoch", type=int, default=32)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--unimix", type=float, default=0.01)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--dynamics-scale", type=float, default=1.0)
    parser.add_argument("--representation-scale", type=float, default=0.1)
    parser.add_argument("--validation-sampling-seed", type=int, default=20260920)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def validate(
    model: OceanDreamerV3,
    loader: DataLoader,
    dataset: GlobalStateDataset,
    device: torch.device,
    sampling_seed: int,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    model_totals = empty_metric_totals(dataset.horizon, 5)
    persistence_totals = empty_metric_totals(dataset.horizon, 5)
    generator = make_generator(device, sampling_seed)
    batches = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            prediction = model.rollout(
                batch["x"],
                batch["valid"],
                batch["geometry"],
                batch["future_calendar"],
                calendar=batch["calendar"],
                generator=generator,
            )
            persistence = batch["x"][:, -1:].expand_as(batch["target"])
            std = torch.as_tensor(dataset.std.reshape(-1), device=device)
            add_field_metrics(
                model_totals,
                prediction,
                batch["target"],
                batch["target_valid"],
                batch["area"],
                std,
                unit_ids=batch["origin_index"],
            )
            add_field_metrics(
                persistence_totals,
                persistence,
                batch["target"],
                batch["target_valid"],
                batch["area"],
                std,
                unit_ids=batch["origin_index"],
            )
            batches += 1
    complete = max_batches is None or batches == len(loader)
    summary = finalize_field_metrics(model_totals, persistence_totals)
    values = [
        value / float(dataset.std.reshape(-1)[channel] ** 2)
        for row in summary["mse"]
        for channel, value in enumerate(row)
        if value is not None
    ]
    selector = sum(values) / len(values) if complete and values else None
    return {
        "complete": complete,
        "batches": batches,
        "selector": selector,
        "sampling_seed": sampling_seed,
        "metrics": summary,
    }


def checkpoint(
    model: OceanDreamerV3,
    optimizer: torch.optim.Optimizer,
    step: int,
    validation: dict[str, object],
    path: Path,
    run_config: dict[str, object],
    model_init: dict[str, object],
    model_kwargs: dict[str, object],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "validation": validation,
        "run_config": run_config,
        "model_init": model_init,
        "model_kwargs": model_kwargs,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def recover_best(output: Path) -> tuple[float | None, int | None]:
    best_score, best_step = None, None
    path = output / "validation.jsonl"
    if not path.is_file():
        return best_score, best_step
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("selector") is not None and (best_score is None or row["selector"] < best_score):
            best_score, best_step = float(row["selector"]), int(row["step"])
    return best_score, best_step


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output}")
    if args.steps <= 0 or args.validation_every <= 0:
        raise ValueError("steps and validation-every must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train_data = GlobalStateDataset(args.release_root, "train", "main", False)
    validation_data = GlobalStateDataset(args.release_root, "validation", "main", False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    channels, height, width = train_data[0]["x"].shape[1:]
    model_init = {"channels": channels, "height": height, "width": width}
    model_kwargs = {
        "deter": args.deter,
        "hidden": args.hidden,
        "stoch": args.stoch,
        "classes": args.classes,
        "blocks": args.blocks,
        "depth": args.depth,
        "unimix": args.unimix,
        "free_nats": args.free_nats,
        "dynamics_scale": args.dynamics_scale,
        "representation_scale": args.representation_scale,
    }
    model = OceanDreamerV3(**model_init, **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_config = {
        "args": public_args(args),
        "method": "dreamerv3_rssm_forecasting_adaptation",
        "adaptation_status": "official_source_pytorch_port",
        "official_preset": "size12m",
        "excluded_components": ["actor", "critic", "reward_head", "continuation_head", "environment_interaction"],
        "data_contract": data_contract(train_data),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_init": model_init,
        "model_kwargs": model_kwargs,
        "loss_scheme": "dreamerv3_world_model_rec1_dyn1_rep0.1",
        "forecasting_contract": "posterior_history_then_prior_only_three_step_rollout",
        "test_evaluated": False,
    }
    manifest_path = args.output / "manifest.json"
    if args.resume:
        saved = torch.load(args.output / "checkpoint_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state_all" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        start_step = int(saved["step"])
        best_score, best_step = recover_best(args.output)
    else:
        manifest_path.write_text(json.dumps(run_config, indent=2, default=str) + "\n")
        start_step, best_score, best_step = 0, None, None
    stream = iter(train_loader)
    started = time.perf_counter()
    with (args.output / "training.jsonl").open("a" if args.resume else "w") as log, (
        args.output / "validation.jsonl"
    ).open("a" if args.resume else "w") as validation_log:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iter(train_loader)
                batch = next(stream)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            model.train()
            loss, terms = model.world_model_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = clip_grad_norm_finite(model.parameters(), 1.0)
            optimizer.step()
            row = {
                "step": step,
                **{key: float(value.detach()) for key, value in terms.items()},
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            if step % args.validation_every == 0 or step == args.steps:
                validation = validate(
                    model,
                    validation_loader,
                    validation_data,
                    device,
                    args.validation_sampling_seed,
                    args.max_validation_batches,
                )
                row["validation"] = validation
                validation_log.write(json.dumps({"step": step, **validation}, allow_nan=False) + "\n")
                validation_log.flush()
                if validation["selector"] is not None and (best_score is None or validation["selector"] < best_score):
                    best_score, best_step = float(validation["selector"]), step
                    checkpoint(
                        model,
                        optimizer,
                        step,
                        validation,
                        args.output / "checkpoint_best.pt",
                        run_config,
                        model_init,
                        model_kwargs,
                    )
                checkpoint(
                    model,
                    optimizer,
                    step,
                    validation,
                    args.output / "checkpoint_last.pt",
                    run_config,
                    model_init,
                    model_kwargs,
                )
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 10 == 0:
                log.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
    report = {
        "status": "completed",
        "method": "dreamerv3_rssm_forecasting_adaptation",
        "steps": args.steps,
        "best_step": best_step,
        "best_selector": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_complete": args.max_validation_batches is None,
        "test_evaluated": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
