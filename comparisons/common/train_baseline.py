"""Ocean baseline training with periodic validation."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines import OceanBaseline
from src.data import GlobalStateDataset
from src.metrics import add_field_metrics, finalize_field_metrics
from run_metadata import data_contract, public_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["unet", "convlstm", "fno", "pde_transformer_mse", "neuralom", "dpot_pretrained", "dpot_scratch", "poseidon_pretrained", "poseidon_scratch", "pde_transformer_pretrained"], required=True)
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
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint_last.pt in --output")
    parser.add_argument("--pretrained-checkpoint", type=Path)
    return parser.parse_args()


def masked_loss(prediction, target, valid, area):
    weight = valid * area[:, None]
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1.0)


def clip_grad_norm_finite(parameters, max_norm):
    """Compute the global L2 norm in float64, fail on non-finite gradients, and clip."""

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.zeros((), dtype=torch.float64)
    norms = [torch.linalg.vector_norm(gradient.detach(), ord=2, dtype=torch.float64) for gradient in gradients]
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    if not torch.isfinite(total_norm):
        raise FloatingPointError("non-finite gradient norm")
    coefficient = min(1.0, float(max_norm / (total_norm.item() + 1e-12)))
    if coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(coefficient)
    return total_norm


def validate(model, loader, dataset, device, max_batches=None):
    model.eval()
    totals = {"model": __import__("src.metrics", fromlist=["empty_metric_totals"]).empty_metric_totals(dataset.horizon, 5), "persistence": __import__("src.metrics", fromlist=["empty_metric_totals"]).empty_metric_totals(dataset.horizon, 5)}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            prediction = model.rollout(batch["x"], batch["valid"], batch["geometry"], batch["calendar"], batch["future_calendar"])
            persistence = batch["x"][:, -1:].expand_as(batch["target"])
            add_field_metrics(totals["model"], prediction, batch["target"], batch["target_valid"], batch["area"], torch.as_tensor(dataset.std.reshape(-1), device=device), unit_ids=batch["origin_index"])
            add_field_metrics(totals["persistence"], persistence, batch["target"], batch["target_valid"], batch["area"], torch.as_tensor(dataset.std.reshape(-1), device=device), unit_ids=batch["origin_index"])
            batches += 1
    complete = max_batches is None or batches == len(loader)
    summary = finalize_field_metrics(totals["model"], totals["persistence"])
    values = [value / float(dataset.std.reshape(-1)[channel] ** 2) for row in summary["mse"] for channel, value in enumerate(row) if value is not None]
    selector = sum(values) / len(values) if complete and values else None
    return {"complete": complete, "batches": batches, "selector": selector, "metrics": summary}


def checkpoint(model, optimizer, step, validation, path, run_config):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "validation": validation, "run_config": run_config}, path)


def recover_best(output):
    best_score, best_step = None, None
    path = output / "validation.jsonl"
    if not path.is_file():
        return best_score, best_step
    for line in path.read_text().splitlines():
        row = json.loads(line)
        score = row.get("selector")
        if score is not None and (best_score is None or score < best_score):
            best_score, best_step = score, int(row["step"])
    return best_score, best_step


def main():
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output}")
    if args.validation_every < 1 or args.steps < 1:
        raise ValueError("steps and validation-every must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train_data = GlobalStateDataset(args.release_root, "train", "main", False)
    validation_data = GlobalStateDataset(args.release_root, "validation", "main", False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = OceanBaseline(args.method, pretrained_checkpoint=str(args.pretrained_checkpoint) if args.pretrained_checkpoint else None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_config = {"args": public_args(args), "method": args.method, "data_contract": data_contract(train_data), "model_parameters": sum(parameter.numel() for parameter in model.parameters())}
    manifest_path = args.output / "manifest.json"
    if args.resume:
        checkpoint_path = args.output / "checkpoint_last.pt"
        if not checkpoint_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("--resume requires manifest.json and checkpoint_last.pt")
        saved = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
        best_score, best_step = recover_best(args.output)
    else:
        manifest_path.write_text(json.dumps(run_config, indent=2, default=str) + "\n")
        start_step, best_score, best_step = 0, None, None
    stream = iter(train_loader)
    started = time.perf_counter()
    log_mode = "a" if args.resume else "w"
    validation_mode = "a" if args.resume else "w"
    with (args.output / "training.jsonl").open(log_mode) as log, (args.output / "validation.jsonl").open(validation_mode) as validation_log:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iter(train_loader)
                batch = next(stream)
            model.train()
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            prediction = model.rollout(batch["x"], batch["valid"], batch["geometry"], batch["calendar"], batch["future_calendar"])
            loss = masked_loss(prediction, batch["target"], batch["target_valid"], batch["area"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = clip_grad_norm_finite(model.parameters(), 1.0)
            optimizer.step()
            row = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm), "elapsed_seconds": time.perf_counter() - started}
            if step % args.validation_every == 0 or step == args.steps:
                validation = validate(model, validation_loader, validation_data, device, args.max_validation_batches)
                row["validation"] = validation
                validation_log.write(json.dumps({"step": step, **validation}, allow_nan=False) + "\n")
                validation_log.flush()
                if validation["selector"] is not None and (best_score is None or validation["selector"] < best_score):
                    best_score, best_step = validation["selector"], step
                    checkpoint(model, optimizer, step, validation, args.output / "checkpoint_best.pt", run_config)
                checkpoint(model, optimizer, step, validation, args.output / "checkpoint_last.pt", run_config)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 10 == 0:
                log.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
    report = {"status": "completed", "method": args.method, "steps": args.steps, "best_step": best_step, "best_selector": best_score, "elapsed_seconds": time.perf_counter() - started, "validation_complete": args.max_validation_batches is None}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
