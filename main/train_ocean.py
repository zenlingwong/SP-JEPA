#!/usr/bin/env python3
"""Train the two-stage model on the Global Ocean Carbon–Physics dataset."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_ocean import evaluate_model, persistence_reference
from model.ocean_event import OceanSPJEPA, calendar_month
from model.sp_jepa import SPJEPA, SPJEPASpec, sigreg_v2
from src.ocean_training import NonOverlappingOceanBatchSampler, collate_ocean_batch
from src.unified_ocean import UnifiedOceanDataset
from src.unified_protocol import data_priority_backward, event_loss, masked_field_mse


HISTORY = 12
TC_SAFE = torch.ones(13, dtype=torch.bool)
TC_SAFE[1] = False


def build_basis(train):
    """Noncentered, static-ocean EOFs of training origins, as in the paper run."""
    frames, masks, months = [], [], []
    for i in range(len(train)):
        item = train[i]
        frames.append(item["x"][-1]); masks.append(item["valid"][-1])
        months.append(int(calendar_month(item["calendar"][-1])))
    frames, masks, months = torch.stack(frames).double(), torch.stack(masks), torch.tensor(months)
    area = item["area"].double()
    static = masks.all(0)
    climatology = torch.stack([
        torch.where(masks[months == month], frames[months == month], 0).sum(0)
        / masks[months == month].sum(0).clamp_min(1)
        for month in range(12)
    ])
    ocean = static.reshape(-1).double()
    _, singular, right = torch.linalg.svd(frames.reshape(len(frames), -1) * ocean, full_matrices=False)
    dictionary = (right[:32].T * (singular[:32] / len(frames) ** .5)).float()
    weight = (area.expand_as(static) * static).reshape(-1).float()
    return dictionary, weight, climatology.float()


def event_observation(batch, event_climatology, anomaly_std):
    truth = torch.nan_to_num(batch["event_truth"][:, :HISTORY].float())
    known = batch["event_known"][:, :HISTORY].bool()
    k13 = torch.cat((known[..., :5].repeat_interleave(2, -1),
                     known[..., 5:6].expand(*known.shape[:-1], 3)), -1)
    anomaly = torch.where(k13, (truth - event_climatology[calendar_month(batch["calendar"])]) / anomaly_std,
                          torch.zeros_like(truth)) * TC_SAFE.to(truth.device)
    return torch.cat((anomaly, known.float()), -1)


def encode_history(model, batch, stage, event_climatology=None, anomaly_std=None):
    state = model.encode(batch["x"], batch["valid"], batch["geometry"], batch["calendar"])
    if stage == 2:
        state["cal"] = batch["calendar"]
        state["h"] = model.fuse(state["h"], event_observation(batch, event_climatology, anomaly_std))
    return state


def losses(model, batch, stage, event_climatology=None, anomaly_std=None):
    state = encode_history(model, batch, stage, event_climatology, anomaly_std)
    target = model.encode(batch["target"], batch["target_valid"], batch["geometry"], batch["future_calendar"])
    roll = model.rollout(state, batch["calendar"], batch["future_calendar"])
    fields = model.read_fields(roll, batch["geometry"])
    jepa = F.mse_loss(roll["h"], target["h"])
    field = masked_field_mse(fields, batch["target"], batch["target_valid"], batch.get("area"))
    sigreg = sigreg_v2(torch.cat((state["h"], target["h"]), 1))
    event_raw = torch.cat((model.read_events(state)[:, None], roll["events"]), 1)
    truth = torch.cat((batch["current_event_truth"][:, None], batch["future_event_truth"]), 1)
    known = torch.cat((batch["current_event_known"][:, None], batch["future_event_known"]), 1)
    event = event_loss(event_raw, truth, known, "ocean_factual")
    if stage == 2:
        months = calendar_month(torch.cat((batch["calendar"][:, -1:], batch["future_calendar"]), 1))
        event = event + .01 * (event_raw - model.event_anchor[months]).square().mean()
    return {"data_total": jepa + .09 * sigreg + field,
            "event_total": .02 * event, "jepa": jepa, "field": field,
            "sigreg": sigreg, "event": event}


def save_checkpoint(path, model, stage, step, stats, data_settings):
    torch.save({"model": model.state_dict(), "stage": stage, "step": step,
                "spec": vars(model.spec), "statistics": stats,
                "data": data_settings}, path)


def train_stage(model, stage, train, valid, args, stats, data_settings, event_climatology, anomaly_std):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    sampler = NonOverlappingOceanBatchSampler(train.origins, train.history, train.horizon, args.seed, 8)
    steps = args.steps1 if stage == 1 else args.steps2
    validate_every = args.validate1 if stage == 1 else args.validate2
    prefix = f"stage{stage}"
    train_log = open(args.output / f"{prefix}_train.jsonl", "w")
    valid_log = open(args.output / f"{prefix}_validation.jsonl", "w")
    started = time.time()

    def validate(step):
        metrics = evaluate_model(model, valid, stage, stats, args.device)
        row = {"step": step, "elapsed": time.time() - started, **metrics}
        valid_log.write(json.dumps(row) + "\n"); valid_log.flush()
        return row

    initial = validate(0)
    best = {"step": 0, "field": initial["field"]}
    if args.select == "best":
        save_checkpoint(args.output / f"{prefix}_best.pt", model, stage, 0, stats, data_settings)
    history = [initial]
    for step in range(1, steps + 1):
        batch = collate_ocean_batch(train, sampler.next_batch(), args.device)
        optimizer.zero_grad(set_to_none=True)
        out = losses(model, batch, stage, event_climatology, anomaly_std)
        if stage == 1:
            data_priority_backward(model, out["data_total"], out["event_total"])
        else:
            (out["data_total"] + out["event_total"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0:
            train_log.write(json.dumps({"step": step, **{k: float(v.detach()) for k, v in out.items()},
                                        "elapsed": time.time() - started}) + "\n")
            train_log.flush()
        if step % validate_every == 0 or step == steps:
            row = validate(step)
            history.append(row)
            if args.select == "best" and row["field"] < best["field"]:
                best = {"step": step, "field": row["field"]}
                save_checkpoint(args.output / f"{prefix}_best.pt", model, stage, step, stats, data_settings)
    save_checkpoint(args.output / f"{prefix}_last.pt", model, stage, steps, stats, data_settings)
    tail = [row for row in history if row["step"] >= steps - 2 * validate_every]
    summary = {"endpoint": history[-1], "best": best,
               "last3_field": float(np.mean([row["field"] for row in tail])),
               "last3_nRMSE": float(np.mean([row["nRMSE"] for row in tail])),
               "last3_ACC": float(np.mean([row["ACC"] for row in tail])),
               "seconds": time.time() - started}
    selected_path = args.output / f"{prefix}_{'best' if args.select == 'best' else 'last'}.pt"
    if args.select == "best":
        selected = torch.load(selected_path, map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model"])
        summary["selected"] = {"step": best["step"], **evaluate_model(model, valid, stage, stats, args.device)}
    else:
        summary["selected"] = history[-1]
    train_log.close(); valid_log.close()
    return summary, selected_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--event-statistics", type=Path, required=True,
                        help="JSON with climatology_table [12,13] and anomaly_std [13]")
    parser.add_argument("--fold", default="main")
    parser.add_argument("--purged", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recipe", choices=("main", "development"), default="main",
                        help="main: 28,000+2,000 selected by validation; development: 3,000+2,000 fixed endpoint")
    parser.add_argument("--mode", choices=("pipeline", "stage1", "stage2"), default="pipeline")
    parser.add_argument("--parent", type=Path, help="selected stage-1 checkpoint for stage2 mode")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps1", type=int)
    parser.add_argument("--steps2", type=int)
    parser.add_argument("--validate1", type=int)
    parser.add_argument("--validate2", type=int)
    parser.add_argument("--select", choices=("endpoint", "best"))
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.mode == "stage2" and args.parent is None:
        parser.error("stage2 mode requires --parent")
    defaults = {"main": (28000, 2000, 500, 100, "best"),
                "development": (3000, 2000, 250, 100, "endpoint")}[args.recipe]
    for name, value in zip(("steps1", "steps2", "validate1", "validate2", "select"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train = UnifiedOceanDataset(args.data_root, split="train", fold=args.fold, purged=args.purged)
    valid = UnifiedOceanDataset(args.data_root, split="validation", fold=args.fold, purged=args.purged,
                                event_calibration=train.event_calibration)
    dictionary, weight, field_climatology = build_basis(train)
    event_stats = json.loads(args.event_statistics.read_text())
    event_climatology = torch.as_tensor(event_stats["climatology_table"], dtype=torch.float32, device=device)
    anomaly_std = torch.as_tensor(event_stats["anomaly_std"], dtype=torch.float32, device=device)
    if event_climatology.shape != (12, 13) or anomaly_std.shape != (13,):
        raise ValueError("event statistics require climatology [12,13] and anomaly_std [13]")
    if not torch.isfinite(event_climatology).all() or not torch.isfinite(anomaly_std).all() or (anomaly_std <= 0).any():
        raise ValueError("event statistics must be finite with positive anomaly scales")
    stats = {"field_climatology": field_climatology,
             "event_climatology": event_climatology.cpu(), "event_anomaly_std": anomaly_std.cpu()}
    settings = {"fold": args.fold, "purged": args.purged, "seed": args.seed}
    config = {"mode": args.mode, "recipe": args.recipe, "seed": args.seed, "fold": args.fold, "purged": args.purged,
              "steps1": args.steps1, "steps2": args.steps2, "validate1": args.validate1,
              "validate2": args.validate2, "select": args.select, "batch_size": 8,
              "learning_rate": 3e-4}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    report = {"config": config, "persistence": persistence_reference(valid, stats, device)}
    spec = SPJEPASpec.ocean()
    if args.mode in ("pipeline", "stage1"):
        first = SPJEPA(spec, dictionary, weight, event_feedback=True, event_core_gradient=True).to(device)
        report["stage1_size"] = {"trainable_parameters": sum(p.numel() for p in first.parameters()),
                                 "state_bytes_per_sample": 4 * spec.history * (spec.d_h + spec.rank + 2)}
        report["stage1"], parent_path = train_stage(first, 1, train, valid, args, stats, settings,
                                                       event_climatology, anomaly_std)
        final_model, final_stage = first, 1
    if args.mode in ("pipeline", "stage2"):
        if args.mode == "stage2":
            parent_path = args.parent
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if parent["stage"] != 1:
            raise ValueError("second stage requires a stage-1 checkpoint")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        second = OceanSPJEPA(spec, dictionary, weight, event_stats["climatology_table"])
        missing, unexpected = second.load_state_dict(parent["model"], strict=False)
        if unexpected or set(missing) != {"event_anchor", "event_gate", "event_residual.0.weight",
                                           "event_residual.0.bias", "event_residual.2.weight", "event_residual.2.bias"}:
            raise ValueError(f"stage-1 checkpoint mismatch: {missing}, {unexpected}")
        second.add_fusion_layer()
        second = second.to(device)
        report["stage2_size"] = {"trainable_parameters": sum(p.numel() for p in second.parameters()),
                                 "state_bytes_per_sample": 4 * spec.history * (spec.d_h + spec.rank + 2)}
        report["stage2"], _ = train_stage(second, 2, train, valid, args, stats, settings,
                                           event_climatology, anomaly_std)
        final_model, final_stage = second, 2
    if args.save_predictions:
        kept = evaluate_model(final_model, valid, final_stage, stats, device, keep=True)["_kept"]
        np.savez_compressed(args.output / "validation_predictions.npz", origins=np.asarray(valid.origins), **kept)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": "complete", "selected": args.select, "mode": args.mode}))


if __name__ == "__main__":
    main()
