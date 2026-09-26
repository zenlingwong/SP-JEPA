#!/usr/bin/env python3
"""Evaluate an ocean-model checkpoint on an explicit data split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.ocean_event import OceanSPJEPA, calendar_month
from model.sp_jepa import SPJEPA, SPJEPASpec
from src.ocean_training import collate_ocean_batch
from src.unified_ocean import UnifiedOceanDataset


GATE_CLOSED = [0, 1, 4, 5, 10, 11, 12]
TC_SAFE = torch.ones(13, dtype=torch.bool)
TC_SAFE[1] = False


def encode_history(model, batch, stage, event_climatology, anomaly_std):
    state = model.encode(batch["x"], batch["valid"], batch["geometry"], batch["calendar"])
    if stage == 2:
        state["cal"] = batch["calendar"]
        truth = torch.nan_to_num(batch["event_truth"][:, :12].float())
        known = batch["event_known"][:, :12].bool()
        k13 = torch.cat((known[..., :5].repeat_interleave(2, -1),
                         known[..., 5:6].expand(*known.shape[:-1], 3)), -1)
        anomaly = torch.where(k13, (truth - event_climatology[calendar_month(batch["calendar"])]) / anomaly_std,
                              torch.zeros_like(truth)) * TC_SAFE.to(truth.device)
        state["h"] = model.fuse(state["h"], torch.cat((anomaly, known.float()), -1))
    return state


@torch.no_grad()
def persistence_reference(dataset, stats, device):
    climatology = stats["field_climatology"].to(device)
    mse, acc = [], []
    for start in range(0, len(dataset), 7):
        batch = collate_ocean_batch(dataset, list(range(start, min(start + 7, len(dataset)))), device)
        future = climatology[calendar_month(batch["future_calendar"])]
        now = climatology[calendar_month(batch["calendar"][:, -1])][:, None]
        prediction = batch["x"][:, -1:] + future - now
        weight = batch["target_valid"].float() * batch["area"][:, None]
        mse.append(((((prediction - batch["target"]).square() * weight).sum((-2, -1))) /
                    weight.sum((-2, -1))).cpu())
        a = torch.where(weight > 0, prediction - future, 0)
        b = torch.where(weight > 0, batch["target"] - future, 0)
        a = a - (a * weight).sum((-2, -1), keepdim=True) / weight.sum((-2, -1), keepdim=True)
        b = b - (b * weight).sum((-2, -1), keepdim=True) / weight.sum((-2, -1), keepdim=True)
        acc.append(((weight * a * b).sum((-2, -1)) /
                    ((weight * a * a).sum((-2, -1)) * (weight * b * b).sum((-2, -1))).sqrt()).cpu())
    mse, acc = torch.cat(mse).double(), torch.cat(acc).double()
    return {"field": float(mse.mean()), "nRMSE": float(mse.mean().sqrt()),
            "nRMSE_by_lead": mse.mean((0, 2)).sqrt().tolist(),
            "ACC": float(acc.mean()), "ACC_by_lead": acc.mean((0, 2)).tolist()}


def representation_stats(h):
    h = h.reshape(-1, h.shape[-1]).double()
    centred = h - h.mean(0)
    singular = torch.linalg.svdvals(centred)
    probability = singular.square() / singular.square().sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))
    return {"h_std_mean": float(centred.std(0).mean()), "h_effective_rank": effective_rank}


@torch.no_grad()
def evaluate_model(model, dataset, stage, stats, device, keep=False):
    """Reproduce area-weighted field and per-family event metrics of the paper run."""
    was_training = model.training
    model.eval()
    field_climatology = stats["field_climatology"].to(device)
    event_climatology = stats["event_climatology"].to(device)
    anomaly_std = stats["event_anomaly_std"].to(device)
    mse, bias, acc, current = [], [], [], []
    h_history, h_prediction, coefficient_step = [], [], []
    kept = {key: [] for key in ("pred", "a_hist", "h_hist", "a_roll", "h_roll", "events")}
    event_rows, gated_rows = [], []
    origin_events = {False: [], True: []}
    for start in range(0, len(dataset), 7):
        batch = collate_ocean_batch(dataset, list(range(start, min(start + 7, len(dataset)))), device)
        state = encode_history(model, batch, stage, event_climatology, anomaly_std)
        roll = model.rollout(state, batch["calendar"], batch["future_calendar"])
        pred = model.read_fields(roll, batch["geometry"])
        now = model.read_fields({"a": state["a"][:, -1:], "h": state["h"][:, -1:]}, batch["geometry"])
        weight = batch["target_valid"].float() * batch["area"][:, None]
        reduce = lambda err: ((err * weight).sum((-2, -1)) / weight.sum((-2, -1))).cpu()
        clim = field_climatology[calendar_month(batch["future_calendar"])]

        def correlation(output):
            a = torch.where(weight > 0, output - clim, 0)
            b = torch.where(weight > 0, batch["target"] - clim, 0)
            a = a - (a * weight).sum((-2, -1), keepdim=True) / weight.sum((-2, -1), keepdim=True)
            b = b - (b * weight).sum((-2, -1), keepdim=True) / weight.sum((-2, -1), keepdim=True)
            return ((weight * a * b).sum((-2, -1)) /
                    ((weight * a * a).sum((-2, -1)) * (weight * b * b).sum((-2, -1))).sqrt()).cpu()

        mse.append(reduce((pred - batch["target"]).square()))
        bias.append(reduce(pred - batch["target"]))
        acc.append(correlation(pred))
        wcurrent = batch["valid"][:, -1:].float() * batch["area"][:, None]
        current.append((((now - batch["x"][:, -1:]).square() * wcurrent).sum((-2, -1)) /
                        wcurrent.sum((-2, -1))).cpu())
        h_history.append(state["h"].reshape(-1, state["h"].shape[-1]).cpu())
        h_prediction.append(roll["h"].reshape(-1, roll["h"].shape[-1]).cpu())
        sequence = torch.cat((state["a"][:, -1:], roll["a"]), 1)
        coefficient_step.append((sequence[:, 1:] - sequence[:, :-1]).norm(dim=-1).cpu())
        if keep:
            kept["pred"].append(pred.half().cpu())
            kept["a_hist"].append(state["a"].cpu())
            kept["h_hist"].append(state["h"].cpu())
            kept["a_roll"].append(roll["a"].cpu())
            kept["h_roll"].append(roll["h"].cpu())
            kept["events"].append(roll["events"].cpu())
        truth = batch["event_truth"][:, 12:]
        known = batch["event_known"][:, 12:].bool()
        raw_outputs = [(False, roll["events"], event_rows)]
        if stage == 2:
            saved_gate = model.event_gate.clone()
            model.event_gate[GATE_CLOSED] = 0
            gated = model.rollout(state, batch["calendar"], batch["future_calendar"])["events"]
            model.event_gate.copy_(saved_gate)
            raw_outputs.append((True, gated, gated_rows))
        for is_gated, raw, sink in raw_outputs:
            pairs = raw[..., :10].reshape(*raw.shape[:-1], 5, 2).sigmoid().sort(-1).values
            error = torch.cat((.5 * (pairs - truth[..., :10].reshape_as(pairs)).square().sum(-1),
                               (raw[..., 10:].softmax(-1) - truth[..., 10:]).square().sum(-1)[..., None]), -1).double()
            summed = torch.where(known, error, torch.zeros_like(error))
            sink.append((summed.sum((0, 1)).cpu(), known.sum((0, 1)).double().cpu()))
            origin_events[is_gated].append((summed.sum(1).cpu(), known.sum(1).double().cpu()))
    model.train(was_training)
    mse, bias, acc, current = (torch.cat(rows).double() for rows in (mse, bias, acc, current))
    result = {"field": float(mse.mean()), "nRMSE": float(mse.mean().sqrt()),
              "nRMSE_by_lead": mse.mean((0, 2)).sqrt().tolist(),
              "lead_channel_mse": mse.mean(0).tolist(),
              "ACC": float(acc.mean()), "ACC_by_lead": acc.mean((0, 2)).tolist(),
              "ACC_lead_channel": acc.mean(0).tolist(),
              "abs_bias": float(bias.mean(0).abs().mean()),
              "current_reconstruction_nRMSE": float(current.mean().sqrt()),
              "origin_field_mse": mse.mean((1, 2)).tolist(),
              "origin_nRMSE_by_lead": mse.mean(2).sqrt().tolist()}
    for name, rows, gated in (("events", event_rows, False), ("events_gated", gated_rows, True)):
        if not rows:
            continue
        total = sum(row[0] for row in rows)
        count = sum(row[1] for row in rows)
        result[name] = (total / count.clamp_min(1)).tolist()
        per_origin = origin_events[gated]
        result[f"origin_{name}_sum"] = torch.cat([row[0] for row in per_origin]).tolist()
        result[f"origin_{name}_count"] = torch.cat([row[1] for row in per_origin]).tolist()
    result.update({f"hist_{key}": value for key, value in representation_stats(torch.cat(h_history)).items()})
    result.update({f"pred_{key}": value for key, value in representation_stats(torch.cat(h_prediction)).items()})
    result["delta_a_norm_by_lead"] = torch.cat(coefficient_step).mean(0).tolist()
    if stage == 2:
        result["fusion_weight_norm"] = float(model.ev_obs.weight.norm())
    if keep:
        result["_kept"] = {key: torch.cat(value).numpy() for key, value in kept.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="optional NumPy archive of validation predictions and states")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stage = checkpoint["stage"]
    spec = SPJEPASpec(**checkpoint["spec"])
    dictionary = checkpoint["model"]["A"]
    if stage == 1:
        model = SPJEPA(spec, dictionary, None, event_feedback=True, event_core_gradient=True)
    else:
        model = OceanSPJEPA(spec, dictionary, None, checkpoint["statistics"]["event_climatology"])
        model.add_fusion_layer()
    model.load_state_dict(checkpoint["model"])
    model = model.to(args.device)
    settings = checkpoint["data"]
    train = UnifiedOceanDataset(args.data_root, split="train", fold=settings["fold"], purged=settings["purged"])
    dataset = UnifiedOceanDataset(args.data_root, split=args.split, fold=settings["fold"],
                                  purged=settings["purged"], event_calibration=train.event_calibration)
    metrics = evaluate_model(model, dataset, stage, checkpoint["statistics"], args.device,
                             keep=args.predictions is not None)
    if args.predictions is not None:
        np.savez_compressed(args.predictions, origins=np.asarray(dataset.origins), **metrics.pop("_kept"))
    baseline = persistence_reference(dataset, checkpoint["statistics"], args.device)
    args.output.write_text(json.dumps({"split": args.split, "stage": stage, "step": checkpoint["step"],
                                       "metrics": metrics, "persistence": baseline}, indent=2))
    print(json.dumps({"split": args.split, "stage": stage, "nRMSE": metrics["nRMSE"]}))


if __name__ == "__main__":
    main()
