#!/usr/bin/env python3
"""Score a paired-environment checkpoint on selection, calibration, or held-out test truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.sp_jepa import SPJEPA, SPJEPASpec
from prepare_paired import pack
from src.paired_data import lattice_dictionary, load_dataset
from src.paired_metrics import evaluate, field_calibration
from src.paired_pulse import SHAPE, action_bank, changed_future, generate_root, mechanism_parameters


def build_bank(data: dict, split: str):
    actions = action_bank()
    histories, factual, changed, action_ids, mechanisms, initials, pairs = [], [], [], [], [], [], []
    n_mechanisms, n_initials = SHAPE[split]
    for mechanism in range(n_mechanisms):
        rule = mechanism_parameters(split, mechanism)
        for initial in range(n_initials):
            h, f = generate_root(split, mechanism, initial)
            for action in actions:
                histories.append(h); factual.append(f); changed.append(changed_future(h, rule, action))
                action_ids.append(action.index); mechanisms.append(mechanism); initials.append(initial); pairs.append(-1)
    return pack(histories, factual, changed, action_ids, mechanisms, initials, pairs,
                data["statistics"], data["epsilon_num"])


def run(data_path: Path, checkpoint: Path, output: Path, *, split: str, device: str, batch_size: int):
    data = load_dataset(data_path)
    if "calibration" not in data:
        data["calibration"] = build_bank(data, "validation-calibration")
    if split == "validation-calibration":
        data["selection"] = data["calibration"]
    elif split != "validation-selection":
        data["selection"] = build_bank(data, split)
    model = SPJEPA(SPJEPASpec.paired(), lattice_dictionary(data["statistics"]["std"]), None,
                        event_feedback=True, event_core_gradient=True).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    field = evaluate(model, data, device=device, batch_size=batch_size, endpoints=True)
    calibration = field_calibration(model, data, device=device, batch_size=batch_size)
    registered_training = (payload.get("warmup_steps") == 1000 and payload.get("paired_steps") == 5000)
    qualified = bool(field["response_resolved"] and calibration["field_qualified"]
                     and not data["smoke"] and registered_training)
    result = {"split": split, "checkpoint_step": int(payload["step"]), "smoke_data": bool(data["smoke"]),
              **field, "field_calibration": calibration,
              "registered_training_budget": registered_training,
              "field_qualification_status": "passed" if qualified else "unqualified",
              "field_LCS_qualified": field["LCS"] if qualified else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(output), "split": split, "RS": result["RS"],
                      "LCS": result["LCS"], "field_qualification_status": result["field_qualification_status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation-selection", "validation-calibration", "test"),
                        default="validation-selection")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.data, args.checkpoint, args.output, split=args.split, device=args.device, batch_size=args.batch_size)
