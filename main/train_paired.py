#!/usr/bin/env python3
"""Train on the paired environment: 1,000 natural and 5,000 paired updates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from model.sp_jepa import SPJEPA, SPJEPASpec
from src.paired_data import lattice_dictionary, load_dataset, model_batch
from src.paired_metrics import evaluate
from src.unified_protocol import core_parameter_groups, data_priority_backward, true_response_energy, world_losses


def action_name(action: int) -> str:
    amplitude = .05 if action % 4 < 2 else .10
    return f"common16_c{action // 4}_{'minus' if action % 2 == 0 else 'plus'}_{amplitude:.2f}"


def train(data_path: Path, output: Path, *, device: str, warmup_steps: int,
          paired_steps: int, validation_every: int):
    data = load_dataset(data_path)
    output.mkdir(parents=True, exist_ok=False)
    seed = int(data["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    swap_rng = np.random.default_rng(seed + 1000)
    spec = SPJEPASpec.paired()
    model = SPJEPA(spec, lattice_dictionary(data["statistics"]["std"]), None,
                        event_feedback=True, event_core_gradient=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    roots = data["roots"]
    pairs = data["pairs"]
    by_mechanism = {m: np.where(roots["mechanism"].numpy() == m)[0].tolist()
                    for m in sorted(set(roots["mechanism"].tolist()))}
    pair_keys = sorted(range(len(pairs["history"]) // 2),
                       key=lambda i: f"pair{int(pairs['pair'][2*i])}:{action_name(int(pairs['action'][2*i]))}")

    def natural_batch():
        mechanisms = rng.permutation(list(by_mechanism))[:8]
        chosen = [int(rng.choice(by_mechanism[int(m)])) for m in mechanisms]
        histories = roots["history"][chosen].clone()
        donors = [int(swap_rng.choice([j for j in by_mechanism[int(m)] if j != root]))
                  for m, root in zip(mechanisms, chosen)]
        for k, donor in enumerate(donors):
            if swap_rng.random() < .5:
                histories[k, :-1] = roots["history"][donor, :-1]
        return model_batch(data, chosen, paired=False, sig_rows=chosen, histories=histories, device=device)

    def paired_batch(keys):
        rows = [row for key in keys for row in (2 * key, 2 * key + 1)]
        histories = pairs["history"][rows].clone()
        for k in range(0, len(rows), 2):
            if swap_rng.random() < .5:
                histories[k, :-1] = pairs["history"][rows[k + 1], :-1]
                histories[k + 1, :-1] = pairs["history"][rows[k], :-1]
        sig_rows = [int(pairs["mechanism"][2 * key]) for key in keys]
        # Use the lower root ID of each pair for SIGReg.
        sig_rows = [min(int(pairs["mechanism"][2 * key]) * (2 if data["smoke"] else 16) +
                        int(pairs["initial"][2 * key]),
                        int(pairs["mechanism"][2 * key + 1]) * (2 if data["smoke"] else 16) +
                        int(pairs["initial"][2 * key + 1])) for key in keys]
        return model_batch(data, rows, paired=True, sig_rows=sig_rows, histories=histories, device=device)

    def sampled_keys():
        selected, mechanisms, roots_used = [], set(), set()
        for index in rng.permutation(len(pair_keys)):
            key = pair_keys[int(index)]
            mechanism = int(pairs["mechanism"][2 * key])
            initials = {(mechanism, int(pairs["initial"][2 * key])),
                        (mechanism, int(pairs["initial"][2 * key + 1]))}
            if mechanism in mechanisms or initials & roots_used:
                continue
            selected.append(key); mechanisms.add(mechanism); roots_used.update(initials)
            if len(selected) == 8:
                return selected
        raise ValueError("fewer than eight independent pair-actions")

    def update(batch, response_scale=None, lambda_response=.1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = world_losses(model, batch, response_scale=response_scale, lambda_response=lambda_response)
        data_priority_backward(model, losses["data_total"], losses["event_total"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.ndim == 0}

    with (output / "train.jsonl").open("w") as log:
        for step in range(1, warmup_steps + 1):
            row = update(natural_batch())
            if step % 10 == 0 or step == warmup_steps:
                log.write(json.dumps({"phase": "warmup", "step": step, **row}) + "\n"); log.flush()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": warmup_steps}, output / "warmup.pt")

    # Reset paired-stage streams and consume model-initialization draws.
    rng = np.random.default_rng(seed)
    swap_rng = np.random.default_rng(seed + 1000)
    torch.manual_seed(seed)
    _initialization = SPJEPA(spec, lattice_dictionary(data["statistics"]["std"]), None,
                                  event_feedback=True, event_core_gradient=True)
    del _initialization

    # Calibrate on the first 32 sorted query keys, including history-swap draws.
    calibration_keys = pair_keys[:min(32, len(pair_keys))]
    state_rng = torch.get_rng_state()
    response_scale = true_response_energy(paired_batch(calibration_keys))
    f_params = core_parameter_groups(model)["F"]
    ratios = []
    for start in range(0, len(calibration_keys), 8):
        keys = calibration_keys[start:start + 8]
        losses = world_losses(model, paired_batch(keys), response_scale=response_scale)
        g_field = torch.autograd.grad(losses["field"], f_params, retain_graph=True, allow_unused=True)
        g_response = torch.autograd.grad(losses["response_field"] + losses["response_event"],
                                         f_params, allow_unused=True)
        norm = lambda grads: float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
        ratios.append(norm(g_response) / norm(g_field))
    lambda_response = float(.25 / np.mean(ratios))
    torch.set_rng_state(state_rng)
    model.zero_grad(set_to_none=True)
    baseline = evaluate(model, data, device=device, endpoints=False)
    history = [{"step": 0, **baseline}]
    best = None
    with (output / "train.jsonl").open("a") as log:
        for step in range(1, paired_steps + 1):
            row = update(paired_batch(sampled_keys()), response_scale, lambda_response)
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError(f"non-finite loss at paired step {step}")
            if step % 10 == 0 or step == paired_steps:
                log.write(json.dumps({"phase": "paired", "step": step, **row}) + "\n"); log.flush()
            if step % validation_every == 0 or step == paired_steps:
                scores = evaluate(model, data, device=device, endpoints=False)
                history.append({"step": step, **scores})
                if best is None or scores["selection_score"] < best["selection_score"]:
                    best = {"step": step, "selection_score": scores["selection_score"]}
                    torch.save({"model": model.state_dict(), "step": step, "spec": vars(spec),
                                "data_smoke": bool(data["smoke"]), "warmup_steps": warmup_steps,
                                "paired_steps": paired_steps}, output / "best.pt")
                print(json.dumps({"step": step, **scores}), flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model"])
    report = {"seed": seed, "smoke": bool(data["smoke"]),
              "registered_training_budget": bool(not data["smoke"] and warmup_steps == 1000 and paired_steps == 5000),
              "warmup_steps": warmup_steps,
              "paired_steps": paired_steps, "validation_every": validation_every,
              "response_scale": response_scale, "response_weight": lambda_response,
              "calibration_ratios": ratios, "history": history, "best": best,
              "best_evaluation": evaluate(model, data, device=device, endpoints=True)}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"best": best, "RS": report["best_evaluation"]["RS"],
                      "LCS": report["best_evaluation"]["LCS"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--paired-steps", type=int, default=5000)
    parser.add_argument("--validation-every", type=int, default=250)
    args = parser.parse_args()
    train(args.data, args.output, device=args.device, warmup_steps=args.warmup_steps,
          paired_steps=args.paired_steps, validation_every=args.validation_every)
