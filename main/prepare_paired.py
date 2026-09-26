#!/usr/bin/env python3
"""Generate the Paired Transport–Exchange environment's training and evaluation banks.

--smoke generates a smaller dataset for interface checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.paired_data import event_labels
from src.paired_pulse import DT, PROFILE, action_bank, changed_future, fit_statistics, generate_root, mechanism_parameters
from src.unified_protocol import frozen_pair_table


def select_uniform_pairs(table, seed: int):
    """Seed coverage followed by eight uniform 32-key acquisition rounds."""
    actions = action_bank()
    candidates = [(pair, action.index) for pair in range(len(table)) for action in actions]

    def choice(name, values, count):
        local_seed = seed + sum((i + 1) * ord(ch) for i, ch in enumerate(name))
        indices = np.random.default_rng(local_seed).choice(len(values), count, replace=False)
        return [values[int(i)] for i in np.atleast_1d(indices)]

    def covered(name, values, count, cover):
        groups = {}
        for key in values:
            groups.setdefault(int(table[key[0]][3]), []).append(key)
        selected = []
        for group in choice(name + ":groups", sorted(groups), cover):
            selected.extend(choice(name + f":group:{group}", groups[group], 1))
        selected.extend(choice(name + ":fill", [key for key in values if key not in selected], count - len(selected)))
        return selected

    representatives = [(pair, 0) for pair in range(len(table))]
    selected = []
    for pair, _ in covered("seed_pairs", representatives, 16, 8):
        for tier, action_ids in (("small", range(2)), ("large", range(2, 4))):
            pool = [(pair, action) for action in range(16) if action % 4 in action_ids]
            selected.extend(choice(f"seed_action:{pair}:{tier}", pool, 1))
    for round_index in range(8):
        for tier, action_ids in (("small", range(2)), ("large", range(2, 4))):
            pool = [key for key in candidates if key not in selected and key[1] % 4 in action_ids]
            selected.extend(covered(f"acquire:{round_index}:{tier}:uniform", pool, 16, 8))
    if len(selected) != 288 or len(set(selected)) != 288:
        raise ValueError("uniform query schedule did not produce 288 unique pair-actions")
    return selected


def numerical_epsilon(smoke: bool) -> torch.Tensor:
    """Check adjacent 1/8 to 1/16 numerical refinement."""
    fixtures = [(m, i) for m in range(1 if smoke else 8) for i in ((0,) if smoke else (0, 7, 15))]
    actions = action_bank()[:1] if smoke else action_bank()
    epsilon = torch.zeros(2, dtype=torch.float64)
    worst_field = 0.
    worst_response = 0.
    for mechanism_id, initial_id in fixtures:
        mechanism = mechanism_parameters("train", mechanism_id)
        coarse_h, coarse_f = generate_root("train", mechanism_id, initial_id, 1 / 8)
        fine_h, fine_f = generate_root("train", mechanism_id, initial_id, DT)
        for action in actions:
            coarse_u = changed_future(coarse_h, mechanism, action, 1 / 8)
            fine_u = changed_future(fine_h, mechanism, action, DT)
            for lhs, rhs in ((coarse_f, fine_f), (coarse_u, fine_u)):
                relative = ((lhs - rhs).square().mean((1, 2, 3)).sqrt() / rhs.square().mean((1, 2, 3)).sqrt()).max()
                worst_field = max(worst_field, float(relative))
            coarse_r, fine_r = coarse_u - coarse_f, fine_u - fine_f
            relative = ((coarse_r - fine_r).square().mean((1, 2, 3)).sqrt() /
                        fine_r.square().mean((1, 2, 3)).sqrt()).max()
            worst_response = max(worst_response, float(relative))
            for lhs, rhs in ((torch.cat((coarse_h[-1:], coarse_f)), torch.cat((fine_h[-1:], fine_f))),
                             (torch.cat(((coarse_h[-1:] + action.delta), coarse_u)),
                              torch.cat(((fine_h[-1:] + action.delta), fine_u)))):
                epsilon = torch.maximum(epsilon, (lhs - rhs).abs().amax((0, 2, 3)))
    if not smoke and (worst_field > 1e-4 or worst_response > 1e-3):
        raise RuntimeError("the registered 1/16 numerical qualification failed")
    return epsilon


def pack(history, factual, changed, action, mechanism, initial, pair, stats, epsilon):
    actions = action_bank()
    labels, known = [], []
    for h, f, u, a, p in zip(history, factual, changed, action, pair):
        delta = torch.zeros_like(h[-1]) if p == -1 and torch.equal(u, f) else actions[int(a)].delta
        e, k = event_labels(h, f, u, delta, stats["event_threshold"], epsilon)
        labels.append(e)
        known.append(k)
    return {"history": torch.stack(history), "factual": torch.stack(factual),
            "changed": torch.stack(changed), "action": torch.tensor(action, dtype=torch.long),
            "mechanism": torch.tensor(mechanism, dtype=torch.long), "initial": torch.tensor(initial, dtype=torch.long),
            "pair": torch.tensor(pair, dtype=torch.long), "events": torch.stack(labels), "known": torch.stack(known)}


def prepare(output: Path, seed: int, smoke: bool):
    mechanisms, initials = (8, 2) if smoke else (32, 16)
    roots_h, roots_f, root_m, root_i = [], [], [], []
    for mechanism in range(mechanisms):
        for initial in range(initials):
            h, f = generate_root("train", mechanism, initial)
            roots_h.append(h); roots_f.append(f); root_m.append(mechanism); root_i.append(initial)
    root_history, root_factual = torch.stack(roots_h), torch.stack(roots_f)
    stats = fit_statistics(root_history, root_factual)
    epsilon = numerical_epsilon(smoke)
    actions = action_bank()
    if smoke:
        table = [(0., 2 * m, 2 * m + 1, m) for m in range(8)]
        selected = [(m, 3) for m in range(8)]
    else:
        standardized = ((root_factual - stats["mean"].view(1, 1, 2, 1, 1)) /
                        stats["std"].view(1, 1, 2, 1, 1)).numpy()
        table, _ = frozen_pair_table(standardized, list(range(512)), root_m)
        selected = select_uniform_pairs(table, seed)
    pair_h, pair_f, pair_u, pair_a, pair_m, pair_i, pair_id = [], [], [], [], [], [], []
    for pair, action_id in selected:
        _, first, second, mechanism = table[pair]
        for root_id in (first, second):
            pair_h.append(roots_h[root_id]); pair_f.append(roots_f[root_id])
            pair_u.append(changed_future(roots_h[root_id], mechanism_parameters("train", mechanism), actions[action_id]))
            pair_a.append(action_id); pair_m.append(mechanism); pair_i.append(root_i[root_id]); pair_id.append(pair)
    root_records = pack(roots_h, roots_f, roots_f, [0] * len(roots_h), root_m, root_i,
                        [-1] * len(roots_h), stats, epsilon)
    pair_records = pack(pair_h, pair_f, pair_u, pair_a, pair_m, pair_i, pair_id, stats, epsilon)
    def evaluation_bank(split):
        bank_h, bank_f, bank_u, bank_a, bank_m, bank_i, bank_pair = [], [], [], [], [], [], []
        eval_mechs, eval_initials, eval_actions = (4, 2, (1, 3)) if smoke else (4, 16, range(16))
        for mechanism in range(eval_mechs):
            rule = mechanism_parameters(split, mechanism)
            for initial in range(eval_initials):
                h, f = generate_root(split, mechanism, initial)
                for action_id in eval_actions:
                    bank_h.append(h); bank_f.append(f)
                    bank_u.append(changed_future(h, rule, actions[action_id]))
                    bank_a.append(action_id); bank_m.append(mechanism); bank_i.append(initial); bank_pair.append(-1)
        return pack(bank_h, bank_f, bank_u, bank_a, bank_m, bank_i, bank_pair, stats, epsilon)

    selection = evaluation_bank("validation-selection")
    calibration = evaluation_bank("validation-calibration")
    data = {"schema": "paired-transport-exchange-v1", "profile": PROFILE, "dt": DT,
            "history": 6, "horizon": 6, "seed": seed, "smoke": smoke,
            "statistics": stats, "epsilon_num": epsilon, "roots": root_records,
            "pairs": pair_records, "selection": selection, "calibration": calibration}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(json.dumps({"output": str(output), "smoke": smoke, "roots": len(roots_h),
                      "pair_actions": len(selected), "changed_labels": len(pair_h),
                      "selection_rows": len(selection["history"]),
                      "calibration_rows": len(calibration["history"]), "epsilon_num": epsilon.tolist()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(args.output, args.seed, args.smoke)
