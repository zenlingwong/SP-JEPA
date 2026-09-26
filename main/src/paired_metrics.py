"""Paired-environment natural/changed field and response metrics."""
from __future__ import annotations

import numpy as np
import torch

from .paired_data import model_batch


FIELD_RESPONSE_ENERGY_FLOOR = 100 * 9.787682590515326e-13


def encoded_future(model, batch: dict, changed: bool):
    history = batch["x"].clone()
    branch = batch["changed_target"] if changed else batch["target"]
    if changed:
        history[:, -1] += (batch["action"] @ model.A.T).reshape_as(history[:, -1])
    values = torch.cat((history, branch), 1)
    calendar = torch.cat((batch["calendar"], batch["future_calendar"]), 1)
    a, h = [], []
    for lead in range(6):
        state = model.encode(values[:, lead + 1:lead + 7],
                             torch.ones_like(values[:, lead + 1:lead + 7], dtype=torch.bool),
                             batch["geometry"], calendar[:, lead + 1:lead + 7])
        a.append(state["a"][:, -1]); h.append(state["h"][:, -1])
    return model.read_fields({"a": torch.stack(a, 1), "h": torch.stack(h, 1)}, batch["geometry"])


@torch.no_grad()
def evaluate(model, data: dict, *, device="cpu", batch_size=16, endpoints=False):
    model.eval()
    bank = data["selection"]
    n = len(bank["history"])
    predictions = {key: [] for key in ("factual", "changed", "encoded_factual", "encoded_changed")}
    for start in range(0, n, batch_size):
        batch = model_batch({**data, "pairs": bank}, range(start, min(start + batch_size, n)),
                            paired=True, sig_rows=[0], device=device)
        state = model.encode(batch["x"], batch["valid"], batch["geometry"], batch["calendar"])
        factual = model.rollout(state, batch["calendar"], batch["future_calendar"])
        changed = model.rollout(model.edit(state, batch["action"]), batch["calendar"], batch["future_calendar"])
        predictions["factual"].append(model.read_fields(factual, batch["geometry"]).cpu())
        predictions["changed"].append(model.read_fields(changed, batch["geometry"]).cpu())
        if endpoints:
            predictions["encoded_factual"].append(encoded_future(model, batch, False).cpu())
            predictions["encoded_changed"].append(encoded_future(model, batch, True).cpu())
    preds = {key: torch.cat(values).double() for key, values in predictions.items() if values}
    mean = data["statistics"]["mean"].view(1, 1, 2, 1, 1)
    std = data["statistics"]["std"].view(1, 1, 2, 1, 1)
    truth_f = ((bank["factual"] - mean) / std).float().double()
    truth_u = ((bank["changed"] - mean) / std).float().double()
    r_true = truth_u - truth_f
    r_pred = preds["changed"] - preds["factual"]
    mechanism = bank["mechanism"].numpy()

    def by_mechanism(values):
        row = values.reshape(n, -1).mean(1).numpy()
        return np.asarray([row[mechanism == m].mean() for m in np.unique(mechanism)])

    def equal_mechanism_mean(values):
        return float(by_mechanism(values).mean())

    factual_mse = equal_mechanism_mean((preds["factual"] - truth_f).square())
    changed_mse = equal_mechanism_mean((preds["changed"] - truth_u).square())
    v = equal_mechanism_mean(r_true.square())
    response_error = equal_mechanism_mean((r_pred - r_true).square())
    report = {"rows": n, "mechanisms": int(len(np.unique(mechanism))),
              "selection_score": .5 * (factual_mse + changed_mse),
              "factual_mse": factual_mse, "changed_mse": changed_mse,
              "response_energy": v, "response_error": response_error, "RS": 1 - response_error / v,
              "resolution_floor": FIELD_RESPONSE_ENERGY_FLOOR,
              "response_resolved": bool(v > FIELD_RESPONSE_ENERGY_FLOOR)}
    if endpoints:
        r_encoded = preds["encoded_changed"] - preds["encoded_factual"]
        capture = equal_mechanism_mean((r_encoded - r_true).square()) / v
        transport = equal_mechanism_mean((r_pred - r_encoded).square()) / v
        report.update({"C": capture, "P": transport, "sqrt_C": capture ** .5,
                       "sqrt_P": transport ** .5, "LCS": 1 - (capture ** .5 + transport ** .5) ** 2})
        numerators = {"v": by_mechanism(r_true.square()),
                      "m": by_mechanism((r_pred - r_true).square()),
                      "c": by_mechanism((r_encoded - r_true).square()),
                      "p": by_mechanism((r_pred - r_encoded).square())}
        generator = np.random.default_rng(17)
        draws = generator.integers(0, len(numerators["v"]), size=(2000, len(numerators["v"])))
        sampled = {key: values[draws].mean(1) for key, values in numerators.items()}
        c = sampled["c"] / sampled["v"]
        p = sampled["p"] / sampled["v"]
        distributions = {"C": c, "P": p, "RS": 1 - sampled["m"] / sampled["v"],
                         "LCS": 1 - (np.sqrt(c) + np.sqrt(p)) ** 2}
        report["intervals_95"] = {key: {"lower": float(np.percentile(values, 2.5)),
                                        "upper": float(np.percentile(values, 97.5))}
                                  for key, values in distributions.items()}
        report["bootstrap"] = {"unit": "mechanism", "resamples": 2000, "seed": 17,
                               "synchronized_C_P_V": True}
    model.train()
    return report


@torch.no_grad()
def field_calibration(model, data: dict, *, device="cpu", batch_size=16):
    """Independent field-only capture and endpoint gates on calibration roots."""
    bank = data["calibration"]
    n = len(bank["history"])
    model.eval()
    timeline = torch.cat((data["roots"]["history"], data["roots"]["factual"]), 1)
    mean = data["statistics"]["mean"].view(1, 1, 2, 1, 1)
    std = data["statistics"]["std"].view(1, 1, 2, 1, 1)
    normalized = ((timeline - mean) / std).double()
    train_mechanism = data["roots"]["mechanism"]
    baseline = torch.stack([normalized[train_mechanism == m].mean((0, 1))
                            for m in train_mechanism.unique(sorted=True)]).mean(0)
    arrays = {key: [] for key in ("capture_baseline", "capture_current", "endpoint_baseline", "endpoint_current")}
    for start in range(0, n, batch_size):
        batch = model_batch({**data, "pairs": bank}, range(start, min(start + batch_size, n)),
                            paired=True, sig_rows=[0], device=device)
        state = model.encode(batch["x"], batch["valid"], batch["geometry"], batch["calendar"])
        now = model.read_fields({"a": state["a"][:, -1], "h": state["h"][:, -1]}, batch["geometry"])
        true_now = batch["x"][:, -1]
        encoded_f = encoded_future(model, batch, False)
        encoded_u = encoded_future(model, batch, True)
        response = batch["changed_target"] - batch["target"]
        arrays["capture_baseline"].append((baseline.to(device) - true_now).square().mean((-2, -1)).cpu())
        arrays["capture_current"].append((now - true_now).square().mean((-2, -1)).cpu())
        arrays["endpoint_baseline"].append(response.square().mean((1, 3, 4)).cpu())
        arrays["endpoint_current"].append((encoded_u - encoded_f - response).square().mean((1, 3, 4)).cpu())
    arrays = {key: torch.cat(values).double().numpy() for key, values in arrays.items()}
    mechanism = bank["mechanism"].numpy()
    initial = bank["initial"].numpy()
    known = bank["known"]
    eligible = (known[:, :, 1:, :].reshape(n, -1).all(1) & known[:, 0, 0, :].all(1)).numpy()
    result = {"split": "validation-calibration", "rows": n,
              "roots": int(len(set(zip(bank["mechanism"].tolist(), bank["initial"].tolist())))),
              "mechanisms": int(len(np.unique(mechanism))), "eligible_rows": int(eligible.sum()),
              "eligible_fraction": float(eligible.mean()), "fields": {}}
    generator = np.random.default_rng(17)
    groups = np.unique(mechanism)
    draws = generator.integers(0, len(groups), size=(2000, len(groups)))

    def by_mechanism_and_root(values):
        result = []
        for m in groups:
            roots = []
            for root in np.unique(initial[mechanism == m]):
                rows = (mechanism == m) & (initial == root) & eligible
                if rows.any():
                    roots.append(values[rows].mean())
            result.append(float(np.mean(roots)) if roots else float("nan"))
        return np.asarray(result)

    for channel in range(2):
        field = {}
        for gate in ("capture", "endpoint"):
            base = arrays[f"{gate}_baseline"][:, channel]
            current = arrays[f"{gate}_current"][:, channel]
            base_m = by_mechanism_and_root(base)
            current_m = by_mechanism_and_root(current)
            difference = base_m - current_m
            samples = difference[draws].mean(1)
            lower, upper = np.percentile(samples, [2.5, 97.5])
            field[gate] = {"baseline_error": float(base_m.mean()),
                           "current_error": float(current_m.mean()),
                           "improvement": float(difference.mean()),
                           "improvement_interval_95": {"lower": float(lower), "upper": float(upper)},
                           "passed": bool(np.isfinite(lower) and lower > 0)}
        result["fields"][str(channel)] = field
    result["field_qualified"] = all(gate["passed"] for field in result["fields"].values()
                                    for gate in field.values())
    result["bootstrap"] = {"unit": "mechanism", "resamples": 2000, "seed": 17}
    result["scope"] = "field-only calibration; event and joint gates are not assessed"
    model.train()
    return result
