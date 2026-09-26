"""Area-weighted physical validation metrics for numerical field forecasts."""

import math

import torch


def empty_metric_totals(leads, channels):
    return {"shape": (leads, channels), "rows": [], "next_unit_id": 0, "aggregation": None}


def _weight_fields(valid, area, spatial_mask, sample_mask):
    declared = torch.ones(valid.shape[:2], dtype=torch.bool, device=valid.device)
    if sample_mask is not None:
        declared &= sample_mask.bool()
    declared_weight = area[:, None].to(dtype=torch.float64)
    if spatial_mask is not None:
        declared_weight = declared_weight * spatial_mask[None, None, None].to(declared_weight)
    declared_weight = declared_weight.expand_as(valid)
    truth_weight = declared_weight * valid.to(declared_weight)
    truth_weight = truth_weight * declared[:, :, None, None, None]
    return declared, declared_weight, truth_weight


def _row_statistics(prediction, target, valid, area, std, spatial_mask, sample_mask):
    declared, declared_weight, truth_weight = _weight_fields(valid, area, spatial_mask, sample_mask)
    error = (prediction.double() - target.double()) * std.double().reshape(1, 1, -1, 1, 1)
    finite = torch.isfinite(prediction) & torch.isfinite(error)
    bad = (truth_weight > 0) & ~finite
    safe_error = torch.where(finite, error, torch.zeros_like(error))
    statistics = torch.stack((
        truth_weight.sum((3, 4)),
        declared_weight.sum((3, 4)),
        (truth_weight * finite).sum((3, 4)),
        (safe_error.square() * truth_weight).sum((3, 4)),
        (safe_error.abs() * truth_weight).sum((3, 4)),
        (safe_error * truth_weight).sum((3, 4)),
        bad.flatten(3).any(3).to(truth_weight),
        declared[:, :, None].expand_as(truth_weight.sum((3, 4))).to(truth_weight),
    )).cpu()
    return statistics


def add_field_metrics(totals, prediction, target, valid, area, std, spatial_mask=None, sample_mask=None, unit_ids=None):
    """Accumulate fixed-truth-support metrics, with equal-origin aggregation at finalize."""
    statistics = _row_statistics(prediction, target, valid, area, std, spatial_mask, sample_mask)
    if unit_ids is None:
        unit_ids = list(range(totals["next_unit_id"], totals["next_unit_id"] + prediction.shape[0]))
        totals["next_unit_id"] += prediction.shape[0]
        aggregation = "equal_origin"
    else:
        unit_ids = [int(unit) for unit in unit_ids.detach().cpu().tolist()]
        aggregation = "equal_origin"
    if totals["aggregation"] is None:
        totals["aggregation"] = aggregation
    for batch_index, unit_id in enumerate(unit_ids):
        for lead in range(prediction.shape[1]):
            for channel in range(prediction.shape[2]):
                support, declared_area, finite_weight, squared_error, absolute_error, signed_error, failed, declared = (float(statistics[index, batch_index, lead, channel]) for index in range(8))
                if not declared:
                    continue
                row = {"unit_id": unit_id, "lead": lead, "channel": channel, "support": support, "declared_area": declared_area, "finite_weight": finite_weight, "failed": bool(failed)}
                if support > 0 and not failed:
                    row.update(mse=squared_error / support, mae=absolute_error / support, bias=signed_error / support)
                totals["rows"].append(row)


def _mean(values):
    return sum(values) / len(values) if values else None


def _metric_summary(totals):
    leads, channels = totals["shape"]
    grouped = {}
    for row in totals["rows"]:
        grouped.setdefault((row["lead"], row["channel"], row["unit_id"]), []).append(row)
    fields = {name: [[None for _ in range(channels)] for _ in range(leads)] for name in ("mse", "rmse", "mae", "bias", "area_weight", "declared_area", "status", "declared_count", "eligible_count", "zero_support_count", "failed_count", "branch_count", "eligible_branch_count", "zero_support_branch_count", "failed_branch_count", "finite_coverage", "support_fraction")}
    for lead in range(leads):
        for channel in range(channels):
            units = [rows for (row_lead, row_channel, _), rows in grouped.items() if row_lead == lead and row_channel == channel]
            eligible = [rows for rows in units if any(row["support"] > 0 for row in rows)]
            failed = [rows for rows in eligible if any(row["support"] > 0 and row["failed"] for row in rows)]
            status = "N/A" if not eligible else "FAIL" if failed else "OK"
            fields["status"][lead][channel] = status
            fields["declared_count"][lead][channel] = len(units)
            fields["eligible_count"][lead][channel] = len(eligible)
            fields["zero_support_count"][lead][channel] = len(units) - len(eligible)
            fields["failed_count"][lead][channel] = len(failed)
            fields["branch_count"][lead][channel] = sum(len(rows) for rows in units)
            fields["eligible_branch_count"][lead][channel] = sum(row["support"] > 0 for rows in units for row in rows)
            fields["zero_support_branch_count"][lead][channel] = sum(row["support"] == 0 for rows in units for row in rows)
            fields["failed_branch_count"][lead][channel] = sum(row["support"] > 0 and row["failed"] for rows in units for row in rows)
            fields["area_weight"][lead][channel] = sum(row["support"] for rows in units for row in rows) if units else None
            fields["declared_area"][lead][channel] = sum(row["declared_area"] for rows in units for row in rows) if units else None
            unit_support = [_mean([row["support"] / row["declared_area"] for row in rows if row["declared_area"] > 0]) for rows in units]
            fields["support_fraction"][lead][channel] = _mean([value for value in unit_support if value is not None])
            fields["finite_coverage"][lead][channel] = _mean([_mean([row["finite_weight"] / row["support"] for row in rows if row["support"] > 0]) for rows in eligible]) if eligible else None
            if status == "OK":
                unit_metrics = [{name: _mean([row[name] for row in rows if row["support"] > 0]) for name in ("mse", "mae", "bias")} for rows in eligible]
                fields["mse"][lead][channel] = _mean([row["mse"] for row in unit_metrics])
                fields["rmse"][lead][channel] = math.sqrt(fields["mse"][lead][channel])
                fields["mae"][lead][channel] = _mean([row["mae"] for row in unit_metrics])
                fields["bias"][lead][channel] = _mean([row["bias"] for row in unit_metrics])
    return fields


def finalize_field_metrics(totals, baseline=None):
    summary = _metric_summary(totals)
    out = {"rmse": summary["rmse"], "mse": summary["mse"], "mae": summary["mae"], "bias": summary["bias"], "area_weight": summary["area_weight"], "metric_scheme": "field_metrics_v2", "aggregation": totals["aggregation"] or "equal_origin", "status": summary["status"], "declared_count": summary["declared_count"], "eligible_count": summary["eligible_count"], "zero_support_count": summary["zero_support_count"], "failed_count": summary["failed_count"], "branch_count": summary["branch_count"], "eligible_branch_count": summary["eligible_branch_count"], "zero_support_branch_count": summary["zero_support_branch_count"], "failed_branch_count": summary["failed_branch_count"], "finite_coverage": summary["finite_coverage"], "support_fraction": summary["support_fraction"]}
    if baseline is not None:
        baseline_summary = _metric_summary(baseline)
        persistence_rmse = [[None for _ in row] for row in summary["rmse"]]
        persistence_skill = [[None for _ in row] for row in summary["rmse"]]
        for lead in range(len(persistence_rmse)):
            for channel in range(len(persistence_rmse[lead])):
                baseline_mse = baseline_summary["mse"][lead][channel]
                if baseline_summary["status"][lead][channel] == "OK":
                    persistence_rmse[lead][channel] = math.sqrt(baseline_mse)
                if summary["status"][lead][channel] == "OK" and baseline_summary["status"][lead][channel] == "OK" and baseline_mse != 0:
                    persistence_skill[lead][channel] = 1 - summary["mse"][lead][channel] / baseline_mse
        out["persistence_rmse"] = persistence_rmse
        out["persistence_skill"] = persistence_skill
    return out


def field_metric_rows(origin_index, prediction, target, valid, area, std, prefix, spatial_mask=None, sample_mask=None):
    """Return JSON-ready origin/lead/channel metrics under the fixed truth support."""
    statistics = _row_statistics(prediction, target, valid, area, std, spatial_mask, sample_mask)
    rows = []
    for batch_index, origin in enumerate(origin_index.detach().cpu().tolist()):
        for lead in range(prediction.shape[1]):
            for channel in range(prediction.shape[2]):
                support, declared_area, finite_weight, squared_error, absolute_error, signed_error, failed, declared = (float(statistics[index, batch_index, lead, channel]) for index in range(8))
                if not declared:
                    continue
                status = "N/A" if support == 0 else "FAIL" if failed else "OK"
                row = {"origin_index": int(origin), "lead": lead + 1, "channel": channel, "status": status, "declared_count": 1, "eligible_count": int(support > 0), "zero_support_count": int(support == 0), "failed_count": int(status == "FAIL"), "branch_count": 1, "eligible_branch_count": int(support > 0), "zero_support_branch_count": int(support == 0), "failed_branch_count": int(status == "FAIL"), "area_weight": support, "declared_area": declared_area, "finite_coverage": finite_weight / support if support > 0 else None, "support_fraction": support / declared_area if declared_area > 0 else None}
                if status == "OK":
                    mse = squared_error / support
                    row.update({f"{prefix}_mse": mse, f"{prefix}_rmse": math.sqrt(mse), f"{prefix}_mae": absolute_error / support, f"{prefix}_bias": signed_error / support})
                else:
                    row.update({f"{prefix}_mse": None, f"{prefix}_rmse": None, f"{prefix}_mae": None, f"{prefix}_bias": None})
                rows.append(row)
    return rows
