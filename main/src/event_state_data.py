"""Train-calibrated monthly event bounds."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .event_contract import validate_event_strata_contract


FEATURE_NAMES = (
    "tc_monthly_load", "sst_warm_area_fraction", "cuti_active_band_fraction",
    "carbon_transition_fraction", "carbon_sink_area_fraction",
    "nino34_trailing_warm", "nino34_trailing_cold",
)



EVENT_NAMES = ("tc_sid_day_lower_daily_mean", "tc_sid_day_upper_daily_mean", "cuti_active_band_count_daily_mean")



CARBON_NAMES = ("carbon_transition_fraction_lower", "carbon_transition_fraction_upper", "carbon_known_support_fraction")



CARBON_SOURCE_NAMES = (
    "carbon_transition_fraction_lower", "carbon_transition_fraction_upper", "carbon_sink_to_source_fraction_lower",
    "carbon_sink_to_source_fraction_upper", "carbon_source_to_sink_fraction_lower", "carbon_source_to_sink_fraction_upper",
    "carbon_known_support_fraction",
)



EVENT_CLOCK = "monthly summaries of co-occurrence on the same UTC calendar day; not exact-instant simultaneity"



CARBON_SUPPORT = "joint flux/state validity >=95% in this fold training months; outside support excluded"



def _attr_json(value, name):
    if isinstance(value, bytes):
        value = value.decode()
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name} metadata") from error



def _false_attr(attrs, name):
    value = attrs.get(name)
    if isinstance(value, bytes):
        value = value.decode()
    if value is not False and str(value).lower() != "false":
        raise ValueError(f"source does not declare {name}=false")




def _cell_overlap(latitude, longitude, south, north, west, east):
    """Fraction of each latitude/longitude cell in a rectangular lat/lon box."""
    def edges(centres):
        delta = np.diff(centres)
        if len(delta) == 0 or not np.allclose(delta, delta[0]):
            raise ValueError("latitude/longitude centres must be regularly spaced")
        return centres - delta[0] / 2, centres + delta[0] / 2

    lat_lo, lat_hi = edges(latitude)
    lon_lo, lon_hi = edges(longitude)
    lat_a, lat_b = np.maximum(lat_lo, south), np.minimum(lat_hi, north)
    lat_fraction = np.where(lat_b > lat_a, (np.sin(np.deg2rad(lat_b)) - np.sin(np.deg2rad(lat_a))) /
                            (np.sin(np.deg2rad(lat_hi)) - np.sin(np.deg2rad(lat_lo))), 0.0)
    lon_fraction = np.clip(np.minimum(lon_hi, east) - np.maximum(lon_lo, west), 0, None) / (lon_hi - lon_lo)
    return lat_fraction[:, None] * lon_fraction[None]



def _fit_calibration(state, state_valid, area, latitude, longitude, train_stop):
    train_state, train_valid = state[:train_stop], state_valid[:train_stop]
    month = np.arange(train_stop) % 12
    support = train_valid.mean(0) >= .95
    thresholds = np.zeros((12,) + state.shape[1:], dtype="float32")
    climatology = np.zeros_like(thresholds)
    for calendar_month in range(12):
        values = train_state[month == calendar_month][:, support].astype("float64")
        known = train_valid[month == calendar_month][:, support]
        if not known.any(0).all():
            raise ValueError("admitted SST support has no observation in a training calendar month")
        values[~known] = np.nan
        thresholds[calendar_month, support] = np.nanquantile(values, .9, axis=0).astype("float32")
        climatology[calendar_month, support] = np.nanmean(values, axis=0).astype("float32")
    nino_fraction = _cell_overlap(latitude, longitude, -5, 5, -170, -120)
    nino_weights = area * nino_fraction
    nino_support = support & (nino_weights > 0)
    if not support.any() or nino_weights[nino_support].sum() <= 0:
        raise ValueError("empty admitted SST support")
    return {
        "sst_thresholds": torch.as_tensor(thresholds),
        "sst_support": torch.as_tensor(support),
        "nino_climatology": torch.as_tensor(climatology),
        "nino_weights": torch.as_tensor(nino_weights.astype("float32")),
        "nino_support": torch.as_tensor(nino_support),
        "train_range": [0, int(train_stop)],
    }



def _array(calibration, name, dtype=None):
    if name not in calibration:
        raise ValueError(f"calibration is missing {name}")
    value = calibration[name]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)



def load_event_state_table(dataset, max_horizon=3, calibration=None):
    """Read monthly event bounds with thresholds fitted on training months."""
    root, fold, split = Path(dataset.root), dataset.fold, dataset.split
    manifest = dataset.manifest
    value_stop = int(manifest["splits"][split][1])
    train_stop = int(manifest["splits"]["train"][1])
    event_path = root / "events" / fold / "monthly_events.h5"
    carbon_path = root / "folds" / fold / "carbon_monthly.h5"
    event_contract = validate_event_strata_contract(root, fold, event_path, EVENT_NAMES, state_manifest=manifest, value_stop=value_stop)
    with h5py.File(root / "global_state.h5") as source, h5py.File(event_path) as events, h5py.File(carbon_path) as carbon:
        time = source["time"][:]  # Clock metadata is deliberately unrestricted.
        if not np.array_equal(events["time"][:], time) or not np.array_equal(carbon["time"][:], time):
            raise ValueError("state, event, and carbon monthly clocks must exactly match")
        months = (np.datetime64("1970-01-01") + time.astype("timedelta64[D]")).astype("datetime64[M]")
        if months[0].astype("int64") % 12 or not np.all(np.diff(months.astype("int64")) == 1):
            raise ValueError("source clock is not the canonical contiguous January-start monthly clock")
        if value_stop > len(time) or source["state"].shape[0] != len(time):
            raise ValueError("split exceeds the global state clock")
        event_names = _attr_json(events.attrs.get("target_names"), "event target_names")
        roles = _attr_json(events.attrs.get("target_roles"), "event target_roles")
        if events.attrs.get("clock") != EVENT_CLOCK:
            raise ValueError("unexpected event monthly clock")
        _false_attr(events.attrs, "encoder_visible")
        _false_attr(carbon.attrs, "encoder_visible")
        retrospective_attr = events.attrs.get("measurement_validity_is_not_online_authorization")
        if retrospective_attr is not True and str(retrospective_attr).lower() != "true":
            raise ValueError("event source does not declare retrospective-only measurement validity")
        primary = set(roles.get("primary_per_source_summary", []))
        if not set(EVENT_NAMES).issubset(primary) or not set(EVENT_NAMES).issubset(event_names):
            raise ValueError("TC/CUTI summaries are not declared primary event roles")
        retrospective = set(roles.get("retrospective_context_not_causal_input", []))
        if not {"enso_positive_context", "enso_negative_context", "enso_any_context"}.issubset(retrospective):
            raise ValueError("ENSO retrospective role is missing")
        carbon_names = _attr_json(carbon.attrs.get("target_names"), "carbon target_names")
        if (carbon_names != list(CARBON_SOURCE_NAMES) or carbon.attrs.get("time_support") != "current and previous month only"
                or carbon.attrs.get("support") != CARBON_SUPPORT):
            raise ValueError("unexpected carbon target contract")
        state = source["state"][:value_stop, 1].astype("float32")
        state_valid = source["state_valid"][:value_stop].astype(bool)
        fgco2 = source["fgco2"][:value_stop].astype("float32")
        fgco2_valid = source["fgco2_valid"][:value_stop].astype(bool)
        area = source["cell_area_km2"][:].astype("float64")
        latitude, longitude = source["latitude"][:], source["longitude"][:]
        event_columns = np.asarray(sorted(event_contract["columns"]), dtype=int)
        event_values, event_valid = events["values"][:value_stop, event_columns], events["valid"][:value_stop, event_columns].astype(bool)
        carbon_values, carbon_valid = carbon["values"][:value_stop], carbon["valid"][:value_stop].astype(bool)
        carbon_support, carbon_weights = carbon["support"][:].astype(bool), carbon["area_weights_km2"][:].astype("float64")

    if state.shape[1:] != area.shape or carbon_support.shape != area.shape or carbon_weights.shape != area.shape:
        raise ValueError("global spatial fields do not share a grid")
    if event_values.shape != event_valid.shape or carbon_values.shape != carbon_valid.shape or event_values.shape[0] != value_stop or carbon_values.shape[0] != value_stop:
        raise ValueError("event/carbon values and validity arrays must align")
    selected_event = [event_columns.tolist().index(event_names.index(name)) for name in EVENT_NAMES]
    selected_carbon = [carbon_names.index(name) for name in CARBON_NAMES]
    if not np.isfinite(event_values[:, selected_event][event_valid[:, selected_event]]).all() or not np.isfinite(carbon_values[:, selected_carbon][carbon_valid[:, selected_carbon]]).all():
        raise ValueError("valid native event values must be finite")
    if not np.isfinite(state[state_valid]).all() or not np.isfinite(fgco2[fgco2_valid]).all():
        raise ValueError("valid state/flux values must be finite")
    if calibration is None:
        calibration = _fit_calibration(state, state_valid, area, latitude, longitude, train_stop)
    else:
        calibration = dict(calibration)
        if list(calibration.get("train_range", [])) != [0, train_stop]:
            raise ValueError("calibration train range does not match this fold")
        for name in ("sst_thresholds", "sst_support", "nino_climatology", "nino_weights", "nino_support"):
            if name in calibration:
                calibration[name] = torch.as_tensor(calibration[name]).cpu()

    sst_thresholds = _array(calibration, "sst_thresholds", "float32")
    sst_support = _array(calibration, "sst_support", bool)
    climatology = _array(calibration, "nino_climatology", "float32")
    nino_weights = _array(calibration, "nino_weights", "float64")
    nino_support = _array(calibration, "nino_support", bool)
    if sst_thresholds.shape != (12,) + area.shape or climatology.shape != (12,) + area.shape or sst_support.shape != area.shape:
        raise ValueError("calibration SST grid does not match source")
    if nino_weights.shape != area.shape or nino_support.shape != area.shape:
        raise ValueError("calibration Nino grid does not match source")

    count = value_stop
    lower, upper = np.zeros((count, 7), dtype="float32"), np.ones((count, 7), dtype="float32")
    valid = np.zeros((count, 7), dtype=bool)
    columns = dict(zip(EVENT_NAMES, selected_event))
    tc_known = event_valid[:, columns[EVENT_NAMES[0]]] & event_valid[:, columns[EVENT_NAMES[1]]]
    tc_lower, tc_upper = event_values[:, columns[EVENT_NAMES[0]]], event_values[:, columns[EVENT_NAMES[1]]]
    if ((tc_lower[tc_known] < 0).any() or (tc_lower[tc_known] > tc_upper[tc_known]).any()):
        raise ValueError("TC lower/upper counts are invalid")
    lower[tc_known, 0] = tc_lower[tc_known] / (1 + tc_lower[tc_known])
    upper[tc_known, 0] = tc_upper[tc_known] / (1 + tc_upper[tc_known])
    valid[:, 0] = tc_known
    cuti_known = event_valid[:, columns[EVENT_NAMES[2]]]
    cuti = event_values[:, columns[EVENT_NAMES[2]]]
    if ((cuti[cuti_known] < 0).any() or (cuti[cuti_known] > 17).any()):
        raise ValueError("CUTI active bands must be in [0,17]")
    lower[cuti_known, 2] = upper[cuti_known, 2] = cuti[cuti_known] / 17
    valid[:, 2] = cuti_known
    c0, c1, c_known = (carbon_names.index(name) for name in CARBON_NAMES)
    carbon_known = carbon_valid[:, c0] & carbon_valid[:, c1] & carbon_valid[:, c_known]
    if ((carbon_values[carbon_known, c0] < 0).any() or (carbon_values[carbon_known, c1] > 1).any()
            or (carbon_values[carbon_known, c0] > carbon_values[carbon_known, c1]).any()
            or (carbon_values[carbon_known, c_known] < 0).any() or (carbon_values[carbon_known, c_known] > 1).any()):
        raise ValueError("carbon transition bounds must be ordered fractions")
    carbon_known &= (carbon_values[:, c_known] > 0) & ((carbon_values[:, c1] - carbon_values[:, c0]) < 1)
    lower[carbon_known, 3], upper[carbon_known, 3] = carbon_values[carbon_known, c0], carbon_values[carbon_known, c1]
    valid[:, 3] = carbon_known

    support_weight = area * sst_support
    support_denominator = support_weight.sum()
    if support_denominator <= 0:
        raise ValueError("empty SST support")
    known = state_valid & sst_support
    warm = known & (state > sst_thresholds[np.arange(count) % 12])
    lower[:, 1] = (warm * support_weight).sum((1, 2)) / support_denominator
    upper[:, 1] = ((warm | (sst_support & ~known)) * support_weight).sum((1, 2)) / support_denominator
    valid[:, 1] = known[:, sst_support].any(1)

    carbon_denominator = carbon_weights[carbon_support].sum()
    if carbon_denominator <= 0:
        raise ValueError("empty frozen carbon support")
    known = state_valid & fgco2_valid & carbon_support
    sink = known & (fgco2 > 0)
    lower[:, 4] = (sink * carbon_weights).sum((1, 2)) / carbon_denominator
    upper[:, 4] = ((sink | (carbon_support & ~known)) * carbon_weights).sum((1, 2)) / carbon_denominator
    valid[:, 4] = known[:, carbon_support].any(1)

    nino_denominator = nino_weights[nino_support].sum()
    if nino_denominator <= 0:
        raise ValueError("empty calibrated Nino support")
    nino_climatology = climatology[np.arange(count) % 12]
    nino_known = state_valid & nino_support & np.isfinite(nino_climatology)
    nino_monthly_valid = nino_known[:, nino_support].all(1)
    anomaly = np.where(np.isfinite(state) & np.isfinite(nino_climatology), state - nino_climatology, 0)
    nino_monthly = (anomaly[:, nino_support] * nino_weights[nino_support]).sum(1) / nino_denominator
    nino_sum = np.cumsum(np.r_[0., nino_monthly])
    trailing = (nino_sum[3:] - nino_sum[:-3]) / 3
    trailing_valid = np.zeros(count, bool); trailing_valid[2:] = nino_monthly_valid[:-2] & nino_monthly_valid[1:-1] & nino_monthly_valid[2:]
    lower[trailing_valid, 5] = upper[trailing_valid, 5] = trailing[trailing_valid[2:]] > .5
    lower[trailing_valid, 6] = upper[trailing_valid, 6] = trailing[trailing_valid[2:]] < -.5
    valid[:, 5] = valid[:, 6] = trailing_valid
    if (lower > upper).any() or (lower < 0).any() or (upper > 1).any():
        raise ValueError("event identification bounds must be ordered unit fractions")
    return {"lower": torch.as_tensor(lower), "upper": torch.as_tensor(upper), "valid": torch.as_tensor(valid),
            "calibration": calibration,
            "metadata": {"feature_names": list(FEATURE_NAMES), "train_range": list(calibration["train_range"]), "value_stop": value_stop}}
