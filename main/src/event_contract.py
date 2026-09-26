"""Boundary checks for retrospective event strata used in validation."""

import json
from pathlib import Path

import h5py
import numpy as np


def _json_attr(value, name):
    if isinstance(value, bytes):
        value = value.decode()
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"event source has invalid {name} metadata") from error


def validate_event_strata_contract(release_root, fold, event_load_path, selected_names, state_manifest=None, value_stop=None):
    """Validate event/state clocks and producer declarations before joining labels.

    HDF5 attributes describe the produced columns, but package manifests provide
    availability and provenance.  The latter is therefore checked separately
    and is never inferred from an HDF5 availability attribute.
    """
    release_root = Path(release_root)
    event_load_path = Path(event_load_path).resolve()
    selected_names = list(selected_names)
    package_manifest_path = release_root / "events" / "manifest.json"
    if not package_manifest_path.is_file():
        raise ValueError(f"missing event package manifest: {package_manifest_path}")
    package_manifest = json.loads(package_manifest_path.read_text())
    if package_manifest.get("format") != "ocean_global_event_load_v2":
        raise ValueError("event package format is not ocean_global_event_load_v2")
    availability = package_manifest.get("availability", {})
    if availability.get("online_native_supervision_authorized") is not False:
        raise ValueError("event package does not declare retrospective-only availability")
    fold_manifest = package_manifest.get("folds", {}).get(fold)
    if not isinstance(fold_manifest, dict) or "monthly_events_h5" not in fold_manifest:
        raise ValueError(f"event package manifest has no monthly target for fold {fold}")
    if fold_manifest.get("event_values_encoder_visible") is not False:
        raise ValueError("event fold manifest does not declare retrospective-only targets")
    relative = Path(fold_manifest["monthly_events_h5"])
    if relative.is_absolute():
        raise ValueError("event package path must be relative")
    declared_path = (release_root / "events" / relative).resolve()
    if declared_path != event_load_path:
        raise ValueError("event package manifest path does not match selected event file")
    if state_manifest is None:
        state_manifest_path = release_root / "folds" / fold / "manifest.json"
        if not state_manifest_path.is_file():
            raise ValueError(f"missing state fold manifest: {state_manifest_path}")
        state_manifest = json.loads(state_manifest_path.read_text())
    declared_state_names = state_manifest.get("event_target_names")
    if declared_state_names is not None and not set(selected_names).issubset(declared_state_names):
        raise ValueError("selected event names are absent from the state manifest")
    if state_manifest.get("event_targets_encoder_visible") is True:
        raise ValueError("event targets are declared encoder-visible")

    global_path = release_root / "global_state.h5"
    if not global_path.is_file():
        raise ValueError(f"missing global state source: {global_path}")
    with h5py.File(global_path, "r") as state, h5py.File(event_load_path, "r") as events:
        if "time" not in state or "time" not in events or not np.array_equal(state["time"][:], events["time"][:]):
            raise ValueError("state and event monthly clocks are not exactly aligned")
        names = _json_attr(events.attrs.get("target_names"), "target_names")
        roles = _json_attr(events.attrs.get("target_roles"), "target_roles")
        primary_names = roles.get("primary_per_source_summary", [])
        if not set(selected_names).issubset(set(names)):
            raise ValueError("selected event name is absent from produced event columns")
        if not set(selected_names).issubset(set(primary_names)):
            raise ValueError("selected event name is not a declared primary source summary")
        if events.attrs.get("clock") != "monthly summaries of co-occurrence on the same UTC calendar day; not exact-instant simultaneity":
            raise ValueError("event clock declaration does not match the monthly source contract")
        if "encoder_visible" not in events.attrs or bool(events.attrs["encoder_visible"]) is not False:
            raise ValueError("event producer does not declare encoder_visible=false")
        values, valid = events["values"], events["valid"]
        if values.shape != valid.shape or values.shape[0] != state["time"].shape[0]:
            raise ValueError("event values/valid shape is not aligned with global state")
        if value_stop is None:
            value_stop = values.shape[0]
        if not 0 <= value_stop <= values.shape[0]:
            raise ValueError("event value_stop exceeds the shared clock")
        columns = np.asarray(sorted(names.index(name) for name in selected_names), dtype=int)
        selected_values = values[:value_stop, columns]
        selected_valid = valid[:value_stop, columns].astype(bool)
        if not np.isfinite(selected_values[selected_valid]).all():
            raise ValueError("event values contain non-finite measured entries")
    return {
        "target_names": names,
        "target_roles": roles,
        "columns": [names.index(name) for name in selected_names],
        "package_manifest": str(package_manifest_path),
        "state_manifest_event_names": declared_state_names,
        "clock": "monthly summaries of co-occurrence on the same UTC calendar day; not exact-instant simultaneity",
        "encoder_visible": False,
    }
