"""Monthly fields and source-backed event observations."""
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .data import GlobalStateDataset
from .event_state_data import load_event_state_table


OCEAN_PROFILE = "ocean_factual"



EVENT_FAMILY_NAMES = (
    "tc_monthly_load",
    "sst_warm_area_fraction",
    "cuti_active_band_fraction",
    "carbon_transition_fraction",
    "carbon_sink_area_fraction",
    "nino34_trailing_class",
)



EVENT_LOGIT_NAMES = (
    "tc_monthly_load_lower", "tc_monthly_load_upper",
    "sst_warm_area_fraction_lower", "sst_warm_area_fraction_upper",
    "cuti_active_band_fraction_lower", "cuti_active_band_fraction_upper",
    "carbon_transition_fraction_lower", "carbon_transition_fraction_upper",
    "carbon_sink_area_fraction_lower", "carbon_sink_area_fraction_upper",
    "nino34_trailing_cold", "nino34_trailing_neutral", "nino34_trailing_warm",
)



def _split_origins(root, fold, purged):
    path = Path(root) / "folds" / fold / ("windows_purged.csv" if purged else "windows.csv")
    origins = {"train": [], "validation": [], "test": []}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in origins:
                raise ValueError(f"unknown Ocean split in {path}: {split}")
            origins[split].append(int(row["origin_index"]))
    for split, values in origins.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Ocean {split} origins must be present and unique")
    return path, origins



def _encode_event_truth(table):
    """Convert seven source features to five endpoint pairs plus Niño class."""
    lower = table["lower"].float()
    upper = table["upper"].float()
    valid = table["valid"].bool()
    if lower.shape != upper.shape or lower.shape != valid.shape or lower.ndim != 2 or lower.shape[1] != 7:
        raise ValueError("Ocean event table must contain seven aligned source features")
    if (lower > upper).any() or (lower < 0).any() or (upper > 1).any():
        raise ValueError("Ocean event bounds must be ordered unit intervals")

    endpoint_truth = torch.stack((lower[:, :5], upper[:, :5]), -1).flatten(1)
    nino_known = valid[:, 5] & valid[:, 6]
    nino_known &= lower[:, 5].eq(upper[:, 5]) & lower[:, 6].eq(upper[:, 6])
    warm, cold = lower[:, 5], lower[:, 6]
    if ((warm[nino_known] != 0) & (warm[nino_known] != 1)).any() or ((cold[nino_known] != 0) & (cold[nino_known] != 1)).any():
        raise ValueError("known Niño warm/cold targets must be binary")
    if (warm[nino_known] + cold[nino_known] > 1).any():
        raise ValueError("known Niño warm and cold targets cannot both be active")
    nino_truth = torch.stack((cold, 1 - warm - cold, warm), -1)
    nino_truth = torch.where(nino_known[:, None], nino_truth, torch.zeros_like(nino_truth))
    truth = torch.cat((endpoint_truth, nino_truth), -1).contiguous()
    known = torch.cat((valid[:, :5], nino_known[:, None]), -1).contiguous()
    if truth.shape[1] != 13 or known.shape[1] != 6 or not torch.isfinite(truth).all():
        raise ValueError("invalid unified Ocean event tensors")
    return truth, known



class UnifiedOceanDataset(Dataset):
    """Adapt an existing synchronized Ocean release to ``ocean_factual``.

    ``event_truth`` has shape ``[history + K, 13]`` in chronological order.
    Its first ten entries are five source lower/upper pairs and the final three
    are ``cold/neutral/warm``.  ``event_known`` is family-level ``[history + K,
    6]``; callers must expand it over endpoint/class logits before a loss.
    """

    profile = OCEAN_PROFILE
    event_family_names = EVENT_FAMILY_NAMES
    event_logit_names = EVENT_LOGIT_NAMES
    intervention_eligible = False

    def __init__(self, root, split="train", fold="main", purged=False, event_calibration=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Ocean split must be train, validation, or test")
        self.state = GlobalStateDataset(root, split=split, fold=fold, purged=purged)
        self.root, self.split, self.fold, self.purged = Path(root), split, fold, bool(purged)
        self.history, self.horizon = self.state.history, self.state.horizon
        if (self.history, self.horizon) != (12, 3):
            raise ValueError("ocean_factual requires exactly 12 history months and K=3")
        manifest_format = self.state.manifest.get("format")
        if manifest_format not in (None, "global_state_fold_v1"):
            raise ValueError(f"unsupported Ocean fold format: {manifest_format}")

        window_path, split_origins = _split_origins(self.root, fold, purged)
        if split_origins[split] != self.state.origins:
            raise ValueError("Ocean split origins drifted between adapter and state reader")
        declared_counts = self.state.manifest.get("purged_counts" if purged else "counts")
        actual_counts = {name: len(values) for name, values in split_origins.items()}
        if declared_counts is not None and {name: int(declared_counts[name]) for name in actual_counts} != actual_counts:
            raise ValueError("Ocean window counts drifted from the fold manifest")
        table = load_event_state_table(self.state, max_horizon=self.horizon, calibration=event_calibration)
        self.event_truth, self.event_known = _encode_event_truth(table)
        self.event_calibration = table["calibration"]
        self.event_metadata = table["metadata"]
        self.origins = self.state.origins
        self.mean, self.std = self.state.mean, self.state.std

    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        item = self.state[index]
        origin = int(item["origin_index"])
        lo = origin - self.history + 1
        hi = origin + self.horizon + 1
        if lo < 0 or hi > len(self.event_truth):
            raise ValueError("Ocean event timeline does not cover the factual state window")
        item["event_truth"] = self.event_truth[lo:hi].clone()
        item["event_known"] = self.event_known[lo:hi].clone()
        return item

    def close(self):
        self.state.close()
