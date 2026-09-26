"""Direct reader for synchronized global monthly fields."""

import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def calendar_features(indices):
    angle = 2 * np.pi * np.asarray(indices) / 12
    return np.stack((np.sin(angle), np.cos(angle)), -1).astype("float32")


class GlobalStateDataset(Dataset):
    def __init__(self, root, split="train", fold="main", purged=False):
        self.root, self.split, self.fold, self.purged = Path(root), split, fold, purged
        self.manifest = json.loads((self.root / "folds" / fold / "manifest.json").read_text())
        self.history, self.horizon = self.manifest["history"], self.manifest["horizon"]
        filename = "windows_purged.csv" if purged else "windows.csv"
        with (self.root / "folds" / fold / filename).open(newline="") as stream:
            self.origins = [int(row["origin_index"]) for row in csv.DictReader(stream) if row["split"] == split]
        self.mean = np.asarray(self.manifest["state_mean"], dtype="float32")[:, None, None]
        self.std = np.asarray(self.manifest["state_std"], dtype="float32")[:, None, None]
        if not self.origins or len(set(self.origins)) != len(self.origins):
            raise ValueError("global time origins must be present and unique")
        if self.mean.shape != (5, 1, 1) or self.std.shape != (5, 1, 1) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("invalid training normalization")
        with h5py.File(self.root / "global_state.h5") as source:
            if source["state"].shape != (372, 5, 90, 180) or source["state_valid"].shape != (372, 90, 180):
                raise ValueError("expected synchronized 372x5x90x180 global state")
            lat, lon = np.meshgrid(source["latitude"][:], source["longitude"][:], indexing="ij")
            self.geometry = np.stack((np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)), np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)))).astype("float32")
            self.area = source["cell_area_km2"][:].astype("float32")[None]
        self.handle, self.pid = None, None

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        if self.handle is None or self.pid != os.getpid():
            if self.handle is not None:
                self.handle.close()
            self.handle = h5py.File(self.root / "global_state.h5", "r")
            self.pid = os.getpid()
        t = self.origins[index]
        lo, hi = self.manifest["splits"][self.split]
        if t + 1 < lo or t + self.horizon >= hi or (self.purged and t - self.history + 1 < lo):
            raise ValueError("window violates its chronological split")
        values = self.handle["state"][t - self.history + 1:t + self.horizon + 1].astype("float32")
        state_valid = self.handle["state_valid"][t - self.history + 1:t + self.horizon + 1].astype(bool)
        valid = np.broadcast_to(state_valid[:, None], values.shape).copy()
        values = np.where(valid, (values - self.mean) / self.std, 0).astype("float32")
        out = {"x": values[:self.history], "valid": valid[:self.history], "target": values[self.history:], "target_valid": valid[self.history:], "calendar": calendar_features(range(t - self.history + 1, t + 1)), "future_calendar": calendar_features(range(t + 1, t + self.horizon + 1)), "geometry": self.geometry, "area": self.area, "origin_index": np.int64(t)}
        return {key: torch.as_tensor(value) for key, value in out.items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handle"], state["pid"] = None, None
        return state

    def close(self):
        if self.handle is not None:
            self.handle.close()
        self.handle, self.pid = None, None
