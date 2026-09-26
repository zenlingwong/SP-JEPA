"""Eight non-overlapping monthly windows per training batch."""
import hashlib
import json
import math
import torch
from torch.utils.data import default_collate

BATCH_SIZE = 8


FORBIDDEN_OCEAN_KEYS = {
    "action", "changed_target", "changed_target_valid", "changed_event_truth",
    "changed_event_known", "changed_current_event_truth", "mask", "response",
}



class NonOverlappingOceanBatchSampler:
    """Random full-release sampler with an explicit resumable draw state.

    The serialized permutation is over every dataset row.  A draw scans forward
    from ``cursor`` and accepts a row only when its complete history-plus-target
    interval is disjoint from every row already in the batch.  When the current
    permutation cannot complete a batch, a new full permutation is generated.
    """

    kind = "nonoverlapping_ocean_b8_v1"

    def __init__(self, origins, history: int, horizon: int, seed: int, batch_size: int = BATCH_SIZE):
        self.origins = [int(origin) for origin in origins]
        self.history, self.horizon = int(history), int(horizon)
        self.seed, self.batch_size = int(seed), int(batch_size)
        if self.batch_size != BATCH_SIZE:
            raise ValueError("Ocean SIGReg requires exactly B=8")
        if len(self.origins) != len(set(self.origins)):
            raise ValueError("Ocean sampler origins must be unique")
        if self.history < 1 or self.horizon < 1:
            raise ValueError("history and horizon must be positive")
        intervals = sorted(self._interval(index) for index in range(len(self.origins)))
        maximum = 0
        last_end = -math.inf
        for start, end in sorted(intervals, key=lambda item: item[1]):
            if start > last_end:
                maximum += 1
                last_end = end
        if maximum < self.batch_size:
            raise ValueError("dataset cannot supply eight non-overlapping history-plus-target windows")
        encoded = json.dumps({"origins": self.origins, "history": self.history, "horizon": self.horizon}, separators=(",", ":"))
        self.origin_digest = hashlib.sha256(encoded.encode()).hexdigest()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.permutation = []
        self.cursor = 0
        self.draws = 0
        self._refresh()

    def _interval(self, index: int) -> tuple[int, int]:
        origin = self.origins[index]
        return origin - self.history + 1, origin + self.horizon

    @staticmethod
    def _disjoint(interval, accepted) -> bool:
        start, end = interval
        return all(end < other_start or other_end < start for other_start, other_end in accepted)

    def _refresh(self):
        self.permutation = torch.randperm(len(self.origins), generator=self.generator).tolist()
        self.cursor = 0

    def next_batch(self) -> list[int]:
        for _ in range(1000):
            selected, intervals = [], []
            while self.cursor < len(self.permutation):
                index = int(self.permutation[self.cursor])
                self.cursor += 1
                interval = self._interval(index)
                if self._disjoint(interval, intervals):
                    selected.append(index)
                    intervals.append(interval)
                    if len(selected) == self.batch_size:
                        self.draws += 1
                        return selected
            self._refresh()
        raise RuntimeError("failed to draw an Ocean B=8 non-overlapping batch")

    def state_dict(self) -> dict:
        return {
            "kind": self.kind,
            "origin_digest": self.origin_digest,
            "seed": self.seed,
            "history": self.history,
            "horizon": self.horizon,
            "batch_size": self.batch_size,
            "generator_state": self.generator.get_state(),
            "permutation": list(self.permutation),
            "cursor": int(self.cursor),
            "draws": int(self.draws),
        }

    def load_state_dict(self, state: dict):
        expected = {
            "kind": self.kind,
            "origin_digest": self.origin_digest,
            "seed": self.seed,
            "history": self.history,
            "horizon": self.horizon,
            "batch_size": self.batch_size,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"strict Ocean sampler mismatch for {key}")
        permutation = [int(index) for index in state["permutation"]]
        if sorted(permutation) != list(range(len(self.origins))):
            raise ValueError("checkpoint Ocean sampler permutation is invalid")
        cursor = int(state["cursor"])
        if not 0 <= cursor <= len(permutation):
            raise ValueError("checkpoint Ocean sampler cursor is invalid")
        self.generator.set_state(state["generator_state"])
        self.permutation = permutation
        self.cursor = cursor
        self.draws = int(state["draws"])



def collate_ocean_batch(dataset, indices, device: torch.device) -> dict:
    batch = default_collate([dataset[index] for index in indices])
    forbidden = FORBIDDEN_OCEAN_KEYS.intersection(batch)
    if forbidden:
        raise ValueError(f"Ocean factual batch contains prohibited changed/action keys: {sorted(forbidden)}")
    history, horizon = int(dataset.history), int(dataset.horizon)
    if batch["event_truth"].shape[1:] != (history + horizon, 13):
        raise ValueError("Ocean event truth must be [B,history+K,13]")
    if batch["event_known"].shape[1:] != (history + horizon, 6):
        raise ValueError("Ocean event known mask must be [B,history+K,6]")
    batch["current_event_truth"] = batch["event_truth"][:, history - 1]
    batch["current_event_known"] = batch["event_known"][:, history - 1]
    batch["future_event_truth"] = batch["event_truth"][:, history:history + horizon]
    batch["future_event_known"] = batch["event_known"][:, history:history + horizon]
    for source, target in (
        ("x", "sigreg_x"), ("valid", "sigreg_valid"), ("geometry", "sigreg_geometry"),
        ("calendar", "sigreg_calendar"), ("target", "sigreg_target"),
        ("target_valid", "sigreg_target_valid"), ("future_calendar", "sigreg_future_calendar"),
    ):
        batch[target] = batch[source]
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
