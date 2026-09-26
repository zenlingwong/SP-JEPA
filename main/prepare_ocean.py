#!/usr/bin/env python3
"""Fit event normalization from training-origin months of a prepared release."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.unified_ocean import UnifiedOceanDataset


def event_statistics(dataset):
    sums, counts = np.zeros((12, 13)), np.zeros((12, 13))
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        truth = torch.nan_to_num(item['event_truth'][dataset.history - 1].double())
        known = item['event_known'][dataset.history - 1].bool()
        known = torch.cat((known[:5].repeat_interleave(2), known[5:6].expand(3))).numpy()
        calendar = item['calendar'][-1]
        month = int(round(math.atan2(float(calendar[0]), float(calendar[1])) * 6 / math.pi)) % 12
        sums[month] += truth.numpy() * known
        counts[month] += known
        rows.append((month, truth, known))
    climatology = torch.from_numpy(sums / np.maximum(counts, 1))
    squared, count = np.zeros(13), np.zeros(13)
    for month, truth, known in rows:
        squared += (truth - climatology[month]).numpy() ** 2 * known
        count += known
    std = np.sqrt(squared / np.maximum(count, 1))
    return {'climatology_table': climatology.tolist(), 'climatology_counts': counts.tolist(),
            'anomaly_std': std.tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train = UnifiedOceanDataset(args.data_root, split='train', fold='main', purged=False)
    statistics = event_statistics(train)
    train.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(statistics, stream, indent=2)
