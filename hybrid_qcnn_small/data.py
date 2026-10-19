"""Datasets for raw 1-D factors and feature-wise GASF/GADF images."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FEATURES, SEQ_LEN


def _scale_angles(series: np.ndarray) -> np.ndarray:
    lo, hi = float(series.min()), float(series.max())
    scaled = 2.0 * (series - lo) / max(hi - lo, 1e-8) - 1.0
    return np.arccos(np.clip(scaled, -1.0, 1.0))


def gasf(series: np.ndarray) -> np.ndarray:
    phi = _scale_angles(series)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def gadf(series: np.ndarray) -> np.ndarray:
    phi = _scale_angles(series)
    return np.sin(phi[:, None] - phi[None, :]).astype(np.float32)


def make_gaf(window: np.ndarray, mode: str) -> np.ndarray:
    channels = []
    for feature_index in range(window.shape[1]):
        series = window[:, feature_index]
        if mode in ("gasf", "both"):
            channels.append(gasf(series))
        if mode in ("gadf", "both"):
            channels.append(gadf(series))
    return np.stack(channels)


class HybridDataset(Dataset):
    """Return both representations so one split can serve all input modes."""

    def __init__(self, data: pd.DataFrame, timestamps: set[pd.Timestamp],
                 horizon: int, gaf_mode: str, input_mode: str):
        self.gaf_mode = gaf_mode
        self.input_mode = input_mode
        self.samples = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            values = group[FEATURES].to_numpy(np.float32)
            for end in range(SEQ_LEN - 1, len(group)):
                timestamp = group.loc[end, "timestamp"]
                if timestamp not in timestamps:
                    continue
                target = group.loc[end, f"target_{horizon}"]
                if not np.isfinite(target):
                    continue
                label = int(group.loc[end, f"label_binary_{horizon}"])
                window = values[end - SEQ_LEN + 1:end + 1]
                self.samples.append((window, label, symbol, timestamp, float(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        window, label, symbol, timestamp, target = self.samples[index]
        item = {
            "label": torch.tensor(label, dtype=torch.long),
            "symbol": symbol,
            "timestamp": str(timestamp),
            "target_return": torch.tensor(target, dtype=torch.float32),
        }
        if self.input_mode in ("conv1d", "fusion"):
            item["x1d"] = torch.from_numpy(window.copy())
        if self.input_mode in ("conv2d", "fusion"):
            item["x2d"] = torch.from_numpy(make_gaf(window, self.gaf_mode))
        return item
