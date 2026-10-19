from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from torch.utils.data import Dataset

from config import DATA_DIR, FEATURES, GAF_COLORMAP, GAF_FRAMES, GAF_IMAGE_SIZE, GAF_WINDOW, NEUTRAL_THRESHOLD, SEQ_LEN, SYMBOLS

IMAGENET_MEAN=np.asarray([0.485,0.456,0.406],dtype=np.float32)[:,None,None]
IMAGENET_STD=np.asarray([0.229,0.224,0.225],dtype=np.float32)[:,None,None]


def load_panel() -> pd.DataFrame:
    parts = []
    for symbol in SYMBOLS:
        path = DATA_DIR / f"{symbol}_1min_recent30d.csv"
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
        numeric = ["open", "high", "low", "close", "volume", "amount"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        valid = frame[numeric].notna().all(axis=1) & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        valid &= frame.high >= frame[["open", "close", "low"]].max(axis=1)
        valid &= frame.low <= frame[["open", "close", "high"]].min(axis=1)
        frame = frame.loc[valid, ["timestamp", *numeric]].copy()
        frame["symbol"] = symbol
        parts.append(frame)
    panel = pd.concat(parts, ignore_index=True)
    counts = panel.groupby("timestamp").symbol.nunique()
    common = counts[counts == len(SYMBOLS)].index
    return panel[panel.timestamp.isin(common)].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for _, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("timestamp").copy()
        close, volume = g.close, g.volume.clip(lower=0)
        g["ret1"] = close.pct_change()
        for n in (1, 2, 5, 10, 20):
            g[f"mom{n}"] = close.pct_change(n)
        for n in (5, 10, 20):
            g[f"vol{n}"] = g.ret1.rolling(n).std()
        dv = close * volume
        for n in (5, 10, 20):
            g[f"dv_mean{n}"] = dv.rolling(n).mean()
            g[f"volume_mean{n}"] = volume.rolling(n).mean()
        g["log_volume"] = np.log1p(volume)
        g["log_dv"] = np.log1p(dv)
        # Direction targets: information through t predicts whether the close
        # at t+h is above or below Close[t]. Future closes are labels only.
        for horizon in (1, 5):
            future_time=g["timestamp"].shift(-horizon)
            target=close.shift(-horizon) / close - 1.0
            # Do not silently turn the lunch break or overnight gap into a
            # one/five-minute forecast. Only exact contiguous minute horizons
            # receive a label.
            contiguous=(future_time-g["timestamp"]==pd.Timedelta(minutes=horizon))
            g[f"target_{horizon}"] = target.where(contiguous)
        outputs.append(g)
    data = pd.concat(outputs, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    # Paper preprocessing: within each timestamp, 1/99% winsorization then z-score.
    for feature in FEATURES:
        groups = data.groupby("timestamp")[feature]
        lo = groups.quantile(0.01); hi = groups.quantile(0.99)
        clipped = data[feature].clip(data.timestamp.map(lo), data.timestamp.map(hi))
        clipped_groups = clipped.groupby(data.timestamp)
        mean = data.timestamp.map(clipped_groups.mean())
        std = data.timestamp.map(clipped_groups.std(ddof=0)).replace(0, 1.0)
        data[feature] = (clipped - mean) / std
    for horizon in (1, 5):
        target=data[f"target_{horizon}"]
        data[f"label_binary_{horizon}"]=(target>0).astype("int64")
        data[f"label_neutral_{horizon}"]=np.select(
            [target < -NEUTRAL_THRESHOLD, target > NEUTRAL_THRESHOLD], [0, 2], default=1
        ).astype("int64")
    return data.dropna(subset=FEATURES).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


class SequenceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, timestamps: set[pd.Timestamp], task: str, horizon: int=1):
        self.task = task; self.horizon=horizon; self.samples = []
        by_symbol = {s: g.sort_values("timestamp").reset_index(drop=True) for s, g in data.groupby("symbol")}
        for symbol, g in by_symbol.items():
            values = g[FEATURES].to_numpy(np.float32)
            for end in range(SEQ_LEN - 1, len(g)):
                if g.loc[end, "timestamp"] in timestamps:
                    target=g.loc[end,f"target_{horizon}"]
                    if not np.isfinite(target): continue
                    label = g.loc[end, f"label_binary_{horizon}" if task == "binary" else f"label_neutral_{horizon}"]
                    self.samples.append((values[end-SEQ_LEN+1:end+1], int(label), symbol, g.loc[end, "timestamp"], float(target)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y, s, t, r = self.samples[i]
        return torch.from_numpy(x), torch.tensor(y), s, str(t), torch.tensor(r, dtype=torch.float32)


def feature_series_to_gadf(series: np.ndarray) -> np.ndarray:
    """Convert one already preprocessed QTCNN feature window to a GADF image."""
    lo,hi=float(series.min()),float(series.max())
    scaled=2.0*(series-lo)/max(hi-lo,1e-8)-1.0
    phi=np.arccos(np.clip(scaled,-1.0,1.0))
    return np.sin(phi[:,None]-phi[None,:]).astype(np.float32)


class FeatureGafDataset(Dataset):
    """Encode each of the 17 original QTCNN features as one GADF channel."""
    def __init__(self,data:pd.DataFrame,timestamps:set[pd.Timestamp],task:str="binary",horizon:int=1):
        self.task=task;self.horizon=horizon;self.samples=[]
        for symbol,g in data.groupby("symbol"):
            g=g.sort_values("timestamp").reset_index(drop=True)
            values=g[FEATURES].to_numpy(np.float32)
            for end in range(SEQ_LEN-1,len(g)):
                if g.loc[end,"timestamp"] not in timestamps:continue
                target=g.loc[end,f"target_{horizon}"]
                if not np.isfinite(target):continue
                label=int(g.loc[end,f"label_binary_{horizon}" if task=="binary" else f"label_neutral_{horizon}"])
                self.samples.append((values[end-SEQ_LEN+1:end+1],label,symbol,g.loc[end,"timestamp"],float(target)))
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        x,y,s,t,r=self.samples[i]
        image=np.stack([feature_series_to_gadf(x[:,j]) for j in range(x.shape[1])])
        return torch.from_numpy(image),torch.tensor(y),s,str(t),torch.tensor(r,dtype=torch.float32)
