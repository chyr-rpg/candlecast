"""Normalized 1h OHLCV frames and the incremental local parquet cache.

Normalized frame: ``timestamp`` (UTC, tz-aware, bar OPEN time, on the hour) plus
``open, high, low, close, volume, amount``, sorted and unique by timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "amount"]
PRICE_COLS = ["open", "high", "low", "close"]
BAR = pd.Timedelta(hours=1)


def normalize(raw: pd.DataFrame, fetch_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Lower-case columns, UTC timestamps floored to the hour (collisions merged), drop bad rows.

    ``amount`` (traded value) = vwap * volume when the source gives a vwap, else volume * mean(OHLC)
    (the same fallback Kronos itself uses). Bars that hadn't closed by ``fetch_end`` are dropped.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS).astype({"timestamp": "datetime64[ns, UTC]"})
    df = raw.copy()
    if "timestamp" not in {str(c).lower() for c in df.columns}:
        df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"index": "timestamp", "datetime": "timestamp", "date": "timestamp"})
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts.dt.floor("h")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0).astype(float)
    if "vwap" in df.columns:
        df["amount"] = df["vwap"].astype(float) * df["volume"]
    else:
        df["amount"] = df[PRICE_COLS].astype(float).mean(axis=1) * df["volume"]

    df = df.dropna(subset=PRICE_COLS)
    df = df[(df[PRICE_COLS] > 0).all(axis=1)]
    df = df.sort_values("timestamp", kind="stable")
    df = (
        df.groupby("timestamp", sort=True)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
             volume=("volume", "sum"), amount=("amount", "sum"))
        .reset_index()
    )
    if fetch_end is not None:
        df = df[df["timestamp"] + BAR <= fetch_end]
    return df[COLUMNS].astype({c: float for c in COLUMNS[1:]}).reset_index(drop=True)


def cache_path(bars_dir: Path, ticker: str) -> Path:
    return Path(bars_dir) / f"{ticker}.parquet"


def read_cache(bars_dir: Path, ticker: str) -> pd.DataFrame | None:
    path = cache_path(bars_dir, ticker)
    return pd.read_parquet(path) if path.exists() else None


def write_cache(bars_dir: Path, ticker: str, df: pd.DataFrame) -> None:
    path = cache_path(bars_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


@dataclass
class MergeResult:
    frame: pd.DataFrame
    revised: bool        # overlap disagreed beyond tolerance => caller should re-download fully
    new_bars: int
    max_overlap_diff: float


def merge_incremental(cached: pd.DataFrame, fresh: pd.DataFrame, rtol: float) -> MergeResult:
    """Append ``fresh`` onto ``cached``; fresh rows win on overlap.

    If an overlapping close differs by more than ``rtol`` (relative), history was revised upstream
    (e.g. a split re-adjusted every past price), so stitching would create a fake jump.
    """
    overlap = cached.merge(fresh, on="timestamp", suffixes=("_old", "_new"))
    diff = 0.0
    if not overlap.empty:
        diff = float(np.max(np.abs(overlap["close_new"] / overlap["close_old"] - 1.0)))
    if diff > rtol:
        return MergeResult(cached, True, 0, diff)
    last_old = cached["timestamp"].max() if not cached.empty else None
    merged = pd.concat([cached[~cached["timestamp"].isin(fresh["timestamp"])], fresh], ignore_index=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    new = int((fresh["timestamp"] > last_old).sum()) if last_old is not None else len(fresh)
    return MergeResult(merged, False, new, diff)
