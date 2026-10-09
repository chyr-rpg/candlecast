"""Forecast one instrument from its cached bars and summarize the sampled paths."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from candlecast import kv_inference as kv
from candlecast.calendars import future_bars, to_model_time

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
HORIZON_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass
class ForecastResult:
    ticker: str
    asset_class: str
    as_of_bar: pd.Timestamp       # UTC open time of the last bar used
    last_close: float
    future_ts: pd.DatetimeIndex   # UTC open times of the forecast bars
    paths: np.ndarray             # [n_paths, pred_len, 6] in kv.FEATURES order, price units
    history_bars: int


def seed_for(ticker: str, as_of_bar: pd.Timestamp) -> int:
    """Deterministic per (ticker, as-of bar): a re-run of the same night gives the same paths."""
    return zlib.crc32(f"{ticker}|{pd.Timestamp(as_of_bar).isoformat()}".encode())


def forecast_from_bars(ticker: str, asset_class: str, bars: pd.DataFrame, predictor, model_settings,
                       pred_len: int, n_paths: int, future_ts: pd.DatetimeIndex | None = None,
                       history_len: int | None = None) -> ForecastResult:
    """``history_len`` defaults to ``model_settings.history_bars``; 0/None means the most the KV cache
    allows exactly (max_context - pred_len), which is also the cap."""
    history_len = history_len or getattr(model_settings, "history_bars", 0) or model_settings.max_context
    history_len = min(history_len, model_settings.max_context - pred_len)
    hist = bars.tail(history_len).reset_index(drop=True)
    if len(hist) < history_len:
        raise ValueError(f"{ticker}: only {len(hist)} bars, need {history_len}")
    as_of = hist["timestamp"].iloc[-1]
    if future_ts is None:
        future_ts = future_bars(asset_class, as_of, pred_len)
    torch.manual_seed(seed_for(ticker, as_of))
    paths = kv.forecast_paths(
        predictor, hist[kv.FEATURES], to_model_time(hist["timestamp"]), to_model_time(future_ts),
        pred_len, n_paths, T=model_settings.temperature, top_p=model_settings.top_p,
    )
    return ForecastResult(ticker, asset_class, as_of, float(hist["close"].iloc[-1]), future_ts, paths, len(hist))


def forecast_test_window(ticker: str, asset_class: str, bars: pd.DataFrame, predictor, model_settings,
                            pred_len: int, n_paths: int,
                            history_len: int | None = None) -> tuple[ForecastResult, pd.DataFrame]:
    """Out-of-sample test: forecast the last ``pred_len`` bars that already happened from the bars
    before them only. The forecast bars are the actual bar timestamps, so thin instruments (skipped
    empty hours) line up 1:1 with what traded. Returns (result, actual bars of the test window)."""
    if len(bars) <= pred_len:
        raise ValueError(f"{ticker}: only {len(bars)} bars, need more than {pred_len} for a test forecast")
    cut = len(bars) - pred_len
    actual = bars.iloc[cut:].reset_index(drop=True)
    result = forecast_from_bars(ticker, asset_class, bars.iloc[:cut], predictor, model_settings, pred_len,
                                n_paths, future_ts=pd.DatetimeIndex(actual["timestamp"]), history_len=history_len)
    return result, actual


def bar_summary(result: ForecastResult) -> pd.DataFrame:
    """One row per forecast bar: close quantiles, median high/low, share of paths above last close."""
    close = result.paths[:, :, 3]
    q = np.quantile(close, QUANTILES, axis=0)
    df = pd.DataFrame({"timestamp": result.future_ts, "step": np.arange(1, close.shape[1] + 1)})
    for level, row in zip(QUANTILES, q):
        df[f"close_p{int(level * 100):02d}"] = row
    df["close_mean"] = close.mean(axis=0)
    df["high_p50"] = np.median(result.paths[:, :, 1], axis=0)
    df["low_p50"] = np.median(result.paths[:, :, 2], axis=0)
    df["prob_above_last"] = (close > result.last_close).mean(axis=0)
    return df


def horizon_summary(result: ForecastResult) -> dict:
    """Whole-horizon numbers: end close, highest high and lowest low distributions, P(up)."""
    p = result.paths
    stats = {"end_close": p[:, -1, 3], "max_high": p[:, :, 1].max(axis=1), "min_low": p[:, :, 2].min(axis=1)}
    out = {"last_close": result.last_close, "prob_up": float((stats["end_close"] > result.last_close).mean())}
    for name, values in stats.items():
        for level, v in zip(HORIZON_QUANTILES, np.quantile(values, HORIZON_QUANTILES)):
            out[f"{name}_p{int(level * 100):02d}"] = float(v)
    return out


def paths_long(result: ForecastResult) -> pd.DataFrame:
    """Every sampled path in long format (path, step, timestamp, OHLCV, amount)."""
    n, steps, _ = result.paths.shape
    flat = result.paths.reshape(n * steps, -1)
    df = pd.DataFrame(flat, columns=kv.FEATURES).astype("float32")
    df.insert(0, "timestamp", np.tile(result.future_ts.to_numpy(), n))
    df.insert(0, "step", np.tile(np.arange(1, steps + 1, dtype=np.int16), n))
    df.insert(0, "path", np.repeat(np.arange(n, dtype=np.int16), steps))
    return df
