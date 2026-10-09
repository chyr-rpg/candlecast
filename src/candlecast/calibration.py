"""Calibrated forecast bands.

Kronos's raw bands are far too narrow (its "80%" band held ~40% of actual outcomes in backtests). A
calibrated band at level L% is  median +/- k_L * raw half-band  (p90 - p50 above the median, p50 - p10
below), with k_L fitted so that L% of past actual end closes fell inside it
(benchmarks/fit_band_calibration.py -> config/band_calibration.json). The same k is applied to every
forecast bar - the fit uses end closes only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CALIBRATION_FILE = "config/band_calibration.json"
LEVELS = (40, 50, 60, 70, 80)


def load(root: Path) -> dict:
    path = Path(root) / CALIBRATION_FILE
    return json.loads(path.read_text()) if path.exists() else {"groups": {}}


def factor(calibration: dict, asset_class: str, level: int) -> float | None:
    """k for this asset class and level, or None when there is no calibration (raw bands are drawn)."""
    groups = calibration.get("groups", {})
    group = groups.get(asset_class) or groups.get("other")
    if not group:
        return None
    return group["k"].get(str(int(level)))


def group_size(calibration: dict, asset_class: str) -> int:
    groups = calibration.get("groups", {})
    group = groups.get(asset_class) or groups.get("other") or {}
    return int(group.get("n_forecasts", 0))


def band(summary: pd.DataFrame, k: float) -> tuple[pd.Series, pd.Series]:
    """(lower, upper) calibrated band per forecast bar from a bar_summary() frame."""
    median = summary["close_p50"]
    lower = median - k * (median - summary["close_p10"])
    upper = median + k * (summary["close_p90"] - median)
    return lower, upper


def coverage(test_summary: pd.DataFrame, k: float) -> float | None:
    """Share of actual closes inside the calibrated band in a test-window file (needs actual_close)."""
    if "actual_close" not in test_summary or test_summary["actual_close"].isna().all():
        return None
    lower, upper = band(test_summary, k)
    m = test_summary["actual_close"].notna()
    a = test_summary.loc[m, "actual_close"]
    return float(((a >= lower[m]) & (a <= upper[m])).mean())
