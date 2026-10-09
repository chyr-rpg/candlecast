"""Fit the band-widening factors used by the charts and the panel (config/band_calibration.json).

Kronos's raw p10-p90 band caught only ~40% of actual end closes in the backtests. For each band level
L (40..80%) we find k_L such that L% of past actual end closes lie within
    median +/- k_L * (raw half-band on that side: p90 - p50 above, p50 - p10 below).
Fitted on the backtest forecasts made with the nightly settings (1h bars, 240-bar history, 32/48-bar
horizon, T 1.0, top_p 1.0); validated by fitting on the earlier half of the dates and checking the later
half. US equities get their own factors; the few crypto/commodity/FX forecasts are pooled as "other".

Usage: .venv\\Scripts\\python benchmarks\\fit_band_calibration.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from candlecast.calibration import CALIBRATION_FILE, LEVELS
from candlecast.settings import load_settings

SOURCES = [
    "benchmarks/results/accuracy_20261007_2221_q6_1h_h240_p32_vol.csv",   # 10 US equities x 40 dates
    "benchmarks/results/accuracy_20261003_1851_C_topp10_hist240.csv",     # 12 instruments x 20 dates
]


def z_scores(d: pd.DataFrame) -> pd.Series:
    up = d["actual_end"] >= d["end_close_p50"]
    half = np.where(up, d["end_close_p90"] - d["end_close_p50"], d["end_close_p50"] - d["end_close_p10"])
    return (d["actual_end"] - d["end_close_p50"]).abs() / half


def main() -> None:
    settings = load_settings()
    d = pd.concat([pd.read_csv(settings.root / f) for f in SOURCES], ignore_index=True)
    d["origin"] = pd.to_datetime(d["origin"], utc=True)
    d["z"] = z_scores(d)
    d["group"] = np.where(d["asset_class"] == "us_equity_extended", "us_equity_extended", "other")
    out = {"_about": ("Band widening factors k per band level: band = median +/- k * raw half-band "
                      "(p90-p50 above, p50-p10 below). Fitted by benchmarks/fit_band_calibration.py on "
                      + ", ".join(SOURCES) + "."), "groups": {}}
    cut = d["origin"].quantile(0.5)
    for group, g in d.groupby("group"):
        tr, te = g[g["origin"] <= cut], g[g["origin"] > cut]
        factors, check = {}, {}
        for level in LEVELS:
            factors[str(level)] = round(float(np.quantile(g["z"], level / 100)), 3)
            if len(tr) >= 20 and len(te) >= 20:
                check[str(level)] = round(float((te["z"] <= np.quantile(tr["z"], level / 100)).mean()), 3)
        out["groups"][group] = {"n_forecasts": int(len(g)), "k": factors,
                                "held_out_coverage_when_fitted_on_earlier_half": check,
                                "raw_80pct_band_coverage": round(float((g["z"] <= 1).mean()), 3)}
        print(group, len(g), factors, "held-out", check)
    path = settings.root / CALIBRATION_FILE
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
