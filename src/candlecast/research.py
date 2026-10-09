"""Research helpers for benchmarks/accuracy_backtest.py: alternative bar series and volatility forecasts.

Bar series (``load_bars``), all in the normalized bar format (``timestamp`` = UTC bar-open time):
* ``1h``    - the hourly cache as the nightly run uses it (extended hours, 04:00-20:00 New York).
* ``rth1h`` - hourly bars opening 09:00-15:00 New York only (7 a day; the 09:00 bar still holds the
              09:00-09:30 pre-market half-hour, since Alpaca's hourly bars are aligned to the hour).
* ``4h``    - the extended-hours hourly bars aggregated into 04-08 / 08-12 / 12-16 / 16-20 New York
              buckets (4 a day); an unfinished latest bucket is dropped.
* ``1d``    - regular-session daily bars from benchmarks/fetch_daily.py (data/bars/1d).

Volatility (Kronos paper's task: realized volatility = sqrt(sum of squared log returns) over the
forecast window, from the last known close through each forecast bar):
* ``realized_vol`` of an actual or simulated close series;
* baselines from the history's log returns: ``hist_vol`` (equal-weight std), ``ewma_vol``
  (RiskMetrics lambda 0.94 per bar) and ``garch_vol`` (GARCH(1,1) with variance targeting, fitted
  by Gaussian likelihood on a grid - no extra dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from candlecast.bars import read_cache
from candlecast.calendars import NY

BAR_KINDS = ("1h", "rth1h", "4h", "1d")
# forecast steps that cover ~2 US sessions, the nightly horizon, per bar kind
TWO_SESSION_STEPS = {"1h": 32, "rth1h": 14, "4h": 8, "1d": 2}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}


def rth_hourly(bars: pd.DataFrame) -> pd.DataFrame:
    hour = bars["timestamp"].dt.tz_convert(NY).dt.hour
    return bars[(hour >= 9) & (hour <= 15)].reset_index(drop=True)


def four_hour(bars: pd.DataFrame) -> pd.DataFrame:
    ny = bars["timestamp"].dt.tz_convert(NY)
    hour = ny.dt.hour
    keep = (hour >= 4) & (hour < 20)
    b, ny, hour = bars[keep], ny[keep], hour[keep]
    start = ny.dt.normalize() + pd.to_timedelta(4 + 4 * ((hour - 4) // 4), unit="h")
    out = b.assign(timestamp=start.dt.tz_convert("UTC")).groupby("timestamp", sort=True).agg(_AGG).reset_index()
    last_hour_end = bars["timestamp"].max() + pd.Timedelta(hours=1)
    return out[out["timestamp"] + pd.Timedelta(hours=4) <= last_hour_end].reset_index(drop=True)


def load_bars(settings, ticker: str, kind: str) -> pd.DataFrame | None:
    if kind not in BAR_KINDS:
        raise ValueError(f"bar kind must be one of {BAR_KINDS}")
    if kind == "1d":
        return read_cache(Path(settings.bars_dir).parent / "1d", ticker)
    bars = read_cache(settings.bars_dir, ticker)
    if bars is None or kind == "1h":
        return bars
    return rth_hourly(bars) if kind == "rth1h" else four_hour(bars)


def realized_vol(closes: np.ndarray) -> np.ndarray:
    """sqrt(sum of squared log returns) along the last axis; ``closes`` includes the starting close."""
    r = np.diff(np.log(np.asarray(closes, dtype=float)), axis=-1)
    return np.sqrt((r * r).sum(axis=-1))


def _log_returns(history_close) -> np.ndarray:
    return np.diff(np.log(np.asarray(history_close, dtype=float)))


def hist_vol(history_close, steps: int) -> float:
    return float(_log_returns(history_close).std() * np.sqrt(steps))


def ewma_vol(history_close, steps: int, lam: float = 0.94) -> float:
    """RiskMetrics EWMA variance (flat forecast: the next-bar variance repeated ``steps`` times)."""
    r = _log_returns(history_close)
    var = r[: min(20, len(r))].var()  # seed with the first returns' sample variance
    for x in r:
        var = lam * var + (1 - lam) * x * x
    return float(np.sqrt(var * steps))


def garch_fit(r: np.ndarray) -> tuple[float, float, float, float]:
    """GARCH(1,1) on demeaned returns with variance targeting (omega = v * (1 - a - b)); (a, b) chosen
    on a grid by Gaussian log-likelihood. Returns (omega, alpha, beta, next-step variance)."""
    r = np.asarray(r, dtype=float) - np.mean(r)
    v = r.var()
    a_grid, b_grid = np.arange(0.01, 0.31, 0.01), np.arange(0.50, 0.995, 0.01)
    a, b = (g.ravel() for g in np.meshgrid(a_grid, b_grid))
    ok = a + b < 0.999
    a, b = a[ok], b[ok]
    omega = v * (1 - a - b)
    s2 = np.full(a.shape, v)
    ll = np.zeros(a.shape)
    for x in r:  # vectorized over the whole (a, b) grid
        ll -= 0.5 * (np.log(s2) + x * x / s2)
        s2 = omega + a * x * x + b * s2
    i = int(np.argmax(ll))
    return float(omega[i]), float(a[i]), float(b[i]), float(s2[i])


def garch_vol(history_close, steps: int) -> float:
    """Volatility over the next ``steps`` bars: sum of the GARCH(1,1) h-step variance forecasts."""
    omega, a, b, nxt = garch_fit(_log_returns(history_close))
    persistence = a + b
    long_run = omega / (1 - persistence)
    h = np.arange(steps)
    return float(np.sqrt(np.sum(long_run + persistence ** h * (nxt - long_run))))


def qlike(actual_vol, forecast_vol) -> np.ndarray:
    """QLIKE loss on variances (robust standard for volatility forecasts; 0 = perfect, lower = better)."""
    ratio = (np.asarray(actual_vol, dtype=float) / np.asarray(forecast_vol, dtype=float)) ** 2
    return ratio - np.log(ratio) - 1
