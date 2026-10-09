"""Forward-test scoring: grade each stored forecast once its horizon has passed.

A forecast is scored when the instrument's cached bars reach its last forecast bar
(``horizon_end``). Actual bars are matched by timestamp; thin stocks may have fewer actual bars
than forecast bars, which is recorded (``n_actual_bars``) rather than treated as an error.

Metrics (shared with benchmarks/accuracy_backtest.py):
* abs_err_kronos / abs_err_naive: |median end close - actual| and |last close - actual|, / last close
* dir_hit: sign of the median move matches the actual move
* brier: (P(up) - outcome)^2; a coin flip scores 0.25
* cover_end / cover_high / cover_low: actual end close, highest high, lowest low inside p10-p90
* cover_bars: share of actual bars whose close is inside that bar's p10-p90 band
* crps_kronos / crps_rw: CRPS of the end close from the sampled paths, and from a random walk
  scaled to the history's hourly volatility (both / last close; kronos is NULL once paths are pruned)
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from candlecast.bars import read_cache

SCORES_SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    ticker TEXT NOT NULL,
    as_of_bar TEXT NOT NULL,
    model TEXT NOT NULL,
    scored_at TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    n_forecast_bars INTEGER NOT NULL,
    n_actual_bars INTEGER NOT NULL,
    actual_end REAL, actual_max_high REAL, actual_min_low REAL,
    abs_err_kronos REAL, abs_err_naive REAL, dir_hit REAL, brier REAL,
    cover_end REAL, cover_high REAL, cover_low REAL, cover_bars REAL,
    crps_kronos REAL, crps_rw REAL,
    PRIMARY KEY (ticker, as_of_bar, model)
);
"""


def crps_ensemble(samples: np.ndarray, y: float) -> float:
    s = np.sort(np.asarray(samples, dtype=float))
    n = len(s)
    # E|X - y| - 0.5 E|X - X'|, with E|X - X'| = 2/n^2 * sum_i (2i - n - 1) x_(i)
    return float(np.abs(s - y).mean() - np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / n**2)


def crps_normal(mu: float, sigma: float, y: float) -> float:
    z = (y - mu) / sigma
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return float(sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi)))


def rw_sigma(history_close: pd.Series, last: float, pred_len: int) -> float:
    logret = np.diff(np.log(np.asarray(history_close, dtype=float)))
    return float(last * logret.std() * math.sqrt(pred_len))


def score_outcome(h: dict, last: float, actual: pd.DataFrame, end_samples: np.ndarray | None,
                  sigma: float, bar_bands: pd.DataFrame | None = None) -> dict:
    """``h``: horizon summary (end_close_p10..., prob_up); ``actual``: bars inside the horizon."""
    y_end, y_hi, y_lo = float(actual["close"].iloc[-1]), float(actual["high"].max()), float(actual["low"].min())
    out = {
        "actual_end": y_end, "actual_max_high": y_hi, "actual_min_low": y_lo,
        "abs_err_kronos": abs(h["end_close_p50"] - y_end) / last,
        "abs_err_naive": abs(last - y_end) / last,
        "dir_hit": float(np.sign(h["end_close_p50"] - last) == np.sign(y_end - last)),
        "brier": (h["prob_up"] - float(y_end > last)) ** 2,
        "cover_end": float(h["end_close_p10"] <= y_end <= h["end_close_p90"]),
        "cover_high": float(h["max_high_p10"] <= y_hi <= h["max_high_p90"]),
        "cover_low": float(h["min_low_p10"] <= y_lo <= h["min_low_p90"]),
        "crps_kronos": crps_ensemble(end_samples, y_end) / last if end_samples is not None else None,
        "crps_rw": crps_normal(last, sigma, y_end) / last if sigma > 0 else None,
        "cover_bars": None,
    }
    if bar_bands is not None:
        m = actual.merge(bar_bands[["timestamp", "close_p10", "close_p90"]], on="timestamp")
        if len(m):
            out["cover_bars"] = float(((m["close"] >= m["close_p10"]) & (m["close"] <= m["close_p90"])).mean())
    return out


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCORES_SCHEMA)


def pending(con: sqlite3.Connection) -> list[dict]:
    ensure_schema(con)
    rows = con.execute(
        """SELECT f.* FROM forecasts f LEFT JOIN scores s
           ON s.ticker = f.ticker AND s.as_of_bar = f.as_of_bar AND s.model = f.model
           WHERE s.ticker IS NULL ORDER BY f.as_of_bar"""
    ).fetchall()
    return [dict(r) for r in rows]


def score_pending(con: sqlite3.Connection, settings, now: pd.Timestamp | None = None) -> int:
    """Score every unscored forecast whose horizon the bar cache now covers. Returns # scored."""
    now = now or pd.Timestamp.now(tz="UTC")
    n = 0
    bars_by_ticker: dict[str, pd.DataFrame | None] = {}
    for f in pending(con):
        if f["ticker"] not in bars_by_ticker:
            bars_by_ticker[f["ticker"]] = read_cache(settings.bars_dir, f["ticker"])
        bars = bars_by_ticker[f["ticker"]]
        if bars is None or bars.empty:
            continue
        start, end, as_of = (pd.Timestamp(f[k]) for k in ("horizon_start", "horizon_end", "as_of_bar"))
        if bars["timestamp"].max() < end:
            continue  # horizon not finished in the cache yet
        actual = bars[(bars["timestamp"] >= start) & (bars["timestamp"] <= end)]
        # random-walk volatility from the full KV-cache window (as in the benchmark), not the model's history
        hist = bars[bars["timestamp"] <= as_of].tail(settings.model.max_context - f["pred_len"])
        end_samples = bar_bands = None
        if f.get("paths_path") and (settings.root / f["paths_path"]).exists():
            paths = pd.read_parquet(settings.root / f["paths_path"])
            end_samples = paths.loc[paths["step"] == paths["step"].max(), "close"].to_numpy()
        if (settings.root / f["summary_path"]).exists():
            bar_bands = pd.read_parquet(settings.root / f["summary_path"])
        row = {"ticker": f["ticker"], "as_of_bar": f["as_of_bar"], "model": f["model"],
               "scored_at": now.isoformat(), "asset_class": f["asset_class"],
               "n_forecast_bars": f["pred_len"], "n_actual_bars": len(actual)}
        if len(actual):
            row.update(score_outcome(f, f["last_close"], actual, end_samples,
                                     rw_sigma(hist["close"], f["last_close"], f["pred_len"]), bar_bands))
        cols = list(row)
        con.execute(f"INSERT OR REPLACE INTO scores ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [row[c] for c in cols])
        n += 1
    con.commit()
    return n


def summarize(df: pd.DataFrame) -> pd.Series:
    """Aggregate scored rows (same columns as benchmarks/accuracy_backtest.py)."""
    def mean(col):
        return df[col].dropna().mean() if col in df and df[col].notna().any() else float("nan")

    crps = df.dropna(subset=["crps_kronos", "crps_rw"]) if "crps_kronos" in df else df.iloc[0:0]
    return pd.Series({
        "n": len(df),
        "MAE kronos %": 100 * mean("abs_err_kronos"),
        "MAE no-change %": 100 * mean("abs_err_naive"),
        "MAE ratio": mean("abs_err_kronos") / mean("abs_err_naive"),
        "direction hit": mean("dir_hit"),
        "Brier (coin=0.25)": mean("brier"),
        "cover end (80%)": mean("cover_end"),
        "cover high (80%)": mean("cover_high"),
        "cover low (80%)": mean("cover_low"),
        "cover bars (80%)": mean("cover_bars"),
        "CRPS ratio vs RW": (crps["crps_kronos"].mean() / crps["crps_rw"].mean()) if len(crps) else float("nan"),
    })


def rank_ic_by_group(df: pd.DataFrame, group_col: str, signal_col: str, actual_col: str = "actual_ret",
                     min_n: int = 5) -> pd.Series:
    """RankIC per group (one date / one run): Spearman correlation, ACROSS instruments, between the
    forecast signal and the actual return - the cross-sectional ranking measure the Kronos paper reports.
    Groups with fewer than ``min_n`` instruments (or no variation) are skipped."""
    out = {}
    for key, g in df.dropna(subset=[signal_col, actual_col]).groupby(group_col):
        if len(g) < min_n or g[signal_col].nunique() < 2 or g[actual_col].nunique() < 2:
            continue
        out[key] = g[signal_col].rank().corr(g[actual_col].rank())
    return pd.Series(out, dtype=float)


def describe_rank_ic(per_group: pd.Series) -> str:
    """Mean RankIC with its t-statistic (mean / standard error over groups) and share of groups > 0."""
    n = len(per_group)
    if n == 0:
        return "no dates with enough instruments yet"
    mean = per_group.mean()
    sd = per_group.std(ddof=1) if n > 1 else float("nan")
    t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    return f"mean RankIC {mean:+.3f}  (t = {t:+.2f}, {n} dates, positive on {(per_group > 0).mean():.0%})"


def rank_ic_report(con: sqlite3.Connection, root, model: str | None = None, since: str | None = None,
                   asset_class: str = "us_equity_extended") -> tuple[pd.Series, pd.Series]:
    """Live ranking skill of scored forecasts: per nightly run, rank the instruments of one asset class
    (one shared horizon) by expected return and correlate with what happened. Expected return uses the
    mean of the sampled closes at the last forecast bar (``close_mean`` in the kept summary file), as in
    the Kronos paper, and also the median. Returns (per-run RankIC by mean, per-run RankIC by median)."""
    ensure_schema(con)
    sql = """SELECT f.run_id, f.ticker, f.last_close, f.end_close_p50, f.summary_path, s.actual_end
             FROM forecasts f JOIN scores s
               ON s.ticker = f.ticker AND s.as_of_bar = f.as_of_bar AND s.model = f.model
             WHERE s.n_actual_bars > 0 AND f.asset_class = ?"""
    args: list = [asset_class]
    if model:
        sql += " AND f.model = ?"
        args.append(model)
    if since:
        sql += " AND f.as_of_bar >= ?"
        args.append(pd.Timestamp(since, tz="UTC").isoformat())
    df = pd.read_sql_query(sql, con, params=args)
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    def end_mean(path: str) -> float:
        try:
            return float(pd.read_parquet(Path(root) / path, columns=["close_mean"])["close_mean"].iloc[-1])
        except Exception:  # summary missing: fall back to NaN (that instrument drops out of the mean ranking)
            return float("nan")

    df["exp_ret_mean"] = df["summary_path"].map(end_mean) / df["last_close"] - 1
    df["exp_ret_p50"] = df["end_close_p50"] / df["last_close"] - 1
    df["actual_ret"] = df["actual_end"] / df["last_close"] - 1
    return rank_ic_by_group(df, "run_id", "exp_ret_mean"), rank_ic_by_group(df, "run_id", "exp_ret_p50")


def report(con: sqlite3.Connection, model: str | None = None, since: str | None = None,
           by_ticker: bool = False) -> pd.DataFrame:
    ensure_schema(con)
    sql = "SELECT * FROM scores WHERE n_actual_bars > 0"
    args: list = []
    if model:
        sql += " AND model = ?"
        args.append(model)
    if since:
        sql += " AND as_of_bar >= ?"
        args.append(pd.Timestamp(since, tz="UTC").isoformat())
    df = pd.read_sql_query(sql, con, params=args)
    if df.empty:
        return pd.DataFrame()
    key = "ticker" if by_ticker else "asset_class"
    parts = {k: summarize(g) for k, g in df.groupby(key)}
    parts["ALL"] = summarize(df)
    return pd.DataFrame(parts).T
