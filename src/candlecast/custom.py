"""Custom forecasts for the panel: any ticker, timeframe (1h/4h/1d/1w) and number of forecast bars.

One run = a forward forecast of the next N bars plus a test forecast of the last N bars that already
happened (same N, same history rule, only data before them), scored like the nightly test window.
History is Spec.history (default 128 bars, timeframes.history_bars); with too little data it is
shortened down to timeframes.MIN_USABLE_HISTORY bars (recorded in ``history_bars`` and the data note).

Files: outputs/custom/<TICKER>/<timeframe>_<N>bars_h<history>/<as_of>/ summary.parquet, test.parquet,
history.parquet (closes shown on the chart). Index: table ``custom_forecasts`` in the forecast index DB,
kept apart from the nightly ``forecasts``/``scores`` track record.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from candlecast import scoring, store, timeframes
from candlecast.charts import render
from candlecast.forecaster import bar_summary, forecast_from_bars, forecast_test_window, horizon_summary

log = logging.getLogger(__name__)

TEST_COLS = ["cutoff_bar", "cutoff_close", "test_start", "test_end", *store.TEST_SCORE_COLS]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS custom_forecasts (
    ticker TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    pred_len INTEGER NOT NULL,
    as_of_bar TEXT NOT NULL,
    model TEXT NOT NULL,
    n_paths INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    history_bars INTEGER NOT NULL,
    data_note TEXT,
    horizon_start TEXT NOT NULL,
    horizon_end TEXT NOT NULL,
    {", ".join(f"{c} REAL" for c in store.HORIZON_COLS)},
    test_cutoff_bar TEXT, test_cutoff_close REAL, test_test_start TEXT, test_test_end TEXT,
    {", ".join(f"test_{c} REAL" for c in store.TEST_SCORE_COLS)},
    summary_path TEXT NOT NULL,
    test_path TEXT,
    history_path TEXT NOT NULL,
    seconds REAL,
    data_hash TEXT,
    PRIMARY KEY (ticker, timeframe, pred_len, history_bars, as_of_bar, model)
);
"""


@dataclass(frozen=True)
class Spec:
    timeframe: str
    pred_len: int
    history: int = timeframes.DEFAULT_HISTORY

    @property
    def label(self) -> str:
        return f"{self.pred_len} x {timeframes.LABELS[self.timeframe]}"


def connect(db_path: Path):
    con = store.connect(db_path)
    pk = [r[1] for r in con.execute("PRAGMA table_info(custom_forecasts)").fetchall() if r[5]]
    if pk and "history_bars" not in pk:  # one-time upgrade: history became part of the key (v0.1.0 dev)
        con.execute("ALTER TABLE custom_forecasts RENAME TO custom_forecasts_old")
        con.executescript(SCHEMA)
        con.execute("INSERT OR REPLACE INTO custom_forecasts SELECT * FROM custom_forecasts_old")
        con.execute("DROP TABLE custom_forecasts_old")
        con.commit()
    con.executescript(SCHEMA)
    if "data_hash" not in [r[1] for r in con.execute("PRAGMA table_info(custom_forecasts)")]:
        con.execute("ALTER TABLE custom_forecasts ADD COLUMN data_hash TEXT")
        con.commit()
    return con


def data_fingerprint(bars: pd.DataFrame, n_bars: int) -> str:
    """Hash of the exact bars a forecast uses (history + test window). Sources sometimes revise recent bars
    without adding a new one; a saved forecast is only reused when this is unchanged."""
    import hashlib  # noqa: PLC0415

    tail = bars.tail(n_bars)[["timestamp", "open", "high", "low", "close", "volume"]]
    return hashlib.sha1(pd.util.hash_pandas_object(tail, index=False).to_numpy().tobytes()).hexdigest()[:16]


def _split(row: dict) -> tuple[dict, dict | None]:
    """DB row -> (forecast row, test row in the shape charts/describe expect)."""
    row = dict(row)
    if not row.get("test_path"):
        return row, None
    test = {c: row.get(f"test_{c}") for c in TEST_COLS}
    test["summary_path"] = row["test_path"]
    return row, test


def latest(con, ticker: str, spec: Spec, model: str) -> tuple[dict, dict | None] | None:
    r = con.execute("""SELECT * FROM custom_forecasts WHERE ticker=? AND timeframe=? AND pred_len=? AND model=?
                       AND history_bars=? ORDER BY as_of_bar DESC, created_at DESC LIMIT 1""",
                    (ticker.upper(), spec.timeframe, spec.pred_len, model,
                     timeframes.history_bars(spec.pred_len, history=spec.history))).fetchone()
    return _split(r) if r else None


def _existing(con, ticker, spec: Spec, as_of: pd.Timestamp, model: str, n_paths: int, history: int,
              data_hash: str):
    r = con.execute("""SELECT * FROM custom_forecasts WHERE ticker=? AND timeframe=? AND pred_len=?
                       AND history_bars=? AND as_of_bar=? AND model=? AND n_paths>=? AND data_hash=?""",
                    (ticker, spec.timeframe, spec.pred_len, history, as_of.isoformat(), model, n_paths,
                     data_hash)).fetchone()
    return _split(r) if r else None


def run_one(settings, inst, spec: Spec, predictor, n_paths: int, con, rerun: bool = False, source=None,
            now: pd.Timestamp | None = None) -> tuple[dict, dict | None, str]:
    """Forecast one instrument for ``spec``. Returns (row, test_row, status) with status "new"/"cached".
    Raises with a readable message when there is not enough data."""
    timeframes.validate(spec.timeframe, spec.pred_len)
    t0 = time.perf_counter()
    n = int(spec.pred_len)
    max_ctx = settings.model.max_context
    wanted_history = timeframes.history_bars(n, max_ctx, spec.history)
    bars, note = timeframes.load_bars(settings, inst, spec.timeframe, wanted_history + n, source=source, now=now)
    if bars is None or bars.empty:
        raise ValueError(f"no {spec.timeframe} bars available ({note})")
    history = min(wanted_history, len(bars) - n)
    if history < timeframes.MIN_USABLE_HISTORY:
        raise ValueError(f"only {len(bars)} {spec.timeframe} bars; {spec.label} needs at least "
                         f"{timeframes.MIN_USABLE_HISTORY + n} (ideally {wanted_history + n})")
    if history < wanted_history:
        note += f"; history shortened to {history} bars ({wanted_history} requested)"
    as_of = bars["timestamp"].iloc[-1]
    fingerprint = data_fingerprint(bars, history + n)
    if not rerun:
        found = _existing(con, inst.ticker, spec, as_of, settings.model.name, n_paths, history, fingerprint)
        if found:
            return (*found, "cached")

    future = timeframes.future_bars(inst.asset_class, spec.timeframe, as_of, n)
    res = forecast_from_bars(inst.ticker, inst.asset_class, bars, predictor, settings.model, n, n_paths,
                             future_ts=future, history_len=history)
    summary, horizon = bar_summary(res), horizon_summary(res)
    t_res, actual = forecast_test_window(inst.ticker, inst.asset_class, bars, predictor, settings.model, n,
                                         n_paths, history_len=history)
    t_summary, t_horizon = bar_summary(t_res), horizon_summary(t_res)
    before = bars[bars["timestamp"] <= t_res.as_of_bar]
    sigma = scoring.rw_sigma(before["close"].tail(history), t_res.last_close, n)
    scores = scoring.score_outcome(t_horizon, t_res.last_close, actual, t_res.paths[:, -1, 3], sigma, t_summary)
    scores.update(prob_up=t_horizon["prob_up"], end_close_p50=t_horizon["end_close_p50"])

    out = (settings.root / "outputs" / "custom" / inst.ticker / f"{spec.timeframe}_{n}bars_h{history}"
           / pd.Timestamp(as_of).strftime("%Y%m%dT%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out / "summary.parquet", index=False)
    t_summary.merge(actual[["timestamp", "close"]].rename(columns={"close": "actual_close"}),
                    on="timestamp", how="left").to_parquet(out / "test.parquet", index=False)
    shown = max(settings.chart_history_bars, n + 24)
    bars.tail(shown)[["timestamp", "close"]].to_parquet(out / "history.parquet", index=False)

    row = {
        "ticker": inst.ticker, "timeframe": spec.timeframe, "pred_len": n, "as_of_bar": as_of.isoformat(),
        "model": settings.model.name, "n_paths": n_paths, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "asset_class": inst.asset_class, "history_bars": history, "data_note": note,
        "horizon_start": res.future_ts[0].isoformat(), "horizon_end": res.future_ts[-1].isoformat(),
        **{c: horizon[c] for c in store.HORIZON_COLS},
        "test_cutoff_bar": t_res.as_of_bar.isoformat(), "test_cutoff_close": t_res.last_close,
        "test_test_start": t_res.future_ts[0].isoformat(), "test_test_end": t_res.future_ts[-1].isoformat(),
        **{f"test_{c}": scores[c] for c in store.TEST_SCORE_COLS},
        "summary_path": store.rel(out / "summary.parquet", settings.root),
        "test_path": store.rel(out / "test.parquet", settings.root),
        "history_path": store.rel(out / "history.parquet", settings.root),
        "seconds": round(time.perf_counter() - t0, 2), "data_hash": fingerprint,
    }
    cols = list(row)
    con.execute(f"INSERT OR REPLACE INTO custom_forecasts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[c] for c in cols])
    con.commit()
    return (*_split(row), "new")


def render_custom(settings, row: dict, test: dict | None, path, band_level: int | None,
                  theme: str = "light", header: bool = True) -> None:
    """Draw a saved custom forecast (no inference) with a calibrated band at ``band_level`` (0 = raw)."""
    from candlecast.pipeline import band_for  # noqa: PLC0415

    summary = pd.read_parquet(settings.root / row["summary_path"])
    hist = pd.read_parquet(settings.root / row["history_path"])
    test_obj = None
    test_summary = None
    if test and (settings.root / test["summary_path"]).exists():
        test_summary = pd.read_parquet(settings.root / test["summary_path"])
        test_obj = {"summary": test_summary, "row": test, "cutoff_bar": pd.Timestamp(test["cutoff_bar"]),
                    "cutoff_close": test["cutoff_close"]}
    band = band_for(settings, row["asset_class"], band_level, test_summary)
    render(Path(path), row["ticker"], hist, summary, {c: row[c] for c in store.HORIZON_COLS},
           f"Kronos-{row['model']}", row["n_paths"], test=test_obj, band=band, timeframe=row["timeframe"],
           theme=theme, header=header)
