"""Forecast files and the sqlite index that consumers read.

Layout (run_date = New York date when the run started):
  outputs/forecasts/<run_date>/<TICKER>_summary.parquet   one row per forecast bar (kept forever)
  outputs/forecasts/<run_date>/<TICKER>_paths.parquet     every sampled path (pruned after N days)
  outputs/forecasts/<run_date>/<TICKER>_test.parquet      test forecast per bar + actual close (kept forever)
  outputs/forecasts/<run_date>/<TICKER>.png               chart (pruned after N days)
  outputs/forecast_index.db                               tables ``forecasts``, ``test_forecasts``, ``runs``

``forecasts`` has one row per (ticker, as_of_bar, model); a re-run for the same bar replaces it.
``test_forecasts`` has one row per live forecast that also made a test forecast: the forecast of the
last ``pred_len`` bars that already happened (``cutoff_bar`` = last bar it was allowed to see), scored
with scoring.score_outcome. It is kept apart from ``scores`` so the forward-test record stays pure.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

HORIZON_COLS = [
    "last_close", "prob_up",
    *(f"{s}_p{q}" for s in ("end_close", "max_high", "min_low") for q in ("10", "25", "50", "75", "90")),
]

# scoring.score_outcome keys stored for a test forecast (prob_up / median end copied from the forecast)
TEST_SCORE_COLS = [
    "prob_up", "end_close_p50", "actual_end", "abs_err_kronos", "abs_err_naive", "dir_hit", "brier",
    "cover_end", "cover_bars", "crps_kronos", "crps_rw",
]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS forecasts (
    ticker TEXT NOT NULL,
    as_of_bar TEXT NOT NULL,          -- UTC ISO open time of the last bar used
    model TEXT NOT NULL,
    run_id TEXT NOT NULL,
    run_date TEXT NOT NULL,           -- New York date of the run
    created_at TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    n_paths INTEGER NOT NULL,
    pred_len INTEGER NOT NULL,
    history_bars INTEGER NOT NULL,
    horizon_start TEXT NOT NULL,      -- UTC ISO open time of the first forecast bar
    horizon_end TEXT NOT NULL,        -- UTC ISO open time of the last forecast bar
    {", ".join(f"{c} REAL" for c in HORIZON_COLS)},
    summary_path TEXT NOT NULL,
    paths_path TEXT,
    chart_path TEXT,
    seconds REAL,
    PRIMARY KEY (ticker, as_of_bar, model)
);
CREATE INDEX IF NOT EXISTS forecasts_ticker_created ON forecasts (ticker, created_at);
CREATE TABLE IF NOT EXISTS test_forecasts (
    ticker TEXT NOT NULL,
    as_of_bar TEXT NOT NULL,          -- the live forecast's as_of_bar (the latest bar)
    model TEXT NOT NULL,
    cutoff_bar TEXT NOT NULL,         -- UTC ISO open time of the last bar the test forecast used
    cutoff_close REAL NOT NULL,
    test_start TEXT NOT NULL,         -- first and last actual bar of the test window
    test_end TEXT NOT NULL,
    n_paths INTEGER NOT NULL,
    pred_len INTEGER NOT NULL,
    {", ".join(f"{c} REAL" for c in TEST_SCORE_COLS)},
    summary_path TEXT NOT NULL,
    PRIMARY KEY (ticker, as_of_bar, model)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    model TEXT NOT NULL,
    n_instruments INTEGER,
    n_ok INTEGER,
    n_failed INTEGER,
    failures TEXT                     -- "TICKER: error" lines
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def run_dir(forecasts_dir: Path, run_date: str) -> Path:
    return Path(forecasts_dir) / run_date


def rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def record_forecast(con: sqlite3.Connection, row: dict) -> None:
    cols = list(row)
    con.execute(f"INSERT OR REPLACE INTO forecasts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[c] for c in cols])
    con.commit()


def record_test_forecast(con: sqlite3.Connection, row: dict) -> None:
    cols = list(row)
    con.execute(f"INSERT OR REPLACE INTO test_forecasts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[c] for c in cols])
    con.commit()


def latest_test(con: sqlite3.Connection, ticker: str, as_of_bar: str, model: str) -> dict | None:
    row = con.execute("SELECT * FROM test_forecasts WHERE ticker=? AND as_of_bar=? AND model=?",
                      (ticker.upper(), as_of_bar, model)).fetchone()
    return dict(row) if row else None


def start_run(con, run_id: str, run_date: str, started_at: str, model: str, n: int) -> None:
    con.execute("INSERT OR REPLACE INTO runs (run_id, run_date, started_at, model, n_instruments) VALUES (?,?,?,?,?)",
                (run_id, run_date, started_at, model, n))
    con.commit()


def finish_run(con, run_id: str, finished_at: str, n_ok: int, failures: list[str]) -> None:
    con.execute("UPDATE runs SET finished_at=?, n_ok=?, n_failed=?, failures=? WHERE run_id=?",
                (finished_at, n_ok, len(failures), "\n".join(failures), run_id))
    con.commit()


def latest(con: sqlite3.Connection, ticker: str, model: str | None = None) -> dict | None:
    sql = "SELECT * FROM forecasts WHERE ticker = ?"
    args: list = [ticker.upper()]
    if model:
        sql += " AND model = ?"
        args.append(model)
    row = con.execute(sql + " ORDER BY as_of_bar DESC, created_at DESC LIMIT 1", args).fetchone()
    return dict(row) if row else None


def prune(con: sqlite3.Connection, root: Path, forecasts_dir: Path, keep_days: int,
          today: pd.Timestamp) -> int:
    """Delete path files and charts in run folders older than ``keep_days``; summaries stay."""
    cutoff = (today - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d")
    removed = 0
    for folder in Path(forecasts_dir).glob("????-??-??"):
        if folder.name >= cutoff:
            continue
        for f in list(folder.glob("*_paths.parquet")) + list(folder.glob("*.png")):
            f.unlink()
            removed += 1
    con.execute("UPDATE forecasts SET paths_path = NULL, chart_path = NULL WHERE run_date < ?", (cutoff,))
    con.commit()
    return removed
