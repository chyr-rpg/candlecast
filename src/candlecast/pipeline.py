"""Nightly run: refresh bars, forecast every instrument, save files + chart, index, prune."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

from candlecast import calibration, scoring, store
from candlecast.bars import read_cache
from candlecast.calendars import NY
from candlecast.charts import render
from candlecast.data_update import update_all
from candlecast.forecaster import (
    bar_summary,
    forecast_from_bars,
    horizon_summary,
    paths_long,
    forecast_test_window,
)

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    run_id: str
    run_date: str
    ok: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    scored: int = 0
    seconds: float = 0.0


def already_forecast(con, ticker: str, as_of_bar: pd.Timestamp, model: str, n_paths: int) -> bool:
    row = con.execute("SELECT n_paths FROM forecasts WHERE ticker=? AND as_of_bar=? AND model=?",
                      (ticker, as_of_bar.isoformat(), model)).fetchone()
    return row is not None and row[0] >= n_paths


def forecast_one(inst, settings, predictor, n_paths: int, run_id: str, run_date: str, con,
                 skip_existing: bool = False) -> dict | None:
    """Returns the index row, or None when ``skip_existing`` and the latest bar is already forecast."""
    t0 = time.perf_counter()
    bars = read_cache(settings.bars_dir, inst.ticker)
    if bars is None or bars.empty:
        raise ValueError("no cached bars")
    if skip_existing and already_forecast(con, inst.ticker, bars["timestamp"].iloc[-1], settings.model.name, n_paths):
        return None
    pred_len = settings.horizons[inst.asset_class]
    res = forecast_from_bars(inst.ticker, inst.asset_class, bars, predictor, settings.model, pred_len, n_paths)
    summary = bar_summary(res)
    horizon = horizon_summary(res)

    out = store.run_dir(settings.forecasts_dir, run_date)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"{inst.ticker}_summary.parquet"
    paths_path = out / f"{inst.ticker}_paths.parquet"
    chart_path = out / f"{inst.ticker}.png"
    summary.to_parquet(summary_path, index=False)
    paths_long(res).to_parquet(paths_path, index=False)

    test = None
    if settings.test_forecast:
        try:  # the test forecast only adds context; its failure never blocks the live forecast
            test = run_test_forecast(inst, settings, predictor, bars, pred_len, n_paths, out)
        except Exception:
            log.exception("%s: test forecast failed (live forecast unaffected)", inst.ticker)
    # with a test window, show at least ~1.5 sessions of history before the test forecast's cutoff
    n_hist = max(settings.chart_history_bars, pred_len + 24) if test else settings.chart_history_bars
    hist = bars.tail(n_hist)[["timestamp", "close"]].reset_index(drop=True)
    band = band_for(settings, inst.asset_class, settings.chart_band_level, test["summary"] if test else None)
    render(chart_path, inst.ticker, hist, summary, horizon, f"Kronos-{settings.model.name}", n_paths,
           test=test, band=band)

    row = {
        "ticker": inst.ticker, "as_of_bar": res.as_of_bar.isoformat(), "model": settings.model.name,
        "run_id": run_id, "run_date": run_date, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "asset_class": inst.asset_class, "n_paths": n_paths, "pred_len": pred_len,
        "history_bars": res.history_bars, "horizon_start": res.future_ts[0].isoformat(),
        "horizon_end": res.future_ts[-1].isoformat(), **{c: horizon[c] for c in store.HORIZON_COLS},
        "summary_path": store.rel(summary_path, settings.root), "paths_path": store.rel(paths_path, settings.root),
        "chart_path": store.rel(chart_path, settings.root), "seconds": round(time.perf_counter() - t0, 2),
    }
    store.record_forecast(con, row)
    if test:
        store.record_test_forecast(con, {"ticker": inst.ticker, "as_of_bar": row["as_of_bar"],
                                         "model": settings.model.name, **test["row"],
                                         "summary_path": store.rel(test["path"], settings.root)})
    return row


def band_for(settings, asset_class: str, level: int | None, test_summary: pd.DataFrame | None = None):
    """Chart band settings for a calibrated ``level``% band, or None (raw Kronos bands) when ``level`` is
    0/None or no calibration exists for this asset class."""
    if not level:
        return None
    k = calibration.factor(calibration.load(settings.root), asset_class, level)
    if k is None:
        return None
    cov = calibration.coverage(test_summary, k) if test_summary is not None else None
    return {"level": int(level), "k": k, "test_coverage": cov}


def render_saved(settings, row: dict, test_row: dict | None, path, band_level: int | None) -> None:
    """Re-draw a saved forecast (no inference) - e.g. at another band level, or for export."""
    summary = pd.read_parquet(settings.root / row["summary_path"])
    bars = read_cache(settings.bars_dir, row["ticker"])
    bars = bars[bars["timestamp"] <= pd.Timestamp(row["as_of_bar"])]
    test = None
    if test_row and (settings.root / test_row["summary_path"]).exists():
        test = {"summary": pd.read_parquet(settings.root / test_row["summary_path"]), "row": test_row,
                "cutoff_bar": pd.Timestamp(test_row["cutoff_bar"]), "cutoff_close": test_row["cutoff_close"]}
    n_hist = max(settings.chart_history_bars, row["pred_len"] + 24) if test else settings.chart_history_bars
    hist = bars.tail(n_hist)[["timestamp", "close"]].reset_index(drop=True)
    band = band_for(settings, row["asset_class"], band_level, test["summary"] if test else None)
    render(Path(path), row["ticker"], hist, summary, {c: row[c] for c in store.HORIZON_COLS},
           f"Kronos-{row['model']}", row["n_paths"], test=test, band=band)


def run_test_forecast(inst, settings, predictor, bars: pd.DataFrame, pred_len: int, n_paths: int,
                      out) -> dict:
    """Forecast the last ``pred_len`` bars that already happened from the bars before them, score it
    against what actually traded (same metrics as the forward-test record) and save the per-bar file.
    Returns what the chart and the ``test_forecasts`` row need."""
    res, actual = forecast_test_window(inst.ticker, inst.asset_class, bars, predictor, settings.model,
                                          pred_len, n_paths)
    summary = bar_summary(res)
    horizon = horizon_summary(res)
    before = bars[bars["timestamp"] <= res.as_of_bar]
    sigma = scoring.rw_sigma(before["close"].tail(settings.model.max_context - pred_len), res.last_close, pred_len)
    scores = scoring.score_outcome(horizon, res.last_close, actual, res.paths[:, -1, 3], sigma, summary)

    per_bar = summary.merge(actual[["timestamp", "close"]].rename(columns={"close": "actual_close"}),
                            on="timestamp", how="left")
    path = out / f"{inst.ticker}_test.parquet"
    per_bar.to_parquet(path, index=False)
    scores = {**scores, "prob_up": horizon["prob_up"], "end_close_p50": horizon["end_close_p50"]}
    row = {"cutoff_bar": res.as_of_bar.isoformat(), "cutoff_close": res.last_close,
           "test_start": res.future_ts[0].isoformat(), "test_end": res.future_ts[-1].isoformat(),
           "n_paths": n_paths, "pred_len": pred_len, **{c: scores[c] for c in store.TEST_SCORE_COLS}}
    return {"summary": per_bar, "row": row, "path": path, "cutoff_bar": res.as_of_bar,
            "cutoff_close": res.last_close}


def run(settings, instruments, n_paths: int | None = None, update: bool = True, predictor=None,
        now: pd.Timestamp | None = None, skip_existing: bool = False) -> RunReport:
    """``skip_existing``: don't redo instruments whose latest bar already has a forecast with at least
    ``n_paths`` paths (weekends/holidays, or a second run the same night)."""
    now = now or pd.Timestamp.now(tz="UTC")
    run_date = now.tz_convert(NY).strftime("%Y-%m-%d")
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    report = RunReport(run_id, run_date)
    t0 = time.perf_counter()
    n_paths = n_paths or settings.model.sample_paths

    if update:
        for r in update_all(instruments, settings):
            if r.action == "error":
                log.warning("%s: bar update failed (%s); forecasting from cached bars if any", r.ticker, r.error)

    if predictor is None:
        from candlecast.kronos_loader import configure_torch_threads, predictor_for  # noqa: PLC0415

        configure_torch_threads(settings.model.torch_threads)
        predictor = predictor_for(settings.model)

    con = store.connect(settings.index_db)
    try:
        store.start_run(con, run_id, run_date, now.isoformat(), settings.model.name, len(instruments))
        for inst in instruments:
            if inst.source is None:
                report.failures.append(f"{inst.ticker}: no data source for exchange {inst.exchange}")
                continue
            try:
                row = forecast_one(inst, settings, predictor, n_paths, run_id, run_date, con, skip_existing)
                if row is None:
                    report.skipped.append(inst.ticker)
                    log.info("%s: latest bar already forecast, skipped", inst.ticker)
                    continue
                report.ok.append(inst.ticker)
                log.info("%s: as of %s, P(up) %.0f%%, %.1fs", inst.ticker, row["as_of_bar"],
                         100 * row["prob_up"], row["seconds"])
            except Exception as exc:  # one instrument's failure never stops the run
                log.exception("%s: forecast failed", inst.ticker)
                report.failures.append(f"{inst.ticker}: {type(exc).__name__}: {exc}"[:300])
        store.finish_run(con, run_id, pd.Timestamp.now(tz="UTC").isoformat(), len(report.ok), report.failures)
        try:  # grade earlier forecasts whose horizon the freshly updated bars now cover
            report.scored = scoring.score_pending(con, settings, now=now)
            log.info("scored %d finished forecasts", report.scored)
        except Exception:
            log.exception("scoring failed (forecasts are unaffected)")
        removed = store.prune(con, settings.root, settings.forecasts_dir, settings.keep_paths_days,
                              now.tz_convert(NY).normalize().tz_localize(None))
        if removed:
            log.info("pruned %d old path/chart files", removed)
    finally:
        con.close()
    report.seconds = time.perf_counter() - t0
    return report
