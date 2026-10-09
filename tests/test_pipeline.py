from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from candlecast import forecaster, store
from candlecast.bars import write_cache
from candlecast.cli import synthetic_bars
from candlecast.pipeline import run
from candlecast.settings import load_settings
from candlecast.symbols import Instrument

NOW = pd.Timestamp("2026-10-06 01:30", tz="UTC")  # Mon 21:30 New York


def fake_paths(predictor, df, x_ts, y_ts, pred_len, n_paths, **kw):
    """Path i drifts linearly by (i - mid) * 0.1% per bar from the last close."""
    last = df.iloc[-1].to_numpy()
    drift = (np.arange(n_paths) - (n_paths - 1) / 2) * 0.001
    steps = np.arange(1, pred_len + 1)
    scale = 1 + drift[:, None] * steps[None, :]
    out = np.repeat(last[None, None, :], n_paths, axis=0).repeat(pred_len, axis=1)
    out[:, :, :4] *= scale[:, :, None]
    out[:, :, 1] *= 1.002  # high
    out[:, :, 2] *= 0.998  # low
    return out


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(forecaster.kv, "forecast_paths", fake_paths)
    s = load_settings()
    return replace(s, root=tmp_path, bars_dir=tmp_path / "bars", forecasts_dir=tmp_path / "forecasts",
                   index_db=tmp_path / "index.db", run_log_dir=tmp_path / "logs")


def _bars(last_utc: str, n=600):
    b = synthetic_bars(n)
    b["timestamp"] = pd.date_range(end=pd.Timestamp(last_utc), periods=n, freq="h")
    return b


AAPL = Instrument("AAPL", "", "alpaca_stock", "AAPL", "us_equity_extended")
BTC = Instrument("BTCUSD", "", "alpaca_crypto", "BTC/USD", "crypto")
MISSING = Instrument("NOPE", "", "alpaca_stock", "NOPE", "us_equity_extended")


def test_run_writes_files_index_and_isolates_failures(settings):
    write_cache(settings.bars_dir, "AAPL", _bars("2026-10-05 23:00+00:00"))
    write_cache(settings.bars_dir, "BTCUSD", _bars("2026-10-06 00:00+00:00"))
    report = run(settings, [AAPL, MISSING, BTC], n_paths=9, update=False, predictor=object(), now=NOW)
    assert report.ok == ["AAPL", "BTCUSD"] and len(report.failures) == 1 and report.failures[0].startswith("NOPE")
    assert report.run_date == "2026-10-05"

    folder = settings.forecasts_dir / "2026-10-05"
    for name in ("AAPL_summary.parquet", "AAPL_paths.parquet", "AAPL.png", "BTCUSD.png"):
        assert (folder / name).exists(), name

    con = store.connect(settings.index_db)
    row = store.latest(con, "aapl")
    assert row["pred_len"] == 32 and row["n_paths"] == 9 and row["history_bars"] == settings.model.history_bars == 240
    assert pd.Timestamp(row["horizon_start"]).tz_convert("America/New_York").strftime("%a %H:%M") == "Tue 04:00"
    assert row["end_close_p10"] < row["end_close_p50"] < row["end_close_p90"]
    assert row["prob_up"] == pytest.approx(4 / 9)  # paths 5..8 drift up, path 4 is flat
    assert store.latest(con, "BTCUSD")["pred_len"] == 48
    run_row = dict(con.execute("SELECT * FROM runs").fetchone())
    assert (run_row["n_ok"], run_row["n_failed"]) == (2, 1)
    con.close()

    paths = pd.read_parquet(folder / "AAPL_paths.parquet")
    assert len(paths) == 9 * 32 and set(paths["path"]) == set(range(9))


def test_rerun_same_bar_replaces_row_and_prune_keeps_summaries(settings):
    write_cache(settings.bars_dir, "AAPL", _bars("2026-10-05 23:00+00:00"))
    run(settings, [AAPL], n_paths=5, update=False, predictor=object(), now=NOW)
    run(settings, [AAPL], n_paths=7, update=False, predictor=object(), now=NOW + pd.Timedelta(minutes=5))
    con = store.connect(settings.index_db)
    assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
    assert store.latest(con, "AAPL")["n_paths"] == 7

    later = NOW + pd.Timedelta(days=settings.keep_paths_days + 2)
    removed = store.prune(con, settings.root, settings.forecasts_dir, settings.keep_paths_days,
                          later.tz_convert("America/New_York").normalize().tz_localize(None))
    folder = settings.forecasts_dir / "2026-10-05"
    assert removed == 2
    assert (folder / "AAPL_summary.parquet").exists()
    assert not (folder / "AAPL_paths.parquet").exists() and not (folder / "AAPL.png").exists()
    row = store.latest(con, "AAPL")
    assert row["paths_path"] is None and row["chart_path"] is None and row["summary_path"]
    con.close()


def test_skip_existing_only_redoes_instruments_with_new_bars(settings):
    write_cache(settings.bars_dir, "AAPL", _bars("2026-10-05 23:00+00:00"))
    write_cache(settings.bars_dir, "BTCUSD", _bars("2026-10-06 00:00+00:00"))
    run(settings, [AAPL, BTC], n_paths=5, update=False, predictor=object(), now=NOW)
    write_cache(settings.bars_dir, "BTCUSD", _bars("2026-10-06 05:00+00:00"))  # only BTC got new bars
    later = NOW + pd.Timedelta(hours=6)
    report = run(settings, [AAPL, BTC], n_paths=5, update=False, predictor=object(), now=later, skip_existing=True)
    assert report.ok == ["BTCUSD"] and report.skipped == ["AAPL"]
    # asking for more paths than stored redoes it
    report = run(settings, [AAPL], n_paths=8, update=False, predictor=object(), now=later, skip_existing=True)
    assert report.ok == ["AAPL"]


def test_horizon_and_bar_summary_math():
    paths = np.zeros((4, 3, 6))
    paths[:, :, 3] = [[1, 2, 10], [1, 2, 20], [1, 2, 30], [1, 2, 40]]   # close
    paths[:, :, 1] = paths[:, :, 3] + 1                                   # high
    paths[:, :, 2] = paths[:, :, 3] - 1                                   # low
    res = forecaster.ForecastResult("X", "crypto", pd.Timestamp("2026-01-01", tz="UTC"), 25.0,
                                    pd.date_range("2026-01-01 01:00", periods=3, freq="h", tz="UTC"), paths, 10)
    h = forecaster.horizon_summary(res)
    assert h["prob_up"] == 0.5
    assert h["end_close_p50"] == 25.0
    assert h["max_high_p50"] == 26.0 and h["min_low_p50"] == 0.0
    b = forecaster.bar_summary(res)
    assert b["close_p50"].tolist() == [1.0, 2.0, 25.0]
    assert b["prob_above_last"].tolist() == [0.0, 0.0, 0.5]


def test_seed_is_stable_per_ticker_and_bar():
    t = pd.Timestamp("2026-10-05 23:00", tz="UTC")
    assert forecaster.seed_for("AAPL", t) == forecaster.seed_for("AAPL", t)
    assert forecaster.seed_for("AAPL", t) != forecaster.seed_for("MSFT", t)


def test_test_forecast_covers_last_horizon_without_seeing_it(settings, monkeypatch):
    seen_inputs = []

    def recording(predictor, df, x_ts, y_ts, pred_len, n_paths, **kw):
        seen_inputs.append((pd.Timestamp(x_ts.iloc[-1]), pd.Timestamp(y_ts.iloc[0]), pd.Timestamp(y_ts.iloc[-1])))
        return fake_paths(predictor, df, x_ts, y_ts, pred_len, n_paths, **kw)

    monkeypatch.setattr(forecaster.kv, "forecast_paths", recording)
    bars = _bars("2026-10-05 23:00+00:00")
    write_cache(settings.bars_dir, "AAPL", bars)
    report = run(settings, [AAPL], n_paths=9, update=False, predictor=object(), now=NOW)
    assert report.ok == ["AAPL"]

    con = store.connect(settings.index_db)
    live = store.latest(con, "AAPL")
    t = store.latest_test(con, "AAPL", live["as_of_bar"], live["model"])
    con.close()
    ts = bars["timestamp"].dt.tz_localize("UTC") if bars["timestamp"].dt.tz is None else bars["timestamp"]
    assert pd.Timestamp(t["cutoff_bar"]) == ts.iloc[-33]          # last bar the test forecast could see
    assert pd.Timestamp(t["test_start"]) == ts.iloc[-32]
    assert pd.Timestamp(t["test_end"]) == ts.iloc[-1] == pd.Timestamp(live["as_of_bar"])
    assert t["pred_len"] == 32 and t["cutoff_close"] == pytest.approx(bars["close"].iloc[-33])

    # no leakage: the test call's model input ends at the cutoff, and it forecasts exactly the actual bars
    to_ny = lambda x: x.tz_convert("America/New_York").tz_localize(None)  # noqa: E731
    test_call = [c for c in seen_inputs if c[0] == to_ny(ts.iloc[-33])]
    assert len(test_call) == 1 and test_call[0][1:] == (to_ny(ts.iloc[-32]), to_ny(ts.iloc[-1]))

    # fake paths: path 4 of 9 is flat => median = cutoff close, so the error equals the no-change error
    per_bar = pd.read_parquet(settings.forecasts_dir / "2026-10-05" / "AAPL_test.parquet")
    assert len(per_bar) == 32
    assert per_bar["actual_close"].tolist() == pytest.approx(bars["close"].tail(32).tolist())
    actual_end, c0 = bars["close"].iloc[-1], bars["close"].iloc[-33]
    assert t["actual_end"] == pytest.approx(actual_end)
    assert t["abs_err_kronos"] == pytest.approx(abs(c0 - actual_end) / c0) == pytest.approx(t["abs_err_naive"])
    assert 0 <= t["cover_bars"] <= 1 and t["dir_hit"] in (0.0, 1.0)
    # the test forecast is kept out of the forward-test record
    con = store.connect(settings.index_db)
    from candlecast import scoring
    scoring.ensure_schema(con)
    assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
    con.close()


def test_test_forecast_can_be_disabled_and_failure_never_blocks_live(settings, monkeypatch):
    write_cache(settings.bars_dir, "AAPL", _bars("2026-10-05 23:00+00:00"))
    run(settings, [AAPL], n_paths=5, update=False, predictor=object(), now=NOW)
    folder = settings.forecasts_dir / "2026-10-05"
    assert (folder / "AAPL_test.parquet").exists()

    off = replace(settings, test_forecast=False, index_db=settings.root / "off.db",
                  forecasts_dir=settings.root / "off")
    run(off, [AAPL], n_paths=5, update=False, predictor=object(), now=NOW)
    con = store.connect(off.index_db)
    assert con.execute("SELECT COUNT(*) FROM test_forecasts").fetchone()[0] == 0
    con.close()
    assert not (off.forecasts_dir / "2026-10-05" / "AAPL_test.parquet").exists()

    import candlecast.pipeline as pipeline_mod

    def boom(*a, **k):
        raise RuntimeError("test window exploded")

    monkeypatch.setattr(pipeline_mod, "run_test_forecast", boom)
    broken = replace(settings, index_db=settings.root / "broken.db", forecasts_dir=settings.root / "broken")
    report = run(broken, [AAPL], n_paths=5, update=False, predictor=object(), now=NOW)
    assert report.ok == ["AAPL"] and not report.failures
    assert (broken.forecasts_dir / "2026-10-05" / "AAPL.png").exists()
