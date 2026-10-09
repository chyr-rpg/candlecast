import pandas as pd
import pytest

from candlecast import custom, forecaster
from tests.test_pipeline import fake_paths
from tests.test_timeframes import AAPL, FakeSource, daily_frame, tf_settings  # noqa: F401  (fixture)

NOW = pd.Timestamp("2026-10-09 18:00", tz="UTC")


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(forecaster.kv, "forecast_paths", fake_paths)


def test_run_one_daily_forecast_and_test_window(tf_settings, fake_model):  # noqa: F811
    con = custom.connect(tf_settings.index_db)
    spec = custom.Spec("1d", 16)
    src = FakeSource(daily_frame())
    row, test, status = custom.run_one(tf_settings, AAPL, spec, object(), 9, con, source=src, now=NOW)
    assert status == "new" and row["timeframe"] == "1d" and row["pred_len"] == 16
    assert row["history_bars"] == 128                                  # 8 x 16
    start = pd.Timestamp(row["horizon_start"]).tz_convert("America/New_York")
    assert start.strftime("%a %H:%M") == "Fri 00:00"                   # next session after Thu 10-08
    assert test["cutoff_bar"] < test["test_start"] <= test["test_end"] == row["as_of_bar"]
    per_bar = pd.read_parquet(tf_settings.root / test["summary_path"])
    assert len(per_bar) == 16 and per_bar["actual_close"].notna().all()
    assert 0 <= test["cover_bars"] <= 1 and test["abs_err_naive"] >= 0
    assert (tf_settings.root / row["history_path"]).exists()

    again = custom.run_one(tf_settings, AAPL, spec, object(), 9, con, source=src, now=NOW)
    assert again[2] == "cached"
    assert custom.run_one(tf_settings, AAPL, spec, object(), 9, con, rerun=True, source=src, now=NOW)[2] == "new"
    assert custom.latest(con, "AAPL", spec, tf_settings.model.name)[0]["pred_len"] == 16
    # another horizon for the same ticker/timeframe is a separate record
    row2, _, _ = custom.run_one(tf_settings, AAPL, custom.Spec("1d", 8), object(), 9, con, source=src, now=NOW)
    assert row2["history_bars"] == 128                                 # same history for any bar count
    # a different history is a separate forecast (and separate files), not a replaced one
    row3, _, st3 = custom.run_one(tf_settings, AAPL, custom.Spec("1d", 16, history=64), object(), 9, con,
                                  source=src, now=NOW)
    assert st3 == "new" and row3["history_bars"] == 64 and "_h64" in row3["summary_path"]
    assert con.execute("SELECT COUNT(*) FROM custom_forecasts").fetchone()[0] == 3
    con.close()


def test_connect_upgrades_old_primary_key_and_keeps_rows(tf_settings):  # noqa: F811
    import sqlite3

    old_schema = custom.SCHEMA.replace("PRIMARY KEY (ticker, timeframe, pred_len, history_bars, as_of_bar, model)",
                                       "PRIMARY KEY (ticker, timeframe, pred_len, as_of_bar, model)")
    con = sqlite3.connect(tf_settings.index_db)
    con.executescript(old_schema)
    cols = [r[1] for r in con.execute("PRAGMA table_info(custom_forecasts)")]
    con.execute(f"INSERT INTO custom_forecasts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                ["X" if c not in ("pred_len", "history_bars", "n_paths") else 16 for c in cols])
    con.commit(); con.close()
    con = custom.connect(tf_settings.index_db)
    pk = [r[1] for r in con.execute("PRAGMA table_info(custom_forecasts)") if r[5]]
    assert "history_bars" in pk and con.execute("SELECT COUNT(*) FROM custom_forecasts").fetchone()[0] == 1
    con.close()


def test_run_one_shortens_history_or_refuses(tf_settings, fake_model):  # noqa: F811
    con = custom.connect(tf_settings.index_db)
    few = FakeSource(daily_frame("2026-05-01"))                       # ~110 daily bars
    row, _, _ = custom.run_one(tf_settings, AAPL, custom.Spec("1d", 16), object(), 9, con, source=few, now=NOW)
    assert row["history_bars"] < 128 and "history shortened" in row["data_note"]
    tiny = FakeSource(daily_frame("2026-09-01"))                      # ~27 bars: not enough
    with pytest.raises(ValueError, match="needs at least"):
        custom.run_one(tf_settings, AAPL, custom.Spec("1w", 16), object(), 9, con, source=tiny, now=NOW)
    con.close()


def test_render_custom_draws_daily_chart(tf_settings, fake_model):  # noqa: F811
    con = custom.connect(tf_settings.index_db)
    row, test, _ = custom.run_one(tf_settings, AAPL, custom.Spec("1d", 16), object(), 9, con,
                                  source=FakeSource(daily_frame()), now=NOW)
    out = tf_settings.root / "chart.png"
    custom.render_custom(tf_settings, row, test, out, 80)
    assert out.exists() and out.stat().st_size > 10_000
    con.close()


def test_saved_forecast_not_reused_when_source_revised_old_bars(tf_settings, fake_model):  # noqa: F811
    con = custom.connect(tf_settings.index_db)
    spec = custom.Spec("1d", 16)
    frame = daily_frame()
    assert custom.run_one(tf_settings, AAPL, spec, object(), 9, con, source=FakeSource(frame), now=NOW)[2] == "new"
    assert custom.run_one(tf_settings, AAPL, spec, object(), 9, con, source=FakeSource(frame), now=NOW)[2] == "cached"
    # the source revises a recent bar (same latest timestamp, no new bar): the forecast must be recomputed
    from candlecast.bars import read_cache, write_cache
    d = tf_settings.bars_dir.parent / "1d"
    cached = read_cache(d, "AAPL")
    cached.loc[cached.index[-3], "close"] *= 1.01
    write_cache(d, "AAPL", cached)
    revised = custom.run_one(tf_settings, AAPL, spec, object(), 9, con, source=FakeSource(fail=True), now=NOW)
    assert revised[2] == "new"
    con.close()
