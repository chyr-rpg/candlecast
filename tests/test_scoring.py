import numpy as np
import pandas as pd
import pytest

from candlecast import scoring, store
from candlecast.bars import read_cache, write_cache
from candlecast.calendars import future_bars
from candlecast.pipeline import run
from tests.test_pipeline import AAPL, NOW, _bars, settings  # noqa: F401  (fixture)


def _append_actual(settings, ticker, timestamps, close):
    bars = read_cache(settings.bars_dir, ticker)
    extra = pd.DataFrame({"timestamp": timestamps, "open": close, "high": close * 1.002, "low": close * 0.998,
                          "close": close, "volume": 1000.0, "amount": 1000.0 * close})
    write_cache(settings.bars_dir, ticker, pd.concat([bars, extra], ignore_index=True))


def test_scores_only_after_horizon_and_only_once(settings):  # noqa: F811
    write_cache(settings.bars_dir, "AAPL", _bars("2026-10-05 23:00+00:00"))
    run(settings, [AAPL], n_paths=9, update=False, predictor=object(), now=NOW)
    con = store.connect(settings.index_db)
    f = store.latest(con, "AAPL")
    last = f["last_close"]
    horizon = future_bars("us_equity_extended", pd.Timestamp(f["as_of_bar"]), 32)

    _append_actual(settings, "AAPL", horizon[:10], last * 1.01)  # horizon only partly done
    assert scoring.score_pending(con, settings) == 0

    _append_actual(settings, "AAPL", horizon[10:], last * 1.01)
    assert scoring.score_pending(con, settings) == 1
    assert scoring.score_pending(con, settings) == 0  # idempotent

    s = dict(con.execute("SELECT * FROM scores").fetchone())
    assert s["n_actual_bars"] == 32 and s["actual_end"] == pytest.approx(last * 1.01)
    assert s["abs_err_naive"] == pytest.approx(0.01)
    assert s["abs_err_kronos"] == pytest.approx(0.01, abs=1e-6)  # fake median path is flat
    assert s["dir_hit"] == 0.0                                    # flat median vs up move
    assert s["brier"] == pytest.approx((4 / 9 - 1) ** 2)
    assert s["cover_end"] == 1.0 and s["crps_kronos"] is not None and s["crps_rw"] > 0
    assert 0.0 <= s["cover_bars"] <= 1.0

    table = scoring.report(con)
    assert table.loc["ALL", "n"] == 1 and "us_equity_extended" in table.index
    con.close()


def test_pruned_paths_leave_crps_null_but_score_the_rest(settings):  # noqa: F811
    write_cache(settings.bars_dir, "AAPL", _bars("2026-10-05 23:00+00:00"))
    run(settings, [AAPL], n_paths=9, update=False, predictor=object(), now=NOW)
    con = store.connect(settings.index_db)
    f = store.latest(con, "AAPL")
    (settings.root / f["paths_path"]).unlink()
    horizon = future_bars("us_equity_extended", pd.Timestamp(f["as_of_bar"]), 32)
    # thin stock: every other bar missing, including the horizon's final bar
    _append_actual(settings, "AAPL", horizon[0:31:2], f["last_close"] * 0.99)
    assert scoring.score_pending(con, settings) == 0  # cache hasn't reached horizon_end yet
    _append_actual(settings, "AAPL", pd.DatetimeIndex([horizon[-1] + pd.Timedelta(hours=10)]), f["last_close"])
    assert scoring.score_pending(con, settings) == 1  # a later bar proves the horizon is over
    s = dict(con.execute("SELECT * FROM scores").fetchone())
    assert s["crps_kronos"] is None and s["n_actual_bars"] == 16 and s["abs_err_naive"] == pytest.approx(0.01)
    con.close()


def test_crps_helpers_agree_and_reward_sharp_correct_forecasts():
    rng = np.random.default_rng(1)
    wide, sharp = rng.normal(100, 5, 20000), rng.normal(100, 1, 20000)
    assert scoring.crps_ensemble(wide, 100.5) == pytest.approx(scoring.crps_normal(100, 5, 100.5), rel=0.02)
    assert scoring.crps_ensemble(sharp, 100.5) < scoring.crps_ensemble(wide, 100.5)
    assert scoring.crps_ensemble(np.array([3.0]), 1.0) == pytest.approx(2.0)  # point forecast = abs error


def test_rank_ic_by_group_is_cross_sectional_spearman():
    df = pd.DataFrame({
        "date": ["d1"] * 5 + ["d2"] * 5 + ["d3"] * 3,
        "signal": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
        "actual_ret": [10, 20, 30, 40, 50, 5, 4, 3, 2, 1, 1, 2, 3],   # d1 perfect, d2 reversed, d3 too small
    })
    ic = scoring.rank_ic_by_group(df, "date", "signal")
    assert ic.to_dict() == pytest.approx({"d1": 1.0, "d2": -1.0})
    assert "mean RankIC +0.000" in scoring.describe_rank_ic(ic) and "2 dates" in scoring.describe_rank_ic(ic)
    assert scoring.describe_rank_ic(pd.Series(dtype=float)).startswith("no dates")


def test_rank_ic_report_ranks_instruments_within_each_run(settings):  # noqa: F811
    from tests.test_pipeline import Instrument

    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    insts = [Instrument(t, "", "alpaca_stock", t, "us_equity_extended") for t in tickers]
    for t in tickers:
        write_cache(settings.bars_dir, t, _bars("2026-10-05 23:00+00:00"))
    run(settings, insts, n_paths=9, update=False, predictor=object(), now=NOW)
    con = store.connect(settings.index_db)
    # actual 2-day returns: AAA best ... EEE worst
    for rank, t in enumerate(tickers):
        f = store.latest(con, t)
        horizon = future_bars("us_equity_extended", pd.Timestamp(f["as_of_bar"]), 32)
        _append_actual(settings, t, horizon, f["last_close"] * (1.05 - 0.02 * rank))
    assert scoring.score_pending(con, settings) == 5
    # fake forecasts: every path set is symmetric around the last close => all expected returns equal
    ic_mean, ic_p50 = scoring.rank_ic_report(con, settings.root)
    assert ic_p50.empty                      # no variation in the signal => that run is skipped, not scored 0
    # give the median signal a known order (the reverse of the actual order) and check RankIC = -1
    for rank, t in enumerate(tickers):
        con.execute("UPDATE forecasts SET end_close_p50 = last_close * ? WHERE ticker = ?", (1 + 0.01 * rank, t))
    _, ic_p50 = scoring.rank_ic_report(con, settings.root)
    con.close()
    assert len(ic_p50) == 1 and ic_p50.iloc[0] == pytest.approx(-1.0)
