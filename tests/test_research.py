import numpy as np
import pandas as pd
import pytest

from candlecast import research


def _hourly(day: str, hours=range(4, 20)):
    ts = [pd.Timestamp(f"{day} {h:02d}:00", tz="America/New_York").tz_convert("UTC") for h in hours]
    close = np.arange(1, len(ts) + 1, dtype=float)
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1.0, "amount": close})


def test_rth_and_four_hour_bars():
    bars = _hourly("2026-10-01")
    rth = research.rth_hourly(bars)
    assert rth["timestamp"].dt.tz_convert("America/New_York").dt.hour.tolist() == list(range(9, 16))

    four = research.four_hour(bars)
    assert four["timestamp"].dt.tz_convert("America/New_York").dt.hour.tolist() == [4, 8, 12, 16]
    first = four.iloc[0]  # hourly closes 1..4
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (1, 4.5, 0.5, 4, 4)

    # an unfinished last bucket (only 16:00-17:00 present) is dropped
    partial = research.four_hour(_hourly("2026-10-01", hours=range(4, 17)))
    assert partial["timestamp"].dt.tz_convert("America/New_York").dt.hour.tolist() == [4, 8, 12]


def test_realized_vol_and_simple_baselines():
    closes = np.array([100.0, 101.0, 100.0])
    r = np.log([1.01, 100 / 101])
    assert research.realized_vol(closes) == pytest.approx(np.sqrt((r ** 2).sum()))
    flat_ret = 100 * np.exp(np.cumsum(np.tile([0.01, -0.01], 200)))  # constant |return| = 1%
    assert research.ewma_vol(flat_ret, 4) == pytest.approx(0.01 * 2, rel=1e-3)
    assert research.hist_vol(flat_ret, 4) == pytest.approx(0.01 * 2, rel=1e-3)


def test_garch_fit_recovers_parameters_on_simulated_data():
    rng = np.random.default_rng(7)
    omega, a, b = 1e-6, 0.08, 0.90
    n, s2, r = 6000, omega / (1 - a - b), []
    for _ in range(n):
        x = np.sqrt(s2) * rng.standard_normal()
        r.append(x)
        s2 = omega + a * x * x + b * s2
    _, a_hat, b_hat, _ = research.garch_fit(np.array(r))
    assert a_hat == pytest.approx(a, abs=0.03) and b_hat == pytest.approx(b, abs=0.04)
    # forecast over many steps approaches the long-run variance
    close = 100 * np.exp(np.concatenate([[0], np.cumsum(r)]))
    long_run = np.sqrt(omega / (1 - a - b) * 500)
    assert research.garch_vol(close, 500) == pytest.approx(long_run, rel=0.25)


def test_qlike_is_zero_when_exact_and_penalizes_underprediction_more():
    assert research.qlike(0.02, 0.02) == pytest.approx(0.0)
    assert research.qlike(0.02, 0.01) > research.qlike(0.01, 0.02) > 0
