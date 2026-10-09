import pandas as pd
import pytest

from candlecast.bars import COLUMNS, merge_incremental, normalize


def _raw(times, closes, **extra):
    n = len(times)
    df = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True), "open": closes, "high": [c + 1 for c in closes],
                       "low": [c - 1 for c in closes], "close": closes, "volume": [10.0] * n})
    for k, v in extra.items():
        df[k] = v
    return df


def test_normalize_floors_to_hour_and_merges_collisions():
    raw = _raw(["2026-01-05 14:00", "2026-01-05 14:30", "2026-01-05 15:00"], [100.0, 102.0, 103.0])
    raw.loc[1, "high"] = 110.0
    out = normalize(raw)
    assert list(out.columns) == COLUMNS
    assert len(out) == 2
    first = out.iloc[0]
    assert (first["open"], first["high"], first["close"], first["volume"]) == (100.0, 110.0, 102.0, 20.0)
    assert str(out["timestamp"].dt.tz) == "UTC"


def test_normalize_amount_uses_vwap_when_present_else_mean_ohlc():
    with_vwap = normalize(_raw(["2026-01-05 14:00"], [100.0], vwap=[99.0]))
    assert with_vwap["amount"].iloc[0] == pytest.approx(990.0)
    without = normalize(_raw(["2026-01-05 14:00"], [100.0]))
    assert without["amount"].iloc[0] == pytest.approx(1000.0)  # mean(101, 99, 100, 100) * 10


def test_normalize_drops_unclosed_bar_and_bad_prices():
    raw = _raw(["2026-01-05 13:00", "2026-01-05 14:00", "2026-01-05 15:00"], [100.0, -1.0, 101.0])
    out = normalize(raw, fetch_end=pd.Timestamp("2026-01-05 15:40", tz="UTC"))
    assert out["timestamp"].dt.hour.tolist() == [13]  # 14:00 has a bad price, 15:00 hasn't closed


def test_normalize_accepts_yfinance_style_datetime_index():
    raw = _raw(["2026-01-05 09:00-05:00"], [100.0]).set_index("timestamp")
    raw.index.name = "Datetime"
    out = normalize(raw.rename(columns=str.title))
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-01-05 14:00", tz="UTC")


def test_merge_appends_and_fresh_wins_within_tolerance():
    cached = normalize(_raw(["2026-01-05 13:00", "2026-01-05 14:00"], [100.0, 101.0]))
    fresh = normalize(_raw(["2026-01-05 14:00", "2026-01-05 15:00"], [101.01, 102.0]))
    res = merge_incremental(cached, fresh, rtol=0.005)
    assert not res.revised
    assert res.new_bars == 1
    assert res.frame["close"].tolist() == [100.0, 101.01, 102.0]


def test_merge_flags_revision_like_a_split():
    cached = normalize(_raw(["2026-01-05 13:00", "2026-01-05 14:00"], [100.0, 101.0]))
    fresh = normalize(_raw(["2026-01-05 14:00", "2026-01-05 15:00"], [50.5, 51.0]))
    res = merge_incremental(cached, fresh, rtol=0.005)
    assert res.revised
    assert res.frame is cached
