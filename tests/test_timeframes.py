import numpy as np
import pandas as pd
import pytest

from candlecast import timeframes
from candlecast.bars import read_cache, write_cache
from candlecast.symbols import Instrument

NY = "America/New_York"
AAPL = Instrument("AAPL", "", "alpaca_stock", "AAPL", "us_equity_extended")


def test_history_rule():
    # history no longer depends on the bar count (default 128) - only the 512-bar context caps it
    assert timeframes.history_bars(16) == 128 and timeframes.bars_required(16) == 144
    assert timeframes.history_bars(4) == timeframes.history_bars(64) == 128
    assert timeframes.history_bars(16, history=300) == 300
    assert timeframes.history_bars(16, history=10) == 64        # floor
    assert timeframes.history_bars(16, history=999) == 448      # slider maximum
    assert timeframes.history_bars(128, history=448) == 384     # 448 + 128 > 512 -> capped
    with pytest.raises(ValueError):
        timeframes.validate("2h", 16)
    with pytest.raises(ValueError):
        timeframes.validate("1d", 200)


def _hourly(day: str, hours=range(4, 20)):
    ts = [pd.Timestamp(f"{day} {h:02d}:00", tz=NY).tz_convert("UTC") for h in hours]
    c = np.arange(1, len(ts) + 1, dtype=float)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "volume": 1.0, "amount": c})


def test_aggregate_4h_buckets_and_drops_unfinished():
    four = timeframes.aggregate_4h(_hourly("2026-10-01"))
    assert four["timestamp"].dt.tz_convert(NY).dt.hour.tolist() == [4, 8, 12, 16]
    assert four.iloc[0][["open", "high", "low", "close", "volume"]].tolist() == [1, 4.5, 0.5, 4, 4]
    partial = timeframes.aggregate_4h(_hourly("2026-10-01", range(4, 18)))
    assert partial["timestamp"].dt.tz_convert(NY).dt.hour.tolist() == [4, 8, 12]


def ny(s):
    return pd.Timestamp(s, tz=NY).tz_convert("UTC")


def test_future_bars_per_timeframe():
    # 4h: after Friday's 16:00 bucket the next buckets are Monday 04:00 and 08:00 (equities)
    f = timeframes.future_bars("us_equity_extended", "4h", ny("2026-10-02 16:00"), 2)
    assert [t.tz_convert(NY).strftime("%a %H:%M") for t in f] == ["Mon 04:00", "Mon 08:00"]
    # 4h crypto runs around the clock
    f = timeframes.future_bars("crypto", "4h", ny("2026-10-02 20:00"), 2)
    assert [t.tz_convert(NY).strftime("%a %H:%M") for t in f] == ["Sat 00:00", "Sat 04:00"]
    # 1d equities: Thu 2026-07-02 -> Fri 07-03 is the observed Independence Day holiday -> Mon 07-06
    f = timeframes.future_bars("us_equity_extended", "1d", ny("2026-07-02 00:00"), 2)
    assert [t.tz_convert(NY).strftime("%Y-%m-%d %H:%M") for t in f] == ["2026-07-06 00:00", "2026-07-07 00:00"]
    f = timeframes.future_bars("fx", "1d", ny("2026-10-02 00:00"), 2)         # Fri -> Mon, Tue
    assert [t.tz_convert(NY).strftime("%a") for t in f] == ["Mon", "Tue"]
    f = timeframes.future_bars("crypto", "1d", ny("2026-10-02 00:00"), 2)     # every day
    assert [t.tz_convert(NY).strftime("%a") for t in f] == ["Sat", "Sun"]
    f = timeframes.future_bars("us_equity_extended", "1w", ny("2026-09-28 00:00"), 2)
    assert [t.tz_convert(NY).strftime("%Y-%m-%d") for t in f] == ["2026-10-05", "2026-10-12"]


def daily_frame(start="2025-01-01", end="2026-10-08"):
    days = pd.bdate_range(start, end)
    ts = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(NY) for d in days]).tz_convert("UTC")
    c = 100 + np.cumsum(np.sin(np.arange(len(ts)) / 7.0))
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c,
                         "volume": 1e6, "vwap": c})


class FakeSource:
    name = "fake"
    min_delay = pd.Timedelta(minutes=16)

    def __init__(self, frame=None, fail=False):
        self.frame, self.fail, self.calls = frame, fail, []

    def fetch(self, symbol, start, end, timeframe="1h"):
        self.calls.append((pd.Timestamp(start), timeframe))
        if self.fail:
            raise ConnectionError("offline")
        f = self.frame
        return f[(f["timestamp"] >= pd.Timestamp(start)) & (f["timestamp"] <= pd.Timestamp(end))].copy()


@pytest.fixture
def tf_settings(tmp_path):
    from dataclasses import replace

    from candlecast.settings import load_settings
    s = load_settings()
    return replace(s, root=tmp_path, bars_dir=tmp_path / "bars" / "1h", forecasts_dir=tmp_path / "forecasts",
                   index_db=tmp_path / "index.db")


def test_load_bars_downloads_then_refreshes_then_falls_back(tf_settings):
    now = pd.Timestamp("2026-10-08 18:00", tz="UTC")   # Thu 14:00 New York: today's daily bar not finished
    src = FakeSource(daily_frame())
    bars, note = timeframes.load_bars(tf_settings, AAPL, "1d", 144, source=src, now=now)
    assert "downloaded online" in note and len(bars) >= 144
    assert bars["timestamp"].max().tz_convert(NY).strftime("%Y-%m-%d") == "2026-10-07"   # 10-08 unfinished
    assert read_cache(tf_settings.bars_dir.parent / "1d", "AAPL") is not None             # saved locally

    later = pd.Timestamp("2026-10-09 18:00", tz="UTC")
    bars2, note2 = timeframes.load_bars(tf_settings, AAPL, "1d", 144, source=src, now=later)
    assert "refreshed online (+1 bars)" in note2 and len(bars2) == len(bars) + 1

    offline = FakeSource(fail=True)
    bars3, note3 = timeframes.load_bars(tf_settings, AAPL, "1d", 144, source=offline, now=later)
    assert "download failed" in note3 and len(bars3) == len(bars2)

    with pytest.raises(ConnectionError):  # no local database and no connection
        timeframes.load_bars(tf_settings, AAPL, "1w", 50, source=offline, now=later)


def test_load_bars_redownloads_when_local_history_too_short(tf_settings):
    short = daily_frame("2026-08-01")
    write_cache(tf_settings.bars_dir.parent / "1d", "AAPL", short.drop(columns="vwap").assign(amount=1.0))
    src = FakeSource(daily_frame())
    bars, note = timeframes.load_bars(tf_settings, AAPL, "1d", 300, source=src,
                                      now=pd.Timestamp("2026-10-09 18:00", tz="UTC"))
    assert "downloaded online" in note and len(bars) >= 300
