from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from candlecast.bars import read_cache
from candlecast.data_update import coverage, update_all, update_instrument
from candlecast.settings import load_settings
from candlecast.symbols import Instrument, load_instruments, read_symbol_map, read_watchlist, resolve, universe

NOW = pd.Timestamp("2026-01-10 12:00", tz="UTC")


class FakeSource:
    """Hourly bars at a constant level; ``scale`` changes to simulate a split re-adjustment."""

    min_delay = pd.Timedelta(minutes=16)

    def __init__(self, level=100.0, fail=False):
        self.level, self.fail, self.calls = level, fail, []

    def fetch(self, symbol, start, end):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        ts = pd.date_range(pd.Timestamp(start).ceil("h"), pd.Timestamp(end), freq="h")
        n = len(ts)
        return pd.DataFrame({"timestamp": ts, "open": [self.level] * n, "high": [self.level + 1] * n,
                             "low": [self.level - 1] * n, "close": [self.level] * n, "volume": [5.0] * n})


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    return replace(s, bars_dir=tmp_path / "bars", root=tmp_path,
                   data=replace(s.data, history_days=30, min_bars=100))


INST = Instrument("AAA", "NASDAQ", "alpaca_stock", "AAA", "us_equity_extended")


def test_first_run_full_then_incremental_appends(settings):
    src = FakeSource()
    first = update_instrument(INST, src, settings.data, settings.bars_dir, now=NOW)
    assert first.action == "full"
    n_first = len(read_cache(settings.bars_dir, "AAA"))
    later = NOW + pd.Timedelta(hours=5)
    second = update_instrument(INST, src, settings.data, settings.bars_dir, now=later)
    assert second.action == "incremental" and second.new_bars == 5
    cached = read_cache(settings.bars_dir, "AAA")
    assert len(cached) == n_first + 5 and cached["timestamp"].is_unique
    # incremental request starts overlap_hours before the cached tip, not from scratch
    assert src.calls[-1][0] >= NOW - pd.Timedelta(hours=settings.data.overlap_hours + 2)
    # the last stored bar has fully closed before the source's delay cutoff
    assert cached["timestamp"].max() + pd.Timedelta(hours=1) <= later - src.min_delay


def test_revision_triggers_full_redownload(settings):
    update_instrument(INST, FakeSource(100.0), settings.data, settings.bars_dir, now=NOW)
    res = update_instrument(INST, FakeSource(50.0), settings.data, settings.bars_dir, now=NOW + pd.Timedelta(hours=2))
    assert res.action == "full_after_revision"
    assert (read_cache(settings.bars_dir, "AAA")["close"] == 50.0).all()


def test_update_all_isolates_failures_and_skips_unmapped(settings):
    insts = [INST, Instrument("BAD", "NASDAQ", "alpaca_stock", "BAD", "us_equity_extended"),
             Instrument("ZZZ", "LSE", None, None, None)]
    good, bad = FakeSource(), FakeSource(fail=True)

    class Router:
        def __init__(self):
            self.sources = {"AAA": good, "BAD": bad}
            self.min_delay = good.min_delay

        def fetch(self, symbol, start, end):
            return self.sources[symbol].fetch(symbol, start, end)

    results = update_all(insts, settings, sources={"alpaca_stock": Router})
    assert [r.action for r in results] == ["full", "error", "skipped"]
    assert "boom" in results[1].error
    report = coverage(insts, settings, results, now=NOW)
    assert report.set_index("ticker")["status"].to_dict() == {"AAA": "ok", "BAD": "missing", "ZZZ": "unmapped"}


def test_coverage_short_and_stale(settings):
    s = replace(settings, data=replace(settings.data, min_bars=10_000))
    update_instrument(INST, FakeSource(), s.data, s.bars_dir, now=NOW)
    assert coverage([INST], s, now=NOW)["status"].iloc[0] == "short"
    assert coverage([INST], settings, now=NOW + pd.Timedelta(days=10))["status"].iloc[0] == "stale"


def test_resolve_watchlist_with_overrides(tmp_path):
    (tmp_path / "w.csv").write_text("ticker,exchange\nAAPL,NASDAQ\nBRK.B,NYSE\nEURUSD,OANDA\nBTCUSD,COINBASE\nFOO,LSE\n")
    (tmp_path / "m.csv").write_text(
        "ticker,source,source_symbol,asset_class,note\n"
        "EURUSD,yfinance,EURUSD=X,fx,\nBTCUSD,alpaca_crypto,BTC/USD,crypto,\n")
    got = {i.ticker: i for i in resolve(read_watchlist(tmp_path / "w.csv"), read_symbol_map(tmp_path / "m.csv"))}
    assert (got["AAPL"].source, got["AAPL"].asset_class) == ("alpaca_stock", "us_equity_extended")
    assert got["BRK.B"].source_symbol == "BRK.B"
    assert (got["EURUSD"].source, got["EURUSD"].source_symbol) == ("yfinance", "EURUSD=X")
    assert got["BTCUSD"].source_symbol == "BTC/USD"
    assert got["FOO"].source is None


@pytest.mark.parametrize("content", [None, "", "# only comments\n", "ticker,exchange\n", "# note\nticker\n"])
def test_missing_or_empty_watchlist_falls_back_to_default(tmp_path, content):
    path = tmp_path / "watchlist.csv"
    if content is not None:
        path.write_text(content)
    s = replace(load_settings(), watchlist_csv=path)
    df, label = universe(s)
    assert label == "default"
    assert tuple(df["ticker"]) == s.default_tickers


def test_own_watchlist_wins_and_exchange_column_is_optional(tmp_path):
    path = tmp_path / "watchlist.csv"
    path.write_text("ticker\nnflx\nBTCUSD\n")
    s = replace(load_settings(), watchlist_csv=path)
    df, label = universe(s)
    assert label == str(path) and df["ticker"].tolist() == ["NFLX", "BTCUSD"]
    got = {i.ticker: i for i in load_instruments(s)}
    assert (got["NFLX"].source, got["NFLX"].asset_class) == ("alpaca_stock", "us_equity_extended")
    assert got["BTCUSD"].source == "alpaca_crypto"


def test_default_list_is_fully_mapped():
    s = replace(load_settings(), watchlist_csv=Path("does/not/exist.csv"))
    insts = load_instruments(s)
    assert [i.ticker for i in insts if i.source is None] == []
    assert {i.ticker: i.source for i in insts}["XAUUSD"] == "yfinance"
