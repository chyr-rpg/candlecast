"""Research data: daily bars for the US-equity instruments, for the daily-bar accuracy tests.

Alpaca daily stock bars (regular session, SIP feed, split-adjusted - the same feed/adjustment as the
hourly cache), from --start to the last COMPLETED New York session, into data/bars/1d/<TICKER>.parquet
(normalized columns, ``timestamp`` = UTC time of the bar's New York midnight).

Kronos was pretrained on data up to June 2024: test cutoffs must be from 2024-07-01 on
(accuracy_backtest.py --min-cutoff). Older bars are only used as history.

Usage: .venv\\Scripts\\python benchmarks\\fetch_daily.py [--start 2016-01-01]
"""

from __future__ import annotations

import argparse

import pandas as pd

from candlecast.bars import normalize, write_cache
from candlecast.calendars import NY
from candlecast.settings import load_settings
from candlecast.sources import AlpacaStockSource
from candlecast.symbols import load_instruments, read_symbol_map, read_watchlist, resolve


def fetch_daily(source: AlpacaStockSource, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed  # noqa: PLC0415
    from alpaca.data.requests import StockBarsRequest  # noqa: PLC0415
    from alpaca.data.timeframe import TimeFrame  # noqa: PLC0415

    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end,
                               feed=DataFeed.SIP, adjustment=Adjustment.SPLIT)
    df = source.client.get_stock_bars(request).df
    return df.reset_index() if df is not None and not df.empty else pd.DataFrame()


def instruments_for(settings, tickers_file: str | None):
    """The forecast universe, or the instruments listed in ``tickers_file`` (resolved the same way)."""
    if not tickers_file:
        return load_instruments(settings)
    return resolve(read_watchlist(tickers_file), read_symbol_map(settings.data.symbol_map_csv))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--tickers-file", default=None,
                    help="CSV (ticker[,exchange]) to use instead of the forecast universe")
    args = ap.parse_args()
    settings = load_settings()
    out_dir = settings.bars_dir.parent / "1d"
    source = AlpacaStockSource()
    today_ny = pd.Timestamp.now(tz=NY).normalize()
    end = (today_ny - pd.Timedelta(minutes=16)).tz_convert("UTC")  # the free plan needs bars > 15 min old
    for inst in instruments_for(settings, args.tickers_file):
        if inst.source != "alpaca_stock":
            continue
        try:
            raw = fetch_daily(source, inst.source_symbol, pd.Timestamp(args.start, tz="UTC"), end)
        except Exception as exc:  # one bad symbol never stops the download
            print(f"{inst.ticker}: FAILED {type(exc).__name__}: {exc}"[:200])
            continue
        bars = normalize(raw)
        # keep only completed sessions: a bar dated today (New York) may still be forming
        bars = bars[bars["timestamp"].dt.tz_convert(NY).dt.normalize() < today_ny].reset_index(drop=True)
        if bars.empty:
            print(f"{inst.ticker}: no daily bars returned")
            continue
        write_cache(out_dir, inst.ticker, bars)
        first, last = bars["timestamp"].iloc[[0, -1]].dt.tz_convert(NY).dt.date
        print(f"{inst.ticker}: {len(bars)} daily bars {first} -> {last}")


if __name__ == "__main__":
    main()
