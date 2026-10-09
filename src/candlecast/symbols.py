"""Resolve watchlist tickers to a data source and an asset class."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

US_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"}
ASSET_CLASSES = ("us_equity_extended", "crypto", "fx", "commodity")
SOURCES = ("alpaca_stock", "alpaca_crypto", "yfinance")


@dataclass(frozen=True)
class Instrument:
    ticker: str          # the ticker as typed, used for file names and output
    exchange: str        # optional exchange from the watchlist
    source: str | None   # None => unmapped
    source_symbol: str | None
    asset_class: str | None


def read_watchlist(path: Path) -> pd.DataFrame:
    """Columns ``ticker`` and optional ``exchange``; a missing file reads as empty."""
    empty = pd.DataFrame(columns=["ticker", "exchange"])
    if not Path(path).exists():
        return empty
    try:
        df = pd.read_csv(path, dtype=str, comment="#").fillna("")
    except pd.errors.EmptyDataError:  # zero-byte or comment-only file
        return empty
    if "ticker" not in df.columns:
        raise ValueError(f"{path}: needs a 'ticker' header row")
    if "exchange" not in df.columns:
        df["exchange"] = ""
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["exchange"] = df["exchange"].str.strip().str.upper()
    return df[df["ticker"] != ""].drop_duplicates("ticker").reset_index(drop=True)[["ticker", "exchange"]]


def universe(settings) -> tuple[pd.DataFrame, str]:
    """The owner's watchlist, or the configured default list when it's missing or empty."""
    df = read_watchlist(settings.watchlist_csv)
    if not df.empty:
        return df, str(settings.watchlist_csv)
    return pd.DataFrame({"ticker": list(settings.default_tickers), "exchange": ""}), "default"


def read_symbol_map(path: Path) -> dict[str, dict[str, str]]:
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        ticker = row.ticker.strip().upper()
        if row.source not in SOURCES:
            raise ValueError(f"symbol_map: {ticker} has unknown source {row.source!r}")
        if row.asset_class not in ASSET_CLASSES:
            raise ValueError(f"symbol_map: {ticker} has unknown asset_class {row.asset_class!r}")
        out[ticker] = {"source": row.source, "source_symbol": row.source_symbol.strip(), "asset_class": row.asset_class}
    return out


def resolve(watchlist: pd.DataFrame, overrides: dict[str, dict[str, str]]) -> list[Instrument]:
    """Map file wins; otherwise US-exchange (or exchange-less) tickers go to Alpaca stocks;
    a ticker listed on any other exchange is unmapped."""
    instruments = []
    for row in watchlist.itertuples(index=False):
        if row.ticker in overrides:
            o = overrides[row.ticker]
            instruments.append(Instrument(row.ticker, row.exchange, o["source"], o["source_symbol"], o["asset_class"]))
        elif row.exchange in US_EXCHANGES or row.exchange == "":
            instruments.append(Instrument(row.ticker, row.exchange, "alpaca_stock", row.ticker, "us_equity_extended"))
        else:
            instruments.append(Instrument(row.ticker, row.exchange, None, None, None))
    return instruments


def load_instruments(settings) -> list[Instrument]:
    watchlist, _ = universe(settings)
    return resolve(watchlist, read_symbol_map(settings.data.symbol_map_csv))
