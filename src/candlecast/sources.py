"""Bar sources (hourly by default; daily "1d" and weekly "1w" on request). Each returns a raw frame for ``normalize``: a ``timestamp`` column (or
DatetimeIndex) plus open/high/low/close/volume and optionally vwap.

Alpaca stock bars include extended hours (04:00-20:00 ET) and are requested on the SIP feed, which
the free plan allows for bars older than 15 minutes. Prices are split-adjusted only: dividend
adjustment would re-price all history after every ex-date and force needless full re-downloads.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd

from candlecast.settings import load_secret

ALPACA_MIN_DELAY = pd.Timedelta(minutes=16)
SUPPORTED_TIMEFRAMES = ("1h", "1d", "1w")   # 4h is aggregated from 1h (candlecast.timeframes)


def _alpaca_timeframe(timeframe: str):
    from alpaca.data.timeframe import TimeFrame  # noqa: PLC0415

    try:
        return {"1h": TimeFrame.Hour, "1d": TimeFrame.Day, "1w": TimeFrame.Week}[timeframe]
    except KeyError:
        raise SourceError(f"unsupported timeframe {timeframe!r}") from None


class SourceError(RuntimeError):
    pass


def _bars_frame(response_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if response_df is None or response_df.empty:
        return pd.DataFrame()
    df = response_df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    return df


HTTP_TIMEOUT = (10, 60)   # seconds: connect, read - a stalled connection must fail, not hang forever


def _with_timeout(client):
    """alpaca-py sends requests without a timeout; give its session a default one."""
    import functools  # noqa: PLC0415

    session = getattr(client, "_session", None)
    if session is not None and not getattr(session, "_candlecast_timeout", False):
        original = session.request

        @functools.wraps(original)
        def request(method, url, **kwargs):
            kwargs.setdefault("timeout", HTTP_TIMEOUT)
            return original(method, url, **kwargs)

        session.request = request
        session._candlecast_timeout = True
    return client


class AlpacaStockSource:
    name = "alpaca_stock"
    min_delay = ALPACA_MIN_DELAY

    def __init__(self, client=None):
        if client is None:
            from alpaca.data.historical import StockHistoricalDataClient  # noqa: PLC0415

            key, secret = load_secret("ALPACA_API_KEY_ID"), load_secret("ALPACA_API_SECRET_KEY")
            if not key or not secret:
                raise SourceError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing from .env")
            client = _with_timeout(StockHistoricalDataClient(key, secret))
        self.client = client

    def fetch(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1h") -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed  # noqa: PLC0415
        from alpaca.data.requests import StockBarsRequest  # noqa: PLC0415

        request = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=_alpaca_timeframe(timeframe), start=start, end=end,
            feed=DataFeed.SIP, adjustment=Adjustment.SPLIT,
        )
        return _bars_frame(self.client.get_stock_bars(request).df, symbol)


class AlpacaCryptoSource:
    name = "alpaca_crypto"
    min_delay = pd.Timedelta(minutes=1)

    def __init__(self, client=None):
        if client is None:
            from alpaca.data.historical import CryptoHistoricalDataClient  # noqa: PLC0415

            client = _with_timeout(CryptoHistoricalDataClient())  # crypto data needs no keys
        self.client = client

    def fetch(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1h") -> pd.DataFrame:
        from alpaca.data.requests import CryptoBarsRequest  # noqa: PLC0415

        request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=_alpaca_timeframe(timeframe),
                                    start=start, end=end)
        return _bars_frame(self.client.get_crypto_bars(request).df, symbol)


class YFinanceSource:
    """Unofficial API; 1h history only reaches back ~730 days (daily/weekly go back decades)."""

    name = "yfinance"
    min_delay = pd.Timedelta(minutes=1)
    max_lookback = pd.Timedelta(days=729)

    def fetch(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1h") -> pd.DataFrame:
        import yfinance as yf  # noqa: PLC0415

        interval = {"1h": "1h", "1d": "1d", "1w": "1wk"}.get(timeframe)
        if interval is None:
            raise SourceError(f"unsupported timeframe {timeframe!r}")
        if timeframe == "1h":
            start = max(pd.Timestamp(start), pd.Timestamp(end) - self.max_lookback)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            raise SourceError(f"yfinance returned no data for {symbol}")
        df = df.rename(columns=str.lower)
        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]].reset_index()


def make_sources() -> dict[str, object]:
    """Sources are created lazily so a missing key only breaks the instruments that need it."""
    return {"alpaca_stock": AlpacaStockSource, "alpaca_crypto": AlpacaCryptoSource, "yfinance": YFinanceSource}
