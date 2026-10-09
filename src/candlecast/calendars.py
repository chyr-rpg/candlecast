"""Future 1h bar timestamps per asset class, and the clock Kronos sees.

All bar timestamps are UTC bar-open times on the hour. Kronos gets New York wall-clock time
(``to_model_time``) for every instrument so its hour/weekday features mean the same thing across
markets.

Sessions (New York time):
* us_equity_extended: NYSE trading days (exchange_calendars XNYS, holidays and early closes
  included), 04:00 to regular close + 4h (20:00, or 17:00 on 13:00 early-close days).
* crypto: every hour.
* fx: Sunday 17:00 to Friday 17:00.
* commodity: CME Globex, Sunday 18:00 to Friday 17:00 with a daily 17:00-18:00 break.
  Exchange holidays are not modelled for fx/commodity (a forecast may include a holiday's bars).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

NY = "America/New_York"
HOUR = pd.Timedelta(hours=1)


def to_model_time(ts_utc: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """UTC timestamps -> naive New York wall-clock time (what Kronos's time features read)."""
    s = pd.Series(pd.DatetimeIndex(ts_utc))
    return s.dt.tz_convert(NY).dt.tz_localize(None)


@lru_cache(maxsize=1)
def _xnys():
    import exchange_calendars as xc  # noqa: PLC0415

    return xc.get_calendar("XNYS")


def _equity_hours(start_ny: pd.Timestamp, days: int) -> list[pd.Timestamp]:
    cal = _xnys()
    first = start_ny.normalize().tz_localize(None)
    sessions = cal.sessions_in_range(first, first + pd.Timedelta(days=days))
    hours = []
    for session in sessions:
        close_ny = cal.session_close(session).tz_convert(NY)
        day = pd.Timestamp(session.date()).tz_localize(NY)
        open_ext = day + pd.Timedelta(hours=4)
        close_ext = close_ny + pd.Timedelta(hours=4)
        h = open_ext
        while h + HOUR <= close_ext:
            hours.append(h)
            h += HOUR
    return hours


def _is_open(asset_class: str, ts_ny: pd.Timestamp) -> bool:
    wd, hr = ts_ny.weekday(), ts_ny.hour
    if asset_class == "crypto":
        return True
    if asset_class == "fx":
        if wd == 5:
            return False
        if wd == 4:
            return hr < 17
        if wd == 6:
            return hr >= 17
        return True
    if asset_class == "commodity":
        if wd == 5:
            return False
        if wd == 4:
            return hr < 17
        if wd == 6:
            return hr >= 18
        return hr != 17
    raise ValueError(f"unknown asset class {asset_class!r}")


def future_bars(asset_class: str, last_bar_utc: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """The next ``n`` expected bar-open times (UTC) strictly after ``last_bar_utc``."""
    last_ny = pd.Timestamp(last_bar_utc).tz_convert(NY)
    if asset_class == "us_equity_extended":
        days = 7
        while True:
            hours = [h for h in _equity_hours(last_ny, days) if h > last_ny]
            if len(hours) >= n:
                return pd.DatetimeIndex(hours[:n]).tz_convert("UTC")
            days *= 2
    out = []
    h = last_ny.floor("h")
    while len(out) < n:
        # step in UTC so DST transitions neither repeat nor skip an hour
        h = (h.tz_convert("UTC") + HOUR).tz_convert(NY)
        if _is_open(asset_class, h):
            out.append(h)
    return pd.DatetimeIndex(out).tz_convert("UTC")
