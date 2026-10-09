"""Bar timeframes for custom forecasts: 1h, 4h, 1d, 1w.

* History: a fixed number of bars, independent of how many bars are forecast (default 128, adjustable
  64-448), capped so history + forecast fits Kronos's 512-bar context. Because Kronos generates bar by bar,
  the same history makes a longer forecast simply extend a shorter one (the first N bars are identical);
  an earlier rule (8 x forecast bars) changed the history with the bar count and could flip the direction.
  A test window needs the forecast bars again, so a ticker needs history + forecast bars.
* Local first, online fallback: each timeframe has its own local bar database
  (data/bars/1h - the nightly cache, data/bars/1d, data/bars/1w). Every request refreshes it online
  (Alpaca for US stocks/ETFs and crypto, yfinance for FX/commodities); a database that is too short is
  re-downloaded with enough history; if the download fails, the local bars are used when they suffice.
* 4h bars are aggregated from the 1h database into New York-anchored buckets (00/04/08/12/16/20; US
  equities' 04:00-20:00 extended session gives 04/08/12/16). An unfinished latest bucket is dropped.
* Daily/weekly bars are kept only once the period is over (bar start + 1 day / 7 days <= now).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from candlecast import calendars
from candlecast.bars import merge_incremental, normalize, read_cache, write_cache

log = logging.getLogger(__name__)

TIMEFRAMES = ("1h", "4h", "1d", "1w")
LABELS = {"1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"}
DURATION = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1),
            "1w": pd.Timedelta(days=7)}
MIN_PRED, MAX_PRED = 4, 128
DEFAULT_PRED = {"1h": 32, "4h": 16, "1d": 16, "1w": 12}
DEFAULT_HISTORY, MIN_HISTORY, MAX_HISTORY = 128, 64, 448
MIN_USABLE_HISTORY = 32          # below this a forecast is refused rather than shown
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}


def history_bars(pred_len: int, max_context: int = 512, history: int | None = None) -> int:
    """History bars used: ``history`` (default 128) clamped to 64-448, and capped so history + forecast
    fits the model's context. Independent of ``pred_len`` except for that cap."""
    h = DEFAULT_HISTORY if history is None else int(history)
    return min(max(h, MIN_HISTORY), MAX_HISTORY, max_context - pred_len)


def bars_required(pred_len: int, max_context: int = 512, history: int | None = None) -> int:
    """Bars a ticker needs: the history plus the forecast bars again for the test window."""
    return history_bars(pred_len, max_context, history) + pred_len


def validate(timeframe: str, pred_len: int) -> None:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {TIMEFRAMES}, got {timeframe!r}")
    if not MIN_PRED <= int(pred_len) <= MAX_PRED:
        raise ValueError(f"forecast bars must be {MIN_PRED}-{MAX_PRED}, got {pred_len}")


# ------------------------------------------------------------------ 4h aggregation

def bucket_4h(ts_utc: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """UTC start of the New York-anchored 4h bucket containing each hourly bar."""
    ny = pd.Series(pd.DatetimeIndex(ts_utc)).dt.tz_convert(calendars.NY)
    start = ny.dt.normalize() + pd.to_timedelta(4 * (ny.dt.hour // 4), unit="h")
    return start.dt.tz_convert("UTC")


def aggregate_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly is None or hourly.empty:
        return hourly
    out = (hourly.assign(timestamp=bucket_4h(hourly["timestamp"]).to_numpy())
           .groupby("timestamp", sort=True).agg(_AGG).reset_index())
    last_hour_end = hourly["timestamp"].max() + pd.Timedelta(hours=1)
    return out[out["timestamp"] + DURATION["4h"] <= last_hour_end].reset_index(drop=True)


# ------------------------------------------------------------------ future bar timestamps

def future_bars(asset_class: str, timeframe: str, last_bar_utc: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """The next ``n`` expected bar-open times (UTC) after ``last_bar_utc`` for this timeframe."""
    last = pd.Timestamp(last_bar_utc)
    if timeframe == "1h":
        return calendars.future_bars(asset_class, last, n)
    if timeframe == "4h":
        hours = 4 * n + 16
        while True:  # the next hourly bars after this 4h bucket, grouped into buckets
            hourly = calendars.future_bars(asset_class, last + pd.Timedelta(hours=3), hours)
            starts = pd.unique(bucket_4h(hourly))
            starts = [s for s in starts if s > last]
            if len(starts) > n:  # the last bucket may be incomplete in the generated hours
                return pd.DatetimeIndex(starts[:n])
            hours *= 2
    if timeframe == "1w":
        return pd.DatetimeIndex([last + DURATION["1w"] * k for k in range(1, n + 1)])
    if timeframe == "1d":
        if asset_class == "us_equity_extended":
            cal = calendars._xnys()
            day = last.tz_convert(calendars.NY).normalize().tz_localize(None)
            sessions = cal.sessions_in_range(day + pd.Timedelta(days=1), day + pd.Timedelta(days=2 * n + 15))
            midnight_ny = [pd.Timestamp(s.date()).tz_localize(calendars.NY) for s in sessions[:n]]
            return pd.DatetimeIndex(midnight_ny).tz_convert("UTC")
        out, k = [], 1
        while len(out) < n:
            ts = last + pd.Timedelta(days=k)
            if asset_class == "crypto" or ts.tz_convert(calendars.NY).weekday() < 5:
                out.append(ts)
            k += 1
        return pd.DatetimeIndex(out)
    raise ValueError(f"unknown timeframe {timeframe!r}")


# ------------------------------------------------------------------ local database + online fallback

def bars_dir(settings, timeframe: str) -> Path:
    return Path(settings.bars_dir).parent / ("1h" if timeframe == "4h" else timeframe)


def _lookback_days(timeframe: str, n_bars: int) -> int:
    if timeframe == "1w":
        return int(n_bars * 7 * 1.15) + 30
    return int(n_bars * 1.6) + 30         # 1d: weekends/holidays (equities ~252 bars a year)


def _complete(frame: pd.DataFrame, timeframe: str, now: pd.Timestamp) -> pd.DataFrame:
    return frame[frame["timestamp"] + DURATION[timeframe] <= now].reset_index(drop=True)


def _fetch(source, inst, timeframe: str, start: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    end = (now - source.min_delay).floor("min")
    raw = source.fetch(inst.source_symbol, start.to_pydatetime(), end.to_pydatetime(), timeframe=timeframe)
    return _complete(normalize(raw), timeframe, now)


def load_bars(settings, inst, timeframe: str, required: int, source=None,
              now: pd.Timestamp | None = None) -> tuple[pd.DataFrame | None, str]:
    """Bars for ``inst`` at ``timeframe`` (at least ``required`` when available) and a note on where
    they came from. ``source``: a data source object for the instrument (built from make_sources if None)."""
    from candlecast.data_update import update_instrument  # noqa: PLC0415
    from candlecast.sources import make_sources  # noqa: PLC0415

    now = now or pd.Timestamp.now(tz="UTC")
    if source is None and inst.source is not None:
        try:
            source = make_sources()[inst.source]()
        except Exception as exc:  # e.g. missing Alpaca keys: fall back to the local database
            log.warning("%s: data source unavailable (%s); using local bars only", inst.ticker, exc)

    if timeframe in ("1h", "4h"):
        note = "local 1h database"
        if source is not None:
            try:
                r = update_instrument(inst, source, settings.data, settings.bars_dir, now=now)
                note = f"1h database refreshed online ({r.action}, +{r.new_bars} bars)"
                if r.action == "error":
                    note = f"online refresh failed ({r.error}); local 1h database"
            except Exception as exc:
                note = f"online refresh failed ({type(exc).__name__}); local 1h database"
        bars = read_cache(settings.bars_dir, inst.ticker)
        if bars is not None and timeframe == "4h":
            bars = aggregate_4h(bars)
        return bars, note

    directory = bars_dir(settings, timeframe)
    cached = read_cache(directory, inst.ticker)
    have = 0 if cached is None else len(cached)
    if source is None:
        return cached, f"local {timeframe} database ({have} bars; offline)"
    try:
        if cached is None or have < required:
            start = now - pd.Timedelta(days=_lookback_days(timeframe, required))
            if cached is not None and not cached.empty:
                start = min(start, cached["timestamp"].min())
            fresh = _fetch(source, inst, timeframe, start, now)
            if fresh.empty:
                raise ValueError("source returned no bars")
            write_cache(directory, inst.ticker, fresh)
            return fresh, f"{timeframe} history downloaded online ({len(fresh)} bars)"
        start = cached["timestamp"].max() - 4 * DURATION[timeframe]
        fresh = _fetch(source, inst, timeframe, start, now)
        if fresh.empty:
            return cached, f"local {timeframe} database ({have} bars; no new bars online)"
        merged = merge_incremental(cached, fresh, settings.data.revision_rtol)
        if merged.revised:  # e.g. a split re-adjusted history: take a clean full copy
            full = _fetch(source, inst, timeframe, now - pd.Timedelta(days=_lookback_days(timeframe, required)), now)
            write_cache(directory, inst.ticker, full)
            return full, f"{timeframe} history re-downloaded after a revision ({len(full)} bars)"
        write_cache(directory, inst.ticker, merged.frame)
        return merged.frame, f"local {timeframe} database refreshed online (+{merged.new_bars} bars)"
    except Exception as exc:
        if cached is not None and not cached.empty:
            log.warning("%s: %s download failed (%s); using local database", inst.ticker, timeframe, exc)
            return cached, f"online download failed ({type(exc).__name__}); local {timeframe} database ({have} bars)"
        raise
