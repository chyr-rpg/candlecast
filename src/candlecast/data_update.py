"""Bring every watchlist instrument's local 1h bar cache up to date and report coverage."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

from candlecast.bars import merge_incremental, normalize, read_cache, write_cache
from candlecast.sources import make_sources
from candlecast.symbols import Instrument

log = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    ticker: str
    action: str          # full | incremental | full_after_revision | skipped | error
    new_bars: int = 0
    error: str = ""


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def update_instrument(inst: Instrument, source, data_settings, bars_dir, full: bool = False,
                      now: pd.Timestamp | None = None) -> UpdateResult:
    now = now or _now()
    fetch_end = (now - source.min_delay).floor("min")
    cached = None if full else read_cache(bars_dir, inst.ticker)

    def _full(action: str) -> UpdateResult:
        start = fetch_end - pd.Timedelta(days=data_settings.history_days)
        frame = normalize(source.fetch(inst.source_symbol, start.to_pydatetime(), fetch_end.to_pydatetime()), fetch_end)
        if frame.empty:
            return UpdateResult(inst.ticker, "error", error="source returned no bars")
        write_cache(bars_dir, inst.ticker, frame)
        return UpdateResult(inst.ticker, action, len(frame))

    if cached is None or cached.empty:
        return _full("full")

    start = cached["timestamp"].max() - pd.Timedelta(hours=data_settings.overlap_hours)
    fresh = normalize(source.fetch(inst.source_symbol, start.to_pydatetime(), fetch_end.to_pydatetime()), fetch_end)
    if fresh.empty:
        return UpdateResult(inst.ticker, "incremental", 0)
    merged = merge_incremental(cached, fresh, data_settings.revision_rtol)
    if merged.revised:
        log.info("%s: overlap differs by %.4f (> %.4f), re-downloading full history",
                 inst.ticker, merged.max_overlap_diff, data_settings.revision_rtol)
        return _full("full_after_revision")
    write_cache(bars_dir, inst.ticker, merged.frame)
    return UpdateResult(inst.ticker, "incremental", merged.new_bars)


def update_all(instruments: list[Instrument], settings, full: bool = False, sources=None) -> list[UpdateResult]:
    factories = sources or make_sources()
    built: dict[str, object] = {}
    results = []
    for inst in instruments:
        if inst.source is None:
            results.append(UpdateResult(inst.ticker, "skipped", error=f"no source for exchange {inst.exchange}"))
            continue
        try:
            if inst.source not in built:
                built[inst.source] = factories[inst.source]()
            result = update_instrument(inst, built[inst.source], settings.data, settings.bars_dir, full=full)
        except Exception as exc:  # one instrument's failure never stops the run
            log.warning("%s: update failed: %s", inst.ticker, exc)
            result = UpdateResult(inst.ticker, "error", error=f"{type(exc).__name__}: {exc}"[:300])
        results.append(result)
        log.info("%s: %s (+%d bars)%s", inst.ticker, result.action, result.new_bars,
                 f" {result.error}" if result.error else "")
    return results


def coverage(instruments: list[Instrument], settings, update_results: list[UpdateResult] | None = None,
             now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or _now()
    by_ticker = {r.ticker: r for r in update_results or []}
    rows = []
    for inst in instruments:
        row = {"ticker": inst.ticker, "exchange": inst.exchange, "source": inst.source,
               "source_symbol": inst.source_symbol, "asset_class": inst.asset_class}
        upd = by_ticker.get(inst.ticker)
        if upd:
            row.update({"last_action": upd.action, "error": upd.error})
        df = read_cache(settings.bars_dir, inst.ticker) if inst.source else None
        if df is None or df.empty:
            row["status"] = "unmapped" if inst.source is None else "missing"
            rows.append(row)
            continue
        ts = df["timestamp"]
        recent = df[ts >= ts.max() - pd.Timedelta(days=60)]
        per_day = recent.groupby(recent["timestamp"].dt.date).size()
        row.update({
            "bars": len(df),
            "first_bar": ts.min(),
            "last_bar": ts.max(),
            "median_bars_per_day_60d": float(per_day.median()) if len(per_day) else 0.0,
            "zero_volume_frac": float((df["volume"] == 0).mean()),
            "max_abs_1h_return": float(df["close"].pct_change().abs().max()),
        })
        if len(df) < settings.data.min_bars:
            row["status"] = "short"
        elif now - ts.max() > pd.Timedelta(days=settings.data.stale_days):
            row["status"] = "stale"
        else:
            row["status"] = "ok"
        rows.append(row)
    return pd.DataFrame(rows)


def results_frame(results: list[UpdateResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])
