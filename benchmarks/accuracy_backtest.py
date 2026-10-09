"""Benchmark B: Kronos forecast accuracy at past start dates vs simple baselines.

For each instrument in the active universe and each of ``--origins`` past start dates (spaced so
horizons don't overlap), Kronos sees only bars before the start date and forecasts the next
``pred_len`` bars (the actual next bars' timestamps are used, so the bar grid matches the
outcome exactly). Scored against what happened:

* point: |median end close - actual| vs the "no change" baseline (last close);
* direction: sign of median move, and Brier score of P(up) vs 0.25 for a coin flip;
* bands: share of outcomes inside the p10-p90 band (ideal 80%) for end close, highest high and
  lowest low over the horizon;
* CRPS of the end-close distribution vs a random walk scaled to recent hourly volatility.

Ranking (``--aligned``): every instrument of ``--asset-class`` is forecast from the SAME cutoff
timestamps (taken from the reference instrument's bar grid, SPY by default), so on each date the
instruments can be ranked by expected return, as the Kronos paper evaluates it (RankIC = Spearman
correlation, across instruments on one date, between expected return and actual return). Expected
return uses the MEAN of the sampled end closes (the paper's signal) and also the median.

Usage: .venv\\Scripts\\python benchmarks\\accuracy_backtest.py [--origins 20] [--paths 100] [--threads 6]
       ... --aligned --asset-class us_equity_extended --temperature 0.6 --label T06
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from candlecast import research
from candlecast.forecaster import forecast_from_bars, horizon_summary
from candlecast.kronos_loader import predictor_for
from candlecast.settings import load_settings
from candlecast.scoring import crps_ensemble, crps_normal, describe_rank_ic, rank_ic_by_group
from candlecast.symbols import load_instruments, read_symbol_map, read_watchlist, resolve


def ranking_text(df: pd.DataFrame) -> str:
    """Cross-sectional ranking skill per cutoff date (see candlecast.scoring.rank_ic_by_group)."""
    lines = ["Ranking across instruments on each date (RankIC; 0 = no skill, >0 = ranks returns correctly):"]
    for signal in ("exp_ret_mean", "exp_ret_p50"):
        per_date = rank_ic_by_group(df, "cutoff", signal)
        lines.append(f"  {signal:13s} {describe_rank_ic(per_date)}")
    return "\n".join(lines)


VOL_METHODS = ("rv_kronos_mean", "rv_kronos_median", "rv_hist", "rv_ewma", "rv_garch")


def volatility_text(df: pd.DataFrame) -> str:
    """Item 4: realized-volatility forecasts vs what happened. QLIKE is the standard loss (lower =
    better); 'timing' = within-ticker Spearman correlation with actual volatility, averaged over tickers
    (does the method know WHEN this instrument will move more, beyond its usual level)."""
    lines = ["Realized volatility over the forecast window (lower MAE/QLIKE = better; timing > 0 = good):",
             f"  {'method':17s} {'MAE %':>7s} {'QLIKE':>7s} {'timing':>7s}"]
    d = df[df["rv_actual"] > 0]
    for m in VOL_METHODS:
        timing = np.mean([g[m].rank().corr(g["rv_actual"].rank()) for _, g in d.groupby("ticker") if len(g) > 2])
        lines.append(f"  {m:17s} {100 * (d[m] - d['rv_actual']).abs().mean():7.3f} "
                     f"{research.qlike(d['rv_actual'], d[m]).mean():7.3f} {timing:+7.3f}")
    return "\n".join(lines)


def instruments_for(settings, tickers_file: str | None):
    """The forecast universe, or the instruments listed in ``tickers_file`` (resolved the same way)."""
    if not tickers_file:
        return load_instruments(settings)
    return resolve(read_watchlist(tickers_file), read_symbol_map(settings.data.symbol_map_csv))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=20)
    ap.add_argument("--paths", type=int, default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--top-p", type=float, default=None, help="override [model].top_p")
    ap.add_argument("--temperature", type=float, default=None, help="override [model].temperature")
    ap.add_argument("--aligned", action="store_true",
                    help="same cutoff timestamps for every instrument (needed for the ranking/RankIC test)")
    ap.add_argument("--asset-class", default=None, help="only instruments of this asset class")
    ap.add_argument("--tickers-file", default=None,
                    help="CSV (ticker[,exchange]) to use instead of the forecast universe")
    ap.add_argument("--reference", default="SPY", help="--aligned: instrument whose bar grid sets the dates")
    ap.add_argument("--history", type=int, default=None, help="history bars (default max_context - horizon)")
    ap.add_argument("--bars", choices=research.BAR_KINDS, default="1h",
                    help="bar series (see candlecast.research); default the nightly extended-hours 1h bars")
    ap.add_argument("--pred-len", type=int, default=None,
                    help="forecast steps (default: nightly horizon for 1h, ~2 sessions for the other bar kinds)")
    ap.add_argument("--min-cutoff", default=None,
                    help="skip cutoff dates before this (Kronos pretraining ends June 2024 -> use 2024-07-01)")
    ap.add_argument("--label", default="", help="suffix for the output file names")
    ap.add_argument("--save-samples", action="store_true",
                    help="also save each forecast's end-close / highest-high / lowest-low samples (.npz)")
    args = ap.parse_args()

    settings = load_settings()
    model = settings.model
    if args.top_p is not None:
        model = replace(model, top_p=args.top_p)
    if args.temperature is not None:
        model = replace(model, temperature=args.temperature)
    n_paths = args.paths or model.sample_paths
    torch.set_num_threads(args.threads)
    predictor = predictor_for(model)
    out_dir = settings.root / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M") + (f"_{args.label}" if args.label else "")
    out_csv = out_dir / f"accuracy_{stamp}.csv"

    rows = []
    samples = {"end_close": [], "max_high": [], "min_low": []}
    t_start = time.perf_counter()
    instruments = [i for i in instruments_for(settings, args.tickers_file)
                   if args.asset_class is None or i.asset_class == args.asset_class]
    if args.bars != "1h" and any(i.asset_class != "us_equity_extended" for i in instruments):
        raise SystemExit(f"--bars {args.bars} is only built for US equities - add --asset-class us_equity_extended")
    min_cutoff = pd.Timestamp(args.min_cutoff, tz="UTC") if args.min_cutoff else None

    def steps_for(asset_class: str) -> int:
        if args.pred_len:
            return args.pred_len
        return settings.horizons[asset_class] if args.bars == "1h" else research.TWO_SESSION_STEPS[args.bars]

    cutoffs = None
    if args.aligned:
        classes = {i.asset_class for i in instruments}
        if len(classes) != 1:
            raise SystemExit(f"--aligned needs one asset class (one horizon); got {sorted(classes)} - use --asset-class")
        ref = research.load_bars(settings, args.reference, args.bars)
        ref_len = steps_for(classes.pop())
        # cutoff k = last bar seen; its horizon (ref_len bars) ends before cutoff k-1 (+8 bars gap)
        cutoffs = [ref["timestamp"].iloc[i] for k in range(args.origins)
                   if (i := len(ref) - ref_len - 1 - k * (ref_len + 8)) >= 0]
        if min_cutoff is not None:
            cutoffs = [c for c in cutoffs if c >= min_cutoff]
    for inst in instruments:
        bars = research.load_bars(settings, inst.ticker, args.bars)
        if bars is None or bars.empty:
            print(f"{inst.ticker}: no {args.bars} bars, skipped", flush=True)
            continue
        pred_len = steps_for(inst.asset_class)
        history_len = min(args.history or model.max_context, model.max_context - pred_len)
        spacing = pred_len + 8
        for k in range(len(cutoffs) if cutoffs is not None else args.origins):
            if cutoffs is not None:  # bars up to and including the shared cutoff timestamp
                o = int(bars["timestamp"].searchsorted(cutoffs[k], side="right"))
            else:
                o = len(bars) - pred_len - k * spacing
            if o - history_len < 0:
                break  # dates only get older from here
            if min_cutoff is not None and bars["timestamp"].iloc[o - 1] < min_cutoff:
                break
            if o + pred_len > len(bars):
                continue  # this date's outcome isn't in this instrument's cache yet; older dates still are
            past = bars.iloc[:o]
            actual = bars.iloc[o:o + pred_len]
            res = forecast_from_bars(inst.ticker, inst.asset_class, past, predictor, model, pred_len, n_paths,
                                     future_ts=pd.DatetimeIndex(actual["timestamp"]), history_len=history_len)
            h = horizon_summary(res)
            last = res.last_close
            y_end, y_hi, y_lo = actual["close"].iloc[-1], actual["high"].max(), actual["low"].min()
            ends = res.paths[:, -1, 3]
            if args.save_samples:
                samples["end_close"].append(ends)
                samples["max_high"].append(res.paths[:, :, 1].max(axis=1))
                samples["min_low"].append(res.paths[:, :, 2].min(axis=1))
            # baseline volatility always from the full 480-ish bar window so variants share one yardstick
            vol_window = past["close"].tail(model.max_context - pred_len).to_numpy()
            logret = np.diff(np.log(vol_window))
            sigma = last * logret.std() * math.sqrt(pred_len)
            # item 4: realized volatility over the window (Kronos paper's definition) vs baselines
            path_closes = np.concatenate([np.full((res.paths.shape[0], 1), last), res.paths[:, :, 3]], axis=1)
            rv_paths = research.realized_vol(path_closes)
            vol = {
                "rv_actual": float(research.realized_vol(np.r_[last, actual["close"].to_numpy()])),
                "rv_kronos_mean": float(rv_paths.mean()), "rv_kronos_median": float(np.median(rv_paths)),
                "rv_hist": research.hist_vol(vol_window, pred_len),
                "rv_ewma": research.ewma_vol(vol_window, pred_len),
                "rv_garch": research.garch_vol(vol_window, pred_len),
            }
            rows.append({
                "ticker": inst.ticker, "asset_class": inst.asset_class, "origin": res.as_of_bar,
                "cutoff": cutoffs[k] if cutoffs is not None else res.as_of_bar,
                "pred_len": pred_len, "last_close": last, "actual_end": y_end,
                "end_close_mean": float(ends.mean()),
                "exp_ret_mean": float(ends.mean()) / last - 1, "exp_ret_p50": h["end_close_p50"] / last - 1,
                "actual_ret": y_end / last - 1,
                "actual_max_high": y_hi, "actual_min_low": y_lo, **h,
                "abs_err_kronos": abs(h["end_close_p50"] - y_end) / last,
                "abs_err_naive": abs(last - y_end) / last,
                "dir_hit": float(np.sign(h["end_close_p50"] - last) == np.sign(y_end - last)),
                "brier": (h["prob_up"] - float(y_end > last)) ** 2,
                "cover_end": float(h["end_close_p10"] <= y_end <= h["end_close_p90"]),
                "cover_high": float(h["max_high_p10"] <= y_hi <= h["max_high_p90"]),
                "cover_low": float(h["min_low_p10"] <= y_lo <= h["min_low_p90"]),
                "crps_kronos": crps_ensemble(ends, y_end) / last,
                "crps_rw": crps_normal(last, sigma, y_end) / last,
                **vol,
            })
            pd.DataFrame(rows).to_csv(out_csv, index=False)  # keep partial results if interrupted
            print(f"{inst.ticker} origin {k + 1}/{args.origins} done ({time.perf_counter() - t_start:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    if args.save_samples:  # row i of each array belongs to row i of the CSV
        np.savez_compressed(out_csv.with_suffix(".npz"), **{k: np.stack(v) for k, v in samples.items()})

    def agg(g):
        return pd.Series({
            "n": len(g),
            "MAE kronos %": 100 * g["abs_err_kronos"].mean(),
            "MAE no-change %": 100 * g["abs_err_naive"].mean(),
            "MAE ratio": g["abs_err_kronos"].mean() / g["abs_err_naive"].mean(),
            "direction hit": g["dir_hit"].mean(),
            "Brier (coin=0.25)": g["brier"].mean(),
            "cover end (80%)": g["cover_end"].mean(),
            "cover high (80%)": g["cover_high"].mean(),
            "cover low (80%)": g["cover_low"].mean(),
            "CRPS ratio vs RW": g["crps_kronos"].mean() / g["crps_rw"].mean(),
        })
    summary = pd.concat([df.groupby("asset_class").apply(agg), agg(df).to_frame("ALL").T])
    text = (f"Kronos {model.name}, {n_paths} paths, T={model.temperature}, top_p={model.top_p}, "
            f"bars={args.bars}, pred_len={df['pred_len'].iloc[0]}, "
            f"history={args.history or 'max'}, {args.origins} origins/instrument"
            f"{', aligned dates' if args.aligned else ''}"
            f"{f', cutoffs >= {args.min_cutoff}' if args.min_cutoff else ''}, "
            f"{len(df)} forecasts, {time.perf_counter() - t_start:.0f}s\n\n{summary.round(3).to_string()}\n\n"
            f"Per ticker:\n{df.groupby('ticker').apply(agg).round(3).to_string()}\n")
    if args.aligned:
        text += "\n" + ranking_text(df) + "\n"
    text += "\n" + volatility_text(df) + "\n"
    (out_dir / f"accuracy_{stamp}_summary.txt").write_text(text)
    print(text)
    print(f"rows: {out_csv}")


if __name__ == "__main__":
    main()
