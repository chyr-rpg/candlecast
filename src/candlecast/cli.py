"""Command-line entry point: ``candlecast <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from candlecast.settings import ALLOWED_MODELS, load_secret, load_settings


def synthetic_bars(n: int = 512, seed: int = 0, start: str = "2026-01-05 09:00") -> pd.DataFrame:
    """Geometric-random-walk hourly OHLCV with a ``timestamp`` column (for smoke tests only)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(12.0, 0.5, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * (open_ + high + low + close) / 4.0,
        }
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    from candlecast.kronos_loader import configure_torch_threads, predictor_for

    threads = configure_torch_threads(load_settings().model.torch_threads)
    names = list(ALLOWED_MODELS) if args.model == "all" else [args.model]
    bars = synthetic_bars(args.context)
    x_ts = bars["timestamp"]
    y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(hours=1), periods=args.steps, freq="h"))
    ok = True
    for name in names:
        model = load_settings(model_name=name).model
        t0 = time.perf_counter()
        predictor = predictor_for(model)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = predictor.predict(
            df=bars[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=args.steps,
            T=model.temperature,
            top_p=model.top_p,
            sample_count=args.samples,
            verbose=False,
        )
        infer_s = time.perf_counter() - t0
        finite = bool(np.isfinite(pred.to_numpy()).all())
        last = float(bars["close"].iloc[-1])
        ok &= finite and len(pred) == args.steps
        print(
            f"[{name}] {model.repo}: load {load_s:.1f}s, {args.steps} steps x {args.samples} samples "
            f"in {infer_s:.1f}s (torch threads={threads}); rows={len(pred)} finite={finite}; "
            f"last close {last:.2f} -> forecast close {pred['close'].iloc[0]:.2f} .. {pred['close'].iloc[-1]:.2f}"
        )
    return 0 if ok else 1


def cmd_check_env(args: argparse.Namespace) -> int:
    missing = [k for k in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY") if not load_secret(k)]
    if missing:
        print("Missing in .env: " + ", ".join(missing))
        return 1
    print("Alpaca keys present (values not shown).")
    return 0


def _select(instruments, symbols: list[str] | None):
    if not symbols:
        return instruments
    wanted = {s.upper() for s in symbols}
    chosen = [i for i in instruments if i.ticker in wanted]
    unknown = wanted - {i.ticker for i in chosen}
    if unknown:
        raise SystemExit(f"Not in watchlist: {', '.join(sorted(unknown))}")
    return chosen


def _write_coverage(report: pd.DataFrame, settings) -> None:
    out = settings.root / "outputs" / "coverage_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    counts = report["status"].value_counts().to_dict()
    print(f"Coverage: {len(report)} instruments {counts} -> {out}")
    problems = report[report["status"] != "ok"]
    for row in problems.itertuples(index=False):
        err = getattr(row, "error", "") or ""
        print(f"  {row.ticker:8s} {row.status:8s} {row.source or '-'} {err}")


def cmd_update_bars(args: argparse.Namespace) -> int:
    from candlecast.data_update import coverage, update_all
    from candlecast.symbols import load_instruments

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    settings = load_settings()
    instruments = load_instruments(settings)
    selected = _select(instruments, args.symbols)
    t0 = time.perf_counter()
    results = update_all(selected, settings, full=args.full)
    actions = pd.Series([r.action for r in results]).value_counts().to_dict()
    print(f"Updated {len(results)} instruments in {time.perf_counter() - t0:.0f}s: {actions}")
    _write_coverage(coverage(instruments, settings, results), settings)
    return 0 if all(r.action != "error" for r in results) else 1


def cmd_coverage(args: argparse.Namespace) -> int:
    from candlecast.data_update import coverage
    from candlecast.symbols import load_instruments

    settings = load_settings()
    _write_coverage(coverage(load_instruments(settings), settings), settings)
    return 0


def _setup_run_logging(settings, run_tag: str) -> Path:
    settings.run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.run_log_dir / f"run_{run_tag}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    sh.setLevel(logging.INFO)
    root.addHandler(sh)
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def cmd_run(args: argparse.Namespace) -> int:
    from candlecast.pipeline import run
    from candlecast.symbols import load_instruments, universe

    settings = load_settings()
    log_path = _setup_run_logging(settings, pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ"))
    instruments = _select(load_instruments(settings), args.symbols)
    _, source = universe(settings)
    logging.info("Universe: %d instruments (%s); model Kronos-%s, %s paths", len(instruments), source,
                 settings.model.name, args.paths or settings.model.sample_paths)
    report = run(settings, instruments, n_paths=args.paths, update=not args.no_update,
                 skip_existing=args.skip_existing)
    print(f"Run {report.run_id}: {len(report.ok)} ok, {len(report.skipped)} unchanged, {len(report.failures)} failed, "
          f"{report.scored} earlier forecasts scored, in {report.seconds:.0f}s "
          f"-> {settings.forecasts_dir / report.run_date}  (log: {log_path})")
    for line in report.failures:
        print(f"  FAILED {line}")
    return 0 if not report.failures else 1


def cmd_show(args: argparse.Namespace) -> int:
    from candlecast import store
    from candlecast.calendars import NY

    settings = load_settings()
    if not settings.index_db.exists():
        print("No forecasts yet - run `candlecast run` first.")
        return 1
    con = store.connect(settings.index_db)
    try:
        row = store.latest(con, args.ticker, model=None if args.any_model else settings.model.name)
        test = store.latest_test(con, row["ticker"], row["as_of_bar"], row["model"]) if row else None
    finally:
        con.close()
    if row is None:
        print(f"No forecast for {args.ticker.upper()}.")
        return 1
    as_of = pd.Timestamp(row["as_of_bar"]).tz_convert(NY)
    start = pd.Timestamp(row["horizon_start"]).tz_convert(NY)
    end = pd.Timestamp(row["horizon_end"]).tz_convert(NY)
    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(row["created_at"])
    last = row["last_close"]

    def pct(v):
        return f"{v:,.2f} ({v / last - 1:+.2%})"

    print(f"{row['ticker']}  Kronos-{row['model']}, {row['n_paths']} paths, {row['pred_len']} bars"
          f"  (made {age.total_seconds() / 3600:.1f}h ago)")
    print(f"  as of        {as_of:%a %Y-%m-%d %H:%M} New York, last close {last:,.2f}")
    print(f"  horizon      {start:%a %m-%d %H:%M} -> {end:%a %m-%d %H:%M} New York")
    print(f"  P(end above last close)  {row['prob_up']:.0%}")
    print(f"  end close    p10 {pct(row['end_close_p10'])}   p50 {pct(row['end_close_p50'])}   "
          f"p90 {pct(row['end_close_p90'])}")
    print(f"  highest high p50 {pct(row['max_high_p50'])}   p90 {pct(row['max_high_p90'])}")
    print(f"  lowest low   p10 {pct(row['min_low_p10'])}   p50 {pct(row['min_low_p50'])}")
    if test:
        from candlecast.charts import caption_for_test

        cutoff = pd.Timestamp(test["cutoff_bar"]).tz_convert(NY)
        print(f"  test window  {pd.Timestamp(test['test_start']).tz_convert(NY):%a %m-%d %H:%M} -> "
              f"{pd.Timestamp(test['test_end']).tz_convert(NY):%a %m-%d %H:%M} New York")
        # the chart's "·" separator is not safe on every Windows console code page
        print("  " + caption_for_test(test, cutoff).replace(" · ", ", ").replace("\n", "\n  "))
    if age > pd.Timedelta(days=3):
        print("  WARNING: this forecast is more than 3 days old.")
    chart = settings.root / row["chart_path"] if row.get("chart_path") else None
    print(f"  chart        {chart if chart and chart.exists() else '(pruned)'}")
    print(f"  per-bar data {settings.root / row['summary_path']}")
    if args.bars:
        summary = pd.read_parquet(settings.root / row["summary_path"])
        summary["time (NY)"] = summary["timestamp"].dt.tz_convert(NY).dt.strftime("%a %m-%d %H:%M")
        cols = ["time (NY)", "close_p10", "close_p25", "close_p50", "close_p75", "close_p90", "prob_above_last"]
        print(summary[cols].round(2).to_string(index=False))
    if args.open and chart and chart.exists():
        import os

        os.startfile(chart)  # noqa: S606 - Windows: open in the default image viewer
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from candlecast import scoring, store

    settings = load_settings()
    if not settings.index_db.exists():
        print("No forecasts yet - run `candlecast run` first.")
        return 1
    con = store.connect(settings.index_db)
    try:
        if not args.report_only:
            print(f"Scored {scoring.score_pending(con, settings)} newly finished forecasts.")
        waiting = len(scoring.pending(con))
        model = None if args.any_model else settings.model.name
        table = scoring.report(con, model=model, since=args.since, by_ticker=args.by_ticker)
        ic_mean, ic_p50 = scoring.rank_ic_report(con, settings.root, model=model, since=args.since)
    finally:
        con.close()
    print(f"{waiting} forecasts still waiting for their horizon to finish.")
    if table.empty:
        print("No scored forecasts yet.")
        return 0
    print("Ideal: MAE ratio < 1, direction hit > 0.5, Brier < 0.25, cover ~0.80, CRPS ratio < 1.")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string())
    if table.loc["ALL", "n"] < 30:
        print(f"Only {int(table.loc['ALL', 'n'])} scored forecasts: too few to draw conclusions yet.")
    print("\nRanking US equities against each other per nightly run (the Kronos paper's RankIC; "
          "0 = no skill, > 0 = good):")
    print(f"  by mean expected return    {scoring.describe_rank_ic(ic_mean)}")
    print(f"  by median expected return  {scoring.describe_rank_ic(ic_p50)}")
    if 0 < len(ic_mean) < 20:
        print(f"  Only {len(ic_mean)} runs: a mean RankIC needs dozens of runs before it means anything.")
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    from candlecast.panel import main as panel_main

    panel_main(args.tickers)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="candlecast")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Load Kronos and forecast a synthetic series")
    smoke.add_argument("--model", choices=[*ALLOWED_MODELS, "all"], default="all")
    smoke.add_argument("--steps", type=int, default=32)
    smoke.add_argument("--samples", type=int, default=1)
    smoke.add_argument("--context", type=int, default=512)
    smoke.set_defaults(func=cmd_smoke)

    check_env = sub.add_parser("check-env", help="Report whether Alpaca keys are set")
    check_env.set_defaults(func=cmd_check_env)

    update = sub.add_parser("update-bars", help="Download/refresh the local 1h bar cache, then report coverage")
    update.add_argument("symbols", nargs="*", help="Watchlist tickers (default: all)")
    update.add_argument("--full", action="store_true", help="Re-download full history instead of incremental")
    update.add_argument("-v", "--verbose", action="store_true")
    update.set_defaults(func=cmd_update_bars)

    cov = sub.add_parser("coverage", help="Coverage report from the local cache only (no downloads)")
    cov.set_defaults(func=cmd_coverage)

    run = sub.add_parser("run", help="Nightly run: update bars, forecast, save files/charts, index")
    run.add_argument("symbols", nargs="*", help="Tickers from the active universe (default: all)")
    run.add_argument("--paths", type=int, default=None, help="override [model].sample_paths")
    run.add_argument("--no-update", action="store_true", help="skip the bar download, use the cache as is")
    run.add_argument("--skip-existing", action="store_true",
                     help="skip instruments whose latest bar already has a forecast (used by the nightly task)")
    run.set_defaults(func=cmd_run)

    show = sub.add_parser("show", help="Print the latest cached forecast for a ticker (no inference)")
    show.add_argument("ticker")
    show.add_argument("--bars", action="store_true", help="also print the per-bar quantile table")
    show.add_argument("--open", action="store_true", help="open the chart image")
    show.add_argument("--any-model", action="store_true", help="latest forecast from any model")
    show.set_defaults(func=cmd_show)

    score = sub.add_parser("score", help="Score finished forecasts and print the forward-test track record")
    score.add_argument("--by-ticker", action="store_true", help="break down by ticker instead of asset class")
    score.add_argument("--since", help="only forecasts made on/after this date (YYYY-MM-DD)")
    score.add_argument("--report-only", action="store_true", help="don't score, just print the report")
    score.add_argument("--any-model", action="store_true", help="include forecasts from every model")
    score.set_defaults(func=cmd_score)

    panel = sub.add_parser("panel", help="Open the desktop panel: type tickers, get forecast charts")
    panel.add_argument("tickers", nargs="*", help="fill these in and start the run immediately")
    panel.set_defaults(func=cmd_panel)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
