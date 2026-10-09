# Pre-registered confirmation test: Kronos daily ranking (q5)

Written and committed BEFORE the run, so the result cannot be tuned after the fact.

## Why
In the 2026-10-07 queue (benchmarks/results/accuracy_20261007_2202_q5_1d_h40_p12_summary.txt), the
paper's own daily setup ranked 10 US instruments (AAPL MSFT GOOGL AMZN NVDA META TSLA DIA QQQ SPY)
with mean RankIC +0.129 (t = 2.01, median-based) / +0.123 (t = 1.94, mean-based) over 28 dates. It was
the only one of ~16 tested combinations to reach t ~ 2, so it may be chance. This run checks it on
instruments it has never seen.

## Fixed design (no changes after the run starts)
- Universe: `benchmarks/confirm_q5_universe.csv` - a broader watchlist of US stocks/ETFs
  (114) minus the 10 already tested, minus GOOG (same company as GOOGL) = 103 instruments.
- Data: Alpaca daily bars (SIP, split-adjusted) from `benchmarks/fetch_daily.py --tickers-file ...`.
- Dates: the same cutoff grid as q5 (SPY's daily bars as downloaded 2026-10-05, unchanged), cutoffs
  >= 2024-07-01 (after Kronos pretraining), 40 origins requested -> 28 dates.
- Model/sampling: Kronos-base, 50 paths, temperature 1.0, top_p 1.0, history 40 daily bars,
  forecast 12 daily bars (identical to q5).
- Command:
  `.venv\Scripts\python benchmarks\accuracy_backtest.py --aligned --asset-class us_equity_extended
   --tickers-file benchmarks/confirm_q5_universe.csv --bars 1d --history 40 --pred-len 12 --origins 40
   --min-cutoff 2024-07-01 --paths 50 --threads 8 --temperature 1.0 --top-p 1.0 --label confirm_q5`

## Pass criterion (decided now)
PRIMARY: median-based RankIC (`exp_ret_p50`, the signal that reached t = 2.01 in q5) has
mean > 0 AND t > 2.0 across the dates.
- Pass -> treat daily ranking as a real (if small) edge: candidate nightly ranking product and the
  only setup worth fine-tuning.
- Fail (mean <= 0 or t <= 2.0) -> the q5 result was chance; stop tuning Kronos; ship band calibration.
The mean-based signal and all other metrics are reported but do not change the verdict.

## Caveats known in advance
- Dates overlap q5 (same calendar), so a market-wide regime effect could still be shared; instruments do not.
- 103 instruments per date make each date's RankIC far less noisy than q5's 10.

## Outcome (2026-10-08, appended after the run; design above unchanged)
Run: `benchmarks/results/accuracy_20261008_1935_confirm_q5{.csv,_summary.txt}`, 2,837 forecasts,
28 dates, 100-103 instruments per date.

PRIMARY (median-based RankIC): mean **-0.083, t = -1.84**, positive on 39% of dates -> **FAIL**.
Mean-based: -0.086 (t = -1.84). Without NBIS (see below): -0.083 (t = -1.86) - same verdict.
The q5 result (+0.129, t = 2.01 on 10 instruments) did not replicate; the sign flipped. Treat it as
chance. Other metrics: MAE ratio vs no-change 1.19, end-close 80%-band coverage 39%.

Data note: NBIS has a long trading suspension in its history (636 unchanged daily closes), giving
zero past volatility on 4 early forecasts and breaking the volatility QLIKE averages in the summary.
Excluding those 4 rows: QLIKE GARCH 0.317 (best), hist 0.338, EWMA 0.382, Kronos 0.522 (worst).
