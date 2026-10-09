# M2 benchmarks (2026-10-03, i7-1355U, CPU only, Kronos-base unless noted)

Run from the project root, e.g. `.venv\Scripts\python benchmarks\kv_exactness.py` (needs the bar cache).

## KV cache exactness (`kv_exactness.py`, real AAPL bars)
| model | hidden-state max abs diff | s1 logits max abs diff | mean of paths vs upstream `predict()` (prices) |
|---|---|---|---|
| small | 7.3e-07 | 8.6e-06 | 9.7e-08 relative |
| base  | 2.4e-06 | 7.6e-06 | 9.4e-08 relative |

Same seed => same sampled tokens as upstream. Speed, 4 paths x 32 steps: base 95.4s -> 3.5s, small 20.5s -> 1.3s.

## Threads (`thread_and_path_speed.py`, 30 paths x 32 steps)
| threads | 2 | 4 | 6 | 8 | 10 (default) | 12 |
|---|---|---|---|---|---|---|
| seconds | 28.6 | 19.1 | 10.4 | **9.9** | 11.8 | 11.2 |

## Path count, 8 threads (seconds per instrument)
| paths | 20 | 30 | 50 | 100 |
|---|---|---|---|---|
| 32 steps (US stocks) | 7.8 | 11.2 | 18.3 | 35.7 |
| 48 steps (crypto/FX) | 12.4 | 17.2 | 22.2 | 38.2 |

## Quantile stability (`path_count_stability.py`)
Std error of the quantile estimate as % of the p10-p90 band width, from 400 reference paths
(AAPL, NVDA, BTCUSD, XAUUSD), 300 subsamples each, finite-population corrected. Median over symbols:

| paths | end close p10 / p50 / p90 | max high p10 / p50 / p90 | min low p10 / p50 / p90 | worst symbol p10 / p50 / p90 |
|---|---|---|---|---|
| 20  | 16.3 / 9.9 / 15.2 | 9.5 / 10.9 / 14.8 | 26.3 / 9.9 / 10.2 | 30.0 / 16.6 / 24.1 |
| 30  | 14.6 / 8.2 / 13.3 | 8.5 / 9.8 / 13.8  | 22.3 / 7.7 / 8.1  | 26.0 / 12.9 / 18.0 |
| 50  | 11.5 / 6.6 / 11.0 | 7.0 / 7.8 / 10.7  | 17.7 / 6.4 / 6.5  | 22.8 / 10.6 / 14.1 |
| 100 | 8.6 / 4.7 / 7.1   | 4.7 / 6.1 / 7.8   | 12.9 / 4.7 / 4.7  | 17.1 / 7.8 / 10.9  |

Chosen: 100 paths, 8 threads.
