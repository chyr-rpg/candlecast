"""Kronos-base KV-cached speed: thread sweep (30 paths x 32 steps), then path counts at 8 threads."""
import time

import pandas as pd
import torch

from candlecast import kv_inference as kv
from candlecast.kronos_loader import predictor_for
from candlecast.settings import load_settings

pr = predictor_for(load_settings(model_name="base").model)
bars = pd.read_parquet("data/bars/1h/AAPL.parquet")


def win(P):
    L = 512 - P
    w = bars.iloc[-512:].reset_index(drop=True)
    ts = w["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return w.iloc[:L][kv.FEATURES], ts.iloc[:L], ts.iloc[L:]


def run(P, N):
    df, x, y = win(P)
    best = 1e9
    for _ in range(2):
        t0 = time.perf_counter()
        kv.forecast_paths(pr, df, x, y, P, N)
        best = min(best, time.perf_counter() - t0)
    return best


kv.forecast_paths(pr, *win(32), 32, 4)  # warm-up
for th in [2, 4, 6, 8, 10, 12]:
    torch.set_num_threads(th)
    print(f"threads={th:2d}  P=32 N=30: {run(32, 30):6.1f}s", flush=True)
torch.set_num_threads(8)
for P in [32, 48]:
    for N in [20, 30, 50, 100]:
        print(f"threads=8 P={P} N={N:3d}: {run(P, N):6.1f}s", flush=True)
