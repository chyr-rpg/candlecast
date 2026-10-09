import time, numpy as np, pandas as pd, torch
from candlecast.settings import load_settings
from candlecast.kronos_loader import predictor_for
from candlecast import kv_inference as kv

bars = pd.read_parquet("data/bars/1h/AAPL.parquet")
def window(L, P):
    w = bars.iloc[-(L + P) - 100:-100].reset_index(drop=True)
    ts = w["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return w.iloc[:L][kv.FEATURES], ts.iloc[:L], ts.iloc[L:L + P]

for name in ["small", "base"]:
    m = load_settings(model_name=name).model
    pr = predictor_for(m)
    # (a) hidden-state agreement: incremental cache vs upstream full recompute
    L, P = 480, 32
    df, xts, yts = window(L, P)
    xn, xs, ys, mean, std = kv.prepare_inputs(df, xts, yts, pr.clip)
    T = lambda a: torch.from_numpy(a[None])
    torch.manual_seed(0)
    s1, s2 = kv.generate_tokens(pr.tokenizer, pr.model, torch.clip(T(xn), -5, 5), T(xs), T(ys), P, 2, 512)
    full_stamp = torch.cat([T(xs), T(ys)], dim=1)
    with torch.no_grad():
        logits_up, ctx_up = pr.model.decode_s1(s1[:1], s2[:1], full_stamp)
        cache = kv._Cache(len(pr.model.transformer))
        ctx = [kv._forward_tokens(pr.model, cache, s1[:1, :L], s2[:1, :L], T(xs), 0, True)]
        for i in range(P):
            ctx.append(kv._forward_tokens(pr.model, cache, s1[:1, L+i:L+i+1], s2[:1, L+i:L+i+1], full_stamp[:, L+i:L+i+1], L+i, False))
        ctx = torch.cat(ctx, dim=1)
        lg = pr.model.head(ctx)
    print(f"[{name}] context max|diff| {float((ctx-ctx_up).abs().max()):.2e} (scale {float(ctx_up.abs().max()):.1f}); "
          f"s1 logits max|diff| {float((lg-logits_up).abs().max()):.2e}; argmax agree {float((lg.argmax(-1)==logits_up.argmax(-1)).float().mean()):.4f}")
    # (b) seeded end-to-end vs upstream predict (which averages sample paths)
    for P, N in [(8, 4), (32, 4)]:
        L = 512 - P
        df, xts, yts = window(L, P)
        torch.manual_seed(123)
        t0 = time.perf_counter(); up = pr.predict(df, xts, yts, pred_len=P, T=1.0, top_p=0.9, sample_count=N, verbose=False); t_up = time.perf_counter()-t0
        torch.manual_seed(123)
        t0 = time.perf_counter(); paths = kv.forecast_paths(pr, df, xts, yts, P, N, T=1.0, top_p=0.9); t_kv = time.perf_counter()-t0
        mine = paths.mean(axis=0)
        rel = np.abs(mine - up.to_numpy()) / np.abs(up.to_numpy()).clip(1e-9)
        print(f"[{name}] P={P} N={N}: mean-of-paths vs upstream predict max rel diff {rel[:, :4].max():.2e} (prices), "
              f"{rel.max():.2e} (all cols); upstream {t_up:.1f}s vs kv {t_kv:.1f}s")
