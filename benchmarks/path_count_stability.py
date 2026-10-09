import numpy as np, pandas as pd, torch
from candlecast.settings import load_settings
from candlecast.kronos_loader import predictor_for
from candlecast import kv_inference as kv
torch.set_num_threads(8)
pr = predictor_for(load_settings(model_name="base").model)
REF, SUBS = 400, 300
rng = np.random.default_rng(0)
rows = []
for sym, P in [("AAPL", 32), ("NVDA", 32), ("BTCUSD", 48), ("XAUUSD", 46)]:
    bars = pd.read_parquet(f"data/bars/1h/{sym}.parquet").iloc[-512:].reset_index(drop=True)
    ts = bars["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    L = 512 - P
    torch.manual_seed(0)
    paths = kv.forecast_paths(pr, bars.iloc[:L][kv.FEATURES], ts.iloc[:L], ts.iloc[L:], P, REF)
    stats = {"end_close": paths[:, -1, 3], "max_high": paths[:, :, 1].max(1), "min_low": paths[:, :, 2].min(1)}
    for stat, v in stats.items():
        width = np.quantile(v, 0.9) - np.quantile(v, 0.1)
        for N in [20, 30, 50, 100]:
            est = np.array([np.quantile(v[rng.choice(REF, N, replace=False)], [0.1, 0.5, 0.9]) for _ in range(SUBS)])
            se = est.std(axis=0) / np.sqrt(1 - N / REF)  # finite-population correction
            rows.append({"symbol": sym, "stat": stat, "N": N, "p10": se[0] / width, "p50": se[1] / width, "p90": se[2] / width})
    print(sym, "done", flush=True)
df = pd.DataFrame(rows)
print("\nStd error of quantile estimate as % of the p10-p90 band width (median over 4 symbols):")
print((df.groupby(["stat", "N"])[["p10", "p50", "p90"]].median() * 100).round(1).unstack("stat").to_string())
print("\nWorst symbol:")
print((df.groupby(["N"])[["p10", "p50", "p90"]].max() * 100).round(1).to_string())
