"""KV-cached generation must reproduce upstream Kronos exactly (same tokens for the same seed)."""

import numpy as np
import pandas as pd
import pytest
import torch

from candlecast import kv_inference as kv
from candlecast.cli import synthetic_bars
from candlecast.kronos_loader import predictor_for
from candlecast.settings import load_settings


def _window(history: int, pred_len: int):
    bars = synthetic_bars(history + pred_len, seed=7)
    return bars.iloc[:history][kv.FEATURES], bars["timestamp"].iloc[:history], bars["timestamp"].iloc[history:]


def test_refuses_inputs_that_would_slide_the_window():
    x = torch.zeros(1, 500, 6)
    with pytest.raises(ValueError, match="exceeds max_context"):
        kv.generate_tokens(None, None, x, torch.zeros(1, 500, 5), torch.zeros(1, 32, 5), 32, 2, 512)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["small", "base"])
def test_mean_of_paths_matches_upstream_predict(name):
    model = load_settings(model_name=name).model
    predictor = predictor_for(model)
    pred_len, n_paths = 12, 3
    df, x_ts, y_ts = _window(512 - pred_len, pred_len)

    torch.manual_seed(2024)
    upstream = predictor.predict(df, x_ts, y_ts, pred_len=pred_len, T=1.0, top_p=0.9,
                                 sample_count=n_paths, verbose=False)
    torch.manual_seed(2024)
    paths = kv.forecast_paths(predictor, df, x_ts, y_ts, pred_len, n_paths, T=1.0, top_p=0.9)

    assert paths.shape == (n_paths, pred_len, 6)
    np.testing.assert_allclose(paths.mean(axis=0)[:, :4], upstream.to_numpy()[:, :4], rtol=1e-5)


@pytest.mark.slow
def test_incremental_hidden_states_match_full_recompute():
    predictor = predictor_for(load_settings(model_name="small").model)
    history, pred_len = 200, 6
    df, x_ts, y_ts = _window(history, pred_len)
    xn, xs, ys, _, _ = kv.prepare_inputs(df, x_ts, y_ts, predictor.clip)
    t = lambda a: torch.from_numpy(a[None])  # noqa: E731
    torch.manual_seed(0)
    s1, s2 = kv.generate_tokens(predictor.tokenizer, predictor.model, t(xn), t(xs), t(ys), pred_len, 1, 512)
    stamps = torch.cat([t(xs), t(ys)], dim=1)
    with torch.no_grad():
        logits_full, _ = predictor.model.decode_s1(s1, s2, stamps)
        cache = kv._Cache(len(predictor.model.transformer))
        parts = [kv._forward_tokens(predictor.model, cache, s1[:, :history], s2[:, :history], t(xs), 0, True)]
        for i in range(history, history + pred_len):
            parts.append(kv._forward_tokens(predictor.model, cache, s1[:, i:i + 1], s2[:, i:i + 1],
                                            stamps[:, i:i + 1], i, False))
        logits_inc = predictor.model.head(torch.cat(parts, dim=1))
    assert float((logits_inc - logits_full).abs().max()) < 1e-3
    assert bool((logits_inc.argmax(-1) == logits_full.argmax(-1)).all())


@pytest.mark.slow
def test_paths_differ_and_stay_near_last_price():
    predictor = predictor_for(load_settings(model_name="small").model)
    df, x_ts, y_ts = _window(480, 16)
    torch.manual_seed(1)
    paths = kv.forecast_paths(predictor, df, x_ts, y_ts, 16, 8)
    closes = paths[:, :, 3]
    assert np.isfinite(paths).all()
    assert len({tuple(np.round(c, 6)) for c in closes}) == 8  # sampled paths are not identical
    last = float(df["close"].iloc[-1])
    assert (np.abs(closes / last - 1) < 0.5).all()
