"""KV-cached Kronos generation that keeps every sampled path (no averaging).

Upstream ``auto_regressive_inference`` re-runs the whole context through the transformer for
every generated bar and averages the sampled paths. This module reuses the same pretrained
modules and weights but:

* runs the shared history once (batch 1) and copies the per-layer key/value cache to N paths;
* then feeds only the newest token per step, attending over the cache;
* returns all N paths so callers can compute quantile bands.

Exactness condition: ``history_len + pred_len <= max_context``. Upstream slides its window once a
sequence exceeds ``max_context``, which re-encodes every earlier token with less context; a cache
can't reproduce that, so ``generate_paths`` refuses such inputs instead of approximating them.
Within the limit the computation is the same as upstream's (causal self-attention, rotary
positions counted from the window start, and the s2 cross-attention's degenerate rotary — its
query has length 1, so upstream rotates both q and k by position 0, i.e. not at all), and with the
same torch RNG state it consumes random numbers in the same order (s1 then s2 per step).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from candlecast.kronos_loader import _import_kronos

PRICE_COLS = ["open", "high", "low", "close"]
FEATURES = PRICE_COLS + ["volume", "amount"]


def _rope_tables(rotary, positions: torch.Tensor):
    freqs = torch.einsum("i,j->ij", positions.type_as(rotary.inv_freq), rotary.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _split_heads(t, n_heads):
    b, s, d = t.shape
    return t.view(b, s, n_heads, d // n_heads).transpose(1, 2)


def _merge_heads(t):
    b, h, s, hd = t.shape
    return t.transpose(1, 2).contiguous().view(b, s, h * hd)


class _Cache:
    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers
        self.cross_k: torch.Tensor | None = None  # s2 cross-attention keys over the final context
        self.cross_v: torch.Tensor | None = None

    def expand(self, n: int) -> "_Cache":
        out = _Cache(len(self.k))
        out.k = [t.expand(n, *t.shape[1:]).contiguous() for t in self.k]
        out.v = [t.expand(n, *t.shape[1:]).contiguous() for t in self.v]
        out.cross_k = self.cross_k.expand(n, *self.cross_k.shape[1:]).contiguous()
        out.cross_v = self.cross_v.expand(n, *self.cross_v.shape[1:]).contiguous()
        return out


def _forward_tokens(model, cache: _Cache, s1, s2, stamp, start_pos: int, causal: bool):
    """Run new tokens (positions start_pos..) through the transformer, appending to the cache.

    Returns the final normalized hidden states for the new tokens (upstream's ``context``).
    """
    x = model.embedding([s1, s2])
    x = x + model.time_emb(stamp)
    positions = torch.arange(start_pos, start_pos + s1.shape[1], device=x.device)
    for i, layer in enumerate(model.transformer):
        attn = layer.self_attn
        h = layer.norm1(x)
        q = _split_heads(attn.q_proj(h), attn.n_heads)
        k = _split_heads(attn.k_proj(h), attn.n_heads)
        v = _split_heads(attn.v_proj(h), attn.n_heads)
        cos, sin = _rope_tables(attn.rotary, positions)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if cache.k[i] is not None:
            k = torch.cat([cache.k[i], k], dim=2)
            v = torch.cat([cache.v[i], v], dim=2)
        cache.k[i], cache.v[i] = k, v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + attn.out_proj(_merge_heads(out))
        x = x + layer.ffn(layer.norm2(x))
    context = model.norm(x)

    cross = model.dep_layer.cross_attn
    ck = _split_heads(cross.k_proj(context), cross.n_heads)
    cv = _split_heads(cross.v_proj(context), cross.n_heads)
    cache.cross_k = ck if cache.cross_k is None else torch.cat([cache.cross_k, ck], dim=2)
    cache.cross_v = cv if cache.cross_v is None else torch.cat([cache.cross_v, cv], dim=2)
    return context


def _s2_logits(model, cache: _Cache, context_last, s1_sample):
    """Upstream decode_s2 for the last position: non-causal cross-attention, no effective rotary."""
    cross = model.dep_layer.cross_attn
    sibling = model.embedding.emb_s1(s1_sample)  # [B, 1, d]
    q = _split_heads(cross.q_proj(sibling), cross.n_heads)
    out = F.scaled_dot_product_attention(q, cache.cross_k, cache.cross_v, is_causal=False)
    x2 = model.dep_layer.norm(context_last + cross.out_proj(_merge_heads(out)))
    return model.head.cond_forward(x2)[:, -1, :]


@torch.no_grad()
def generate_tokens(tokenizer, model, x_norm: torch.Tensor, x_stamp: torch.Tensor, y_stamp: torch.Tensor,
                    pred_len: int, n_paths: int, max_context: int, T: float = 1.0, top_k: int = 0,
                    top_p: float = 0.9):
    """Token-level generation for ONE series. ``x_norm``: [1, L, 6]; stamps: [1, L, 5] / [1, pred_len, 5].

    Returns (s1_tokens, s2_tokens), each [n_paths, L + pred_len] (history + generated).
    """
    _, sample_from_logits = _upstream()
    L = x_norm.shape[1]
    if L + pred_len > max_context:
        raise ValueError(f"history {L} + pred_len {pred_len} exceeds max_context {max_context}; "
                         f"use at most {max_context - pred_len} history bars")
    s1_hist, s2_hist = tokenizer.encode(x_norm, half=True)

    cache = _Cache(len(model.transformer))
    context = _forward_tokens(model, cache, s1_hist, s2_hist, x_stamp, 0, causal=True)
    cache = cache.expand(n_paths)
    last = context[:, -1:, :].expand(n_paths, 1, context.shape[-1])

    gen_s1 = s1_hist.new_empty(n_paths, pred_len)
    gen_s2 = s2_hist.new_empty(n_paths, pred_len)
    y_stamp = y_stamp.expand(n_paths, *y_stamp.shape[1:])
    for i in range(pred_len):
        s1_logits = model.head(last)[:, -1, :]
        s1 = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        s2 = sample_from_logits(_s2_logits(model, cache, last, s1), temperature=T, top_k=top_k, top_p=top_p,
                                sample_logits=True)
        gen_s1[:, i] = s1.squeeze(-1)
        gen_s2[:, i] = s2.squeeze(-1)
        if i + 1 < pred_len:  # the final token is never fed back
            last = _forward_tokens(model, cache, s1, s2, y_stamp[:, i:i + 1, :], L + i, causal=False)
    full_s1 = torch.cat([s1_hist.expand(n_paths, L), gen_s1], dim=1)
    full_s2 = torch.cat([s2_hist.expand(n_paths, L), gen_s2], dim=1)
    return full_s1, full_s2


def _upstream():
    """(calc_time_stamps, sample_from_logits) from the vendored package."""
    _import_kronos()
    from model.kronos import calc_time_stamps, sample_from_logits  # noqa: PLC0415

    return calc_time_stamps, sample_from_logits


def prepare_inputs(df: pd.DataFrame, x_timestamp: pd.Series, y_timestamp: pd.Series, clip: float):
    """Same preprocessing as upstream KronosPredictor.predict (z-score per feature, clip)."""
    calc_time_stamps, _ = _upstream()
    df = df.copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0
        df["amount"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[PRICE_COLS].mean(axis=1)
    x = df[FEATURES].to_numpy(dtype=np.float32)
    if np.isnan(x).any():
        raise ValueError("input contains NaN in price/volume columns")
    x_stamp = calc_time_stamps(pd.Series(x_timestamp).reset_index(drop=True)).to_numpy(dtype=np.float32)
    y_stamp = calc_time_stamps(pd.Series(y_timestamp).reset_index(drop=True)).to_numpy(dtype=np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0)
    x_norm = np.clip((x - mean) / (std + 1e-5), -clip, clip)
    return x_norm, x_stamp, y_stamp, mean, std


@torch.no_grad()
def forecast_paths(predictor, df: pd.DataFrame, x_timestamp: pd.Series, y_timestamp: pd.Series, pred_len: int,
                   n_paths: int, T: float = 1.0, top_k: int = 0, top_p: float = 0.9) -> np.ndarray:
    """N sampled future paths for one series, in price units: array [n_paths, pred_len, 6]
    (columns ``FEATURES``)."""
    x_norm, x_stamp, y_stamp, mean, std = prepare_inputs(df, x_timestamp, y_timestamp, predictor.clip)
    to_t = lambda a: torch.from_numpy(a[np.newaxis]).to(predictor.device)  # noqa: E731
    x_t = torch.clip(to_t(x_norm), -predictor.clip, predictor.clip)
    s1, s2 = generate_tokens(predictor.tokenizer, predictor.model, x_t, to_t(x_stamp), to_t(y_stamp),
                             pred_len, n_paths, predictor.max_context, T=T, top_k=top_k, top_p=top_p)
    # Upstream decodes the trailing max_context tokens (here: the whole sequence) and keeps the tail.
    start = max(0, s1.shape[1] - predictor.max_context)
    z = predictor.tokenizer.decode([s1[:, start:].contiguous(), s2[:, start:].contiguous()], half=True)
    preds = z[:, -pred_len:, :].cpu().numpy()
    return preds * (std + 1e-5) + mean
