"""Load the vendored, unmodified Kronos package and its pretrained weights.

Upstream imports itself as a top-level package named ``model``, so ``vendor/kronos`` has to be
on ``sys.path``. Everything that touches Kronos goes through this module.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

from candlecast.settings import APP_DIR, DATA_DIR, FROZEN, ModelSettings, load_secret

VENDOR_DIR = APP_DIR / "vendor" / "kronos"
if FROZEN:  # keep downloaded model weights inside the app's own data folder
    os.environ.setdefault("HF_HOME", str(DATA_DIR / "models"))


def _import_kronos():
    vendor = str(VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: PLC0415

    return Kronos, KronosTokenizer, KronosPredictor


def configure_torch_threads(default: int = 0) -> int:
    """CANDLECAST_TORCH_THREADS (env/.env) overrides ``default`` (settings); 0 = torch's choice."""
    import torch  # noqa: PLC0415

    threads = int(load_secret("CANDLECAST_TORCH_THREADS") or default or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    return torch.get_num_threads()


@lru_cache(maxsize=2)
def load_predictor(repo: str, tokenizer_repo: str, max_context: int):
    """Load tokenizer + model once per process (weights come from the Hugging Face cache)."""
    Kronos, KronosTokenizer, KronosPredictor = _import_kronos()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_repo)
    model = Kronos.from_pretrained(repo)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=max_context)


def predictor_for(model: ModelSettings):
    return load_predictor(model.repo, model.tokenizer_repo, model.max_context)
