"""Load the vendored, unmodified Kronos package and its pretrained weights.

Upstream imports itself as a top-level package named ``model``, so ``vendor/kronos`` has to be
on ``sys.path``. Everything that touches Kronos goes through this module.

Weights: downloaded once from Hugging Face with progress reported through logging (the desktop app shows
it), then always loaded from the local folder - no network check on later starts.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

from candlecast.settings import APP_DIR, DATA_DIR, FROZEN, ModelSettings, load_secret

log = logging.getLogger(__name__)

VENDOR_DIR = APP_DIR / "vendor" / "kronos"
if FROZEN:  # keep downloaded model weights inside the app's own data folder
    os.environ.setdefault("HF_HOME", str(DATA_DIR / "models"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def ensure_console_streams() -> None:
    """A windowed program (pythonw / the packaged exe) has sys.stdout/sys.stderr = None. Libraries that
    print progress bars (tqdm in huggingface_hub) then crash with "'NoneType' object has no attribute
    'write'". Point them at the null device instead."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


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


def _progress_class(label: str):
    """A tqdm subclass that logs download progress every ~10% instead of drawing to a console."""
    from tqdm.auto import tqdm  # noqa: PLC0415

    class LogProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - never draws
            super().__init__(*args, **kwargs)
            self._last_logged = -1

        def update(self, n=1):
            out = super().update(n)
            if self.total and self.unit == "B":
                step = int(10 * self.n / self.total)
                if step != self._last_logged:
                    self._last_logged = step
                    log.info("Downloading %s: %.0f / %.0f MB (%d%%)", label, self.n / 1e6, self.total / 1e6,
                             100 * self.n / self.total)
            return out

    return LogProgress


MODELS_DIR = DATA_DIR / "models"
_COMPLETE = ".download-complete"


def _is_complete(folder) -> bool:
    """A usable model folder has its config and weights (a cancelled or crashed download may leave only
    some files behind)."""
    from pathlib import Path  # noqa: PLC0415

    folder = Path(folder)
    return (folder / "config.json").is_file() and any(folder.glob("*.safetensors"))


def local_model_dir(repo: str) -> str:
    """Folder with the repo's files, loaded without any network check once present.

    Downloads go to plain files in <data>/models/<owner--name> (``local_dir`` mode). The default Hugging
    Face cache layout uses symlinks, which a normal Windows account may not create (WinError 1314), so it
    is only used when a complete copy is already there (e.g. from a developer setup)."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    ensure_console_streams()
    target = MODELS_DIR / repo.replace("/", "--")
    if (target / _COMPLETE).exists() and _is_complete(target):
        return str(target)
    try:
        cached = snapshot_download(repo, local_files_only=True)
        if _is_complete(cached):
            return cached
    except Exception:  # not in the standard cache: download into our own folder
        pass
    log.info("Downloading %s from Hugging Face (one time) ...", repo)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, local_dir=str(target), tqdm_class=_progress_class(repo.split("/")[-1]))
    if not _is_complete(target):
        raise RuntimeError(f"Download of {repo} is incomplete (no config.json / weights) - check the "
                           "internet connection and try again.")
    (target / _COMPLETE).write_text(repo, encoding="utf-8")
    log.info("Downloaded %s to %s", repo, target)
    return str(target)


@lru_cache(maxsize=2)
def load_predictor(repo: str, tokenizer_repo: str, max_context: int):
    """Load tokenizer + model once per process."""
    Kronos, KronosTokenizer, KronosPredictor = _import_kronos()
    tokenizer = KronosTokenizer.from_pretrained(local_model_dir(tokenizer_repo))
    model = Kronos.from_pretrained(local_model_dir(repo))
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=max_context)


def predictor_for(model: ModelSettings):
    return load_predictor(model.repo, model.tokenizer_repo, model.max_context)
