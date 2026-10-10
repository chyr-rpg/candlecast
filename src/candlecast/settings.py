"""Project settings (config/settings.toml) and secrets (.env).

Two folders:
* APP_DIR - the program's own files (bundled config defaults, vendored Kronos). In the installed app
  this is PyInstaller's bundle folder; from source it is the repository root.
* DATA_DIR - everything the app writes (bar databases, forecasts, its editable config, .env with API
  keys, downloaded model weights). CANDLECAST_HOME overrides it; the installed app uses
  %LOCALAPPDATA%\\CandleCast; from source it is the repository root (unchanged behaviour).
"""

from __future__ import annotations

import os
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


def _data_dir() -> Path:
    if os.environ.get("CANDLECAST_HOME"):
        return Path(os.environ["CANDLECAST_HOME"])
    if FROZEN:
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "CandleCast"
    return APP_DIR


DATA_DIR = _data_dir()
PROJECT_ROOT = DATA_DIR          # kept for older call sites: the writable root
ALLOWED_MODELS = ("small", "base")
CUSTOM_MODEL = "custom"         # desktop app: any Kronos checkpoint (Hugging Face id or local folder)
CONFIG_FILES = ("settings.toml", "symbol_map.csv", "band_calibration.json", "watchlist.example.csv")


def ensure_data_dir(data_dir: Path | None = None) -> Path:
    """Create the data folder and copy any missing default config files into it (never overwrites)."""
    data_dir = Path(data_dir or DATA_DIR)
    (data_dir / "config").mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        src, dst = APP_DIR / "config" / name, data_dir / "config" / name
        if src.exists() and not dst.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    return data_dir


@dataclass(frozen=True)
class ModelSettings:
    name: str
    repo: str
    tokenizer_repo: str
    max_context: int
    sample_paths: int
    torch_threads: int
    temperature: float
    top_p: float
    history_bars: int = 0      # 0 = max_context - horizon


@dataclass(frozen=True)
class DataSettings:
    symbol_map_csv: Path
    history_days: int
    overlap_hours: int
    revision_rtol: float
    min_bars: int
    stale_days: int


@dataclass(frozen=True)
class Settings:
    root: Path
    watchlist_csv: Path
    default_tickers: tuple[str, ...]
    model: ModelSettings
    data: DataSettings
    horizons: dict[str, int]
    keep_paths_days: int
    chart_history_bars: int
    bars_dir: Path
    forecasts_dir: Path
    index_db: Path
    run_log_dir: Path
    model_repos: dict[str, str] = field(default_factory=dict)
    test_forecast: bool = True  # also forecast the last horizon that already happened (chart + test_forecasts)
    chart_band_level: int = 80  # calibrated band drawn on charts (config/band_calibration.json); 0 = raw bands
    offline: bool = False       # True: never download prices or models; use only what is on this PC


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_settings(root: Path | None = None, model_name: str | None = None) -> Settings:
    """Load settings; ``model_name`` overrides ``[model].name`` (e.g. for benchmarks)."""
    root = Path(root) if root else ensure_data_dir()
    with open(root / "config" / "settings.toml", "rb") as fh:
        raw = tomllib.load(fh)

    model_raw = raw["model"]
    repos = dict(model_raw["repos"])
    name = model_name or model_raw["name"]
    if name not in ALLOWED_MODELS:
        raise ValueError(f"model must be one of {ALLOWED_MODELS}, got {name!r}")

    model = ModelSettings(
        name=name,
        repo=repos[name],
        tokenizer_repo=model_raw["tokenizer_repo"],
        max_context=int(model_raw["max_context"]),
        sample_paths=int(model_raw["sample_paths"]),
        torch_threads=int(model_raw.get("torch_threads", 0)),
        temperature=float(model_raw["temperature"]),
        top_p=float(model_raw["top_p"]),
        history_bars=int(model_raw.get("history_bars", 0)),
    )
    data_raw = raw["data"]
    data = DataSettings(
        symbol_map_csv=_resolve(root, data_raw["symbol_map_csv"]),
        history_days=int(data_raw["history_days"]),
        overlap_hours=int(data_raw["overlap_hours"]),
        revision_rtol=float(data_raw["revision_rtol"]),
        min_bars=int(data_raw["min_bars"]),
        stale_days=int(data_raw["stale_days"]),
    )
    paths = raw["paths"]
    return Settings(
        root=root,
        watchlist_csv=_resolve(root, raw["universe"]["watchlist_csv"]),
        default_tickers=tuple(t.strip().upper() for t in raw["universe"]["default_tickers"]),
        model=model,
        data=data,
        horizons={k: int(v) for k, v in raw["horizon"].items()},
        keep_paths_days=int(raw["output"]["keep_paths_days"]),
        chart_history_bars=int(raw["output"]["chart_history_bars"]),
        bars_dir=_resolve(root, paths["bars_dir"]),
        forecasts_dir=_resolve(root, paths["forecasts_dir"]),
        index_db=_resolve(root, paths["index_db"]),
        run_log_dir=_resolve(root, paths["run_log_dir"]),
        model_repos=repos,
        test_forecast=bool(raw.get("test_forecast", {}).get("enabled", True)),
        chart_band_level=int(raw["output"].get("chart_band_level", 80)),
    )


PREFS_FILE = "app_prefs.json"


def load_prefs(root: Path | None = None) -> dict:
    """Desktop-app preferences (e.g. {"model": "small"}) from the data folder; {} if none or unreadable."""
    import json  # noqa: PLC0415

    path = (Path(root) if root else DATA_DIR) / PREFS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_prefs(updates: dict, root: Path | None = None) -> dict:
    import json  # noqa: PLC0415

    root = Path(root) if root else DATA_DIR
    prefs = {**load_prefs(root), **updates}
    root.mkdir(parents=True, exist_ok=True)
    (root / PREFS_FILE).write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    return prefs


def save_secrets(values: dict[str, str], root: Path | None = None) -> Path:
    """Write/replace keys in the data folder's .env (other lines are kept). Returns the .env path."""
    env_path = (Path(root) if root else DATA_DIR) / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    keep = [ln for ln in lines if ln.split("=", 1)[0].strip() not in values]
    keep += [f"{k}={v.strip()}" for k, v in values.items()]
    env_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return env_path


def load_secret(name: str, root: Path | None = None) -> str | None:
    """Environment variable first, then the project's .env; empty values count as unset."""
    value = os.environ.get(name)
    if not value:
        env_path = (Path(root) if root else PROJECT_ROOT) / ".env"
        if env_path.exists():
            value = dotenv_values(env_path).get(name)
    return value.strip() if value and value.strip() else None
