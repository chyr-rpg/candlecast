# PyInstaller spec for CandleCast (one-folder windowed app). Build with packaging\build.ps1.
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent          # noqa: F821 - SPECPATH is provided by PyInstaller
ICON = str(ROOT / "src" / "candlecast" / "assets" / "candlecast.ico")



def files(folder, dest, skip=()):
    """Ship a folder file by file, never __pycache__ (it embeds the build machine's paths) or anything in skip."""
    return [(str(p), str(Path(dest) / p.parent.relative_to(folder))) for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name not in skip]


datas = [
    (str(ROOT / "src" / "candlecast" / "assets"), "candlecast/assets"),
    # config/watchlist.csv is the owner's personal list: never shipped (watchlist.example.csv is).
    *files(ROOT / "config", "config", skip={"watchlist.csv"}),
    *files(ROOT / "vendor" / "kronos", "vendor/kronos"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
]
datas += collect_data_files("customtkinter")
datas += collect_data_files("exchange_calendars")
# huggingface_hub (and others) detect optional packages by their installed metadata, which PyInstaller
# drops unless copied: without it Kronos.from_pretrained fails with "name 'safetensors' is not defined"
for dist in ("safetensors", "torch", "huggingface_hub", "tqdm", "numpy", "packaging", "filelock", "requests",
             "pyyaml", "regex", "pandas", "pyarrow", "einops", "alpaca-py", "yfinance", "exchange_calendars"):
    try:
        datas += copy_metadata(dist)
    except Exception:  # not installed in this environment
        pass

hiddenimports = ["model", "model.kronos", "model.module", "einops", "safetensors", "safetensors.torch",
                 "huggingface_hub", "pyarrow", "yfinance"]
hiddenimports += collect_submodules("alpaca")
hiddenimports += collect_submodules("exchange_calendars")
hiddenimports += collect_submodules("safetensors")

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src"), str(ROOT / "vendor" / "kronos")],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "IPython", "notebook", "jupyter", "torch.utils.tensorboard", "tensorboard"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz, a.scripts, [], exclude_binaries=True, name="CandleCast", icon=ICON,
    console=False, upx=False,
    version=str(ROOT / "packaging" / "version_info.txt"),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="CandleCast")  # noqa: F821
