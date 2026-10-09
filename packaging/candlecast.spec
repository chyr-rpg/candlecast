# PyInstaller spec for CandleCast (one-folder windowed app). Build with packaging\build.ps1.
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent          # noqa: F821 - SPECPATH is provided by PyInstaller
ICON = str(ROOT / "src" / "candlecast" / "assets" / "candlecast.ico")

datas = [
    (str(ROOT / "src" / "candlecast" / "assets"), "candlecast/assets"),
    (str(ROOT / "config"), "config"),
    (str(ROOT / "vendor" / "kronos"), "vendor/kronos"),
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
