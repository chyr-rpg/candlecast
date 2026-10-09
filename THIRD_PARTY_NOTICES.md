# Third-party notices

CandleCast is MIT-licensed (see `LICENSE`). It includes or downloads the following third-party work.

## Kronos (model code, vendored unmodified in `vendor/kronos`)
- Source: https://github.com/shiyu-coder/Kronos (commit 67b630e)
- License: MIT, Copyright (c) 2025 ShiYu - full text in `vendor/kronos/LICENSE`.

## Kronos model weights (downloaded on first use, not redistributed)
- `NeoQuasar/Kronos-base`, `NeoQuasar/Kronos-Tokenizer-base` on Hugging Face - MIT licence per the model cards.

## Python libraries bundled in the Windows build
| Library | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| NumPy, pandas, pyarrow | BSD-3-Clause / BSD-3-Clause / Apache-2.0 |
| Matplotlib | Matplotlib licence (PSF-based) |
| CustomTkinter | MIT |
| Pillow | MIT-CMU (HPND) |
| einops | MIT |
| huggingface_hub, safetensors | Apache-2.0 |
| alpaca-py | Apache-2.0 |
| yfinance | Apache-2.0 |
| exchange_calendars | Apache-2.0 |
| python-dotenv | BSD-3-Clause |
| tqdm | MPL-2.0 / MIT |
| Python and Tcl/Tk (bundled runtime) | PSF licence / Tcl/Tk licence (BSD-style) |

## Market data
Prices come from the user's own Alpaca account (Alpaca's terms apply) and from Yahoo Finance via
yfinance (Yahoo's terms; intended for personal use). CandleCast does not redistribute market data.
