<p align="center">
  <img src="src/candlecast/assets/logo.png" width="96" alt="CandleCast logo">
</p>

<h1 align="center">CandleCast</h1>

<p align="center">
  Probabilistic price forecasts on your Windows desktop, powered by the open-source
  <a href="https://github.com/shiyu-coder/Kronos">Kronos</a> foundation model - with a built-in
  <b>test window</b> that shows how the same forecast did on prices that already happened.
</p>

<p align="center">
  <img src="docs/screenshots/candlecast-dark.png" alt="CandleCast in dark mode" width="900">
</p>

> **For fun and study - not financial advice.** In our own backtests (see [Accuracy](#accuracy)) Kronos
> did not beat a simple "price stays where it is" guess. CandleCast is built to make that visible, not hide it.

## What it does

- Type any **US stock or ETF** (e.g. `AAPL, NVDA, SPY`), crypto (`BTCUSD`, `ETHUSD`), gold/silver/oil
  (`XAUUSD`, `XAGUSD`, `WTI`, `BRENT`) or major FX pairs (`EURUSD`, `USDJPY`, ...).
- Pick a **timeframe** - 1 hour, 4 hours, 1 day or 1 week - and how many **bars** to forecast (4-128).
- CandleCast samples 50-200 possible future price paths with Kronos and draws:
  - the **forecast** (blue): median path and a calibrated 40-80% band of the next N bars;
  - the **test window** (orange): the same forecast made N bars ago, using only data up to then,
    drawn against what actually happened - scored against a "no change" guess.
- Switch the band level, light/dark mode, and **save any chart as a JPG**.
- Everything runs **locally on your CPU** - no account with us, no cloud service. Prices are downloaded
  from Alpaca (your own free keys) and Yahoo Finance and kept in a local database.

<p align="center">
  <img src="docs/screenshots/candlecast-light.png" alt="CandleCast in light mode" width="760">
</p>

## Install (Windows 10/11, 64-bit)

1. Download **`CandleCast-Setup-<version>.exe`** from the [latest release](../../releases/latest)
   (or the portable zip, if you prefer not to install anything).
2. Run it. It installs for your user only - no admin rights needed. Windows SmartScreen may warn that the
   app is from an unknown publisher (it is not code-signed); choose *More info -> Run anyway*.
3. Start **CandleCast** from the Start menu.

The first forecast downloads the Kronos model (~410 MB) once; after that it loads in 10-30 seconds.
A forecast takes roughly 30-90 seconds per ticker on a typical laptop CPU.

### Free Alpaca keys (for US stocks and ETFs)

US stock prices come from [Alpaca](https://alpaca.markets/). Create a free account, generate a key pair
(paper-trading keys are fine), and paste them into **Settings** in CandleCast. Crypto, metals, oil and FX
work without any keys. Keys are stored only on your computer, in `%LOCALAPPDATA%\CandleCast\.env`.

## How the forecast works

| Setting | What it means |
|---|---|
| **Timeframe** | Bar size: 1H, 4H (built from 1H bars), 1D or 1W |
| **Bars** | N bars forecast ahead - and the size of the test window |
| **History** | Bars of past prices Kronos sees: 128 by default, adjustable 64-448 (capped at 512 - N). It does not change with N, so a longer forecast extends a shorter one |
| **Paths** | How many futures Kronos samples; more = smoother bands, slower |
| **Band** | Calibrated 40-80% range, or Kronos's raw band |

Kronos's own 80% band turned out to contain only ~40% of real outcomes, so CandleCast widens it with
factors fitted on 640 past test forecasts ([`config/band_calibration.json`](config/band_calibration.json)).
The factors were fitted on 1-hour x 32-bar forecasts; for other combinations the app marks them
*approximate* and the test window shows how well they held.

Why history is fixed: Kronos generates bar by bar, so with the same history the first 16 bars of a
27-bar forecast are identical to a 16-bar forecast. Changing the *history* is what changes the path -
Kronos tends to pull forecasts toward the average of the window it sees, so a longer window can even flip
the direction. Try the History slider to see how sensitive a given forecast is. (Averaging several
history lengths did not improve accuracy in our backtests, so CandleCast does not do it.)

Data is cached per timeframe in `%LOCALAPPDATA%\CandleCast\data` and refreshed online each run; if the
download fails the local copy is used.

## Accuracy

The [`benchmarks/`](benchmarks/) folder holds the research record behind CandleCast: about 6,800 out-of-sample
test forecasts across hourly, 4-hour and daily bars, several history lengths and sampling settings, a ranking
test the way the Kronos paper evaluates it, a volatility comparison against GARCH/EWMA, and a
[pre-registered confirmation test](benchmarks/CONFIRMATION_q5_PLAN.md). Summary:

- No configuration beat the "no change" baseline on price error.
- Ranking stocks by expected return showed no skill once tested on new stocks.
- GARCH(1,1) forecast volatility better than Kronos in every setup.
- On daily/weekly bars Kronos tends to pull forecasts back toward the average of its input window,
  which shows up clearly in the test window.

Independent users report similar results ([#354](https://github.com/shiyu-coder/Kronos/issues/354),
[#355](https://github.com/shiyu-coder/Kronos/issues/355)). CandleCast is a tool to explore and learn from
a time-series foundation model - treat its charts as an illustration of uncertainty.

## Run from source / build the installer

Requires Python 3.11 on Windows.

```powershell
git clone https://github.com/<you>/candlecast.git
cd candlecast
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\candlecast-app            # the desktop app
.venv\Scripts\python -m pytest -m "not slow"
```

Build the Windows app (PyInstaller) and installer ([Inno Setup 6](https://jrsoftware.org/isinfo.php)):

```powershell
.venv\Scripts\python -m pip install pyinstaller
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
# -> dist\CandleCast\CandleCast.exe, dist\CandleCast-Setup-<version>.exe, dist\CandleCast-<version>-portable.zip
```

A command-line interface (`candlecast run / show / score`) and a nightly batch pipeline are also included
for research use - see `candlecast --help`.

## Credits and licence

- Model: [Kronos](https://github.com/shiyu-coder/Kronos) by Shi Yu et al. ([paper](https://arxiv.org/abs/2508.02739)),
  MIT licence; code vendored unmodified in `vendor/kronos`, weights downloaded from Hugging Face
  (`NeoQuasar/Kronos-base`, MIT).
- CandleCast itself: MIT licence (see [LICENSE](LICENSE)); third-party components in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Market data: your own Alpaca account and Yahoo Finance (via yfinance) - their terms apply.
