"""CandleCast desktop app (CustomTkinter).

Same engine as the classic panel (candlecast.panel's runner, timeframes, custom forecasts) with a modern
interface: sidebar with results, light/dark themes, a settings dialog for the user's own Alpaca keys,
summary cards, and themed charts. Start: ``candlecast-app`` / ``python -m candlecast.app`` / CandleCast.exe.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import webbrowser
from pathlib import Path

import customtkinter as ctk
import pandas as pd
from PIL import Image

from candlecast import __version__, calibration, custom, timeframes
from candlecast.calendars import NY
from candlecast.charts import time_format
from candlecast.panel import (
    BAND_CHOICES,
    CALIBRATED_SPEC,
    NOISY_LOGGERS,
    ForecastRunner,
    Result,
    band_level_from_choice,
    parse_tickers,
    requirement_text,
    resolve_tickers,
    save_as_jpg,
)
from candlecast.settings import DATA_DIR, load_secret, load_settings, save_secrets

ASSETS = Path(__file__).resolve().parent / "assets"
ACCENT = ("#2a78d6", "#3987e5")
GOOD, BAD = ("#0a7d32", "#3fbf6a"), ("#c2410c", "#f08a5d")
MUTED = ("#52514e", "#a9a89f")
CARD = ("#ffffff", "#2e2e2d")
TIMEFRAME_LABELS = ("1H", "4H", "1D", "1W")
KRONOS_URL = "https://github.com/shiyu-coder/Kronos"
ALPACA_URL = "https://alpaca.markets/"


LOG_FILE = DATA_DIR / "logs" / "app.log"


def _setup_file_log() -> None:
    """Everything the app logs also goes to %LOCALAPPDATA%\CandleCast\logs\app.log (for bug reports)."""
    from logging.handlers import RotatingFileHandler  # noqa: PLC0415

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(fh)
    root.setLevel(logging.INFO)


def _append_log(line: str) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


ICON = ASSETS / "candlecast.ico"
APP_USER_MODEL_ID = "CandleCast.Desktop.App"


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity, so Windows shows the CandleCast icon instead of grouping
    it under python.exe / pythonw.exe (matters when run from source; the packaged exe has the icon built in)."""
    try:
        import ctypes  # noqa: PLC0415

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def _apply_icon(window) -> None:
    """CustomTkinter puts its own icon on windows ~200 ms after creation; set ours after that."""
    def _set():
        try:
            window.iconbitmap(str(ICON))
        except Exception:
            pass
    _set()
    window.after(250, _set)


def has_alpaca_keys() -> bool:
    return bool(load_secret("ALPACA_API_KEY_ID") and load_secret("ALPACA_API_SECRET_KEY"))


class _QueueLog(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__(logging.INFO)
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.q.put(("log", self.format(record)))


# ------------------------------------------------------------------ dialogs

class SettingsDialog(ctk.CTkToplevel):
    """Alpaca keys (stored in the user's own data folder .env), connection test, data folder."""

    def __init__(self, app: "CandleCastApp"):
        super().__init__(app)
        self.app = app
        self.title("CandleCast settings")
        self.geometry("560x420")
        self.resizable(False, False)
        self.transient(app)
        self.after(150, self.grab_set)
        _apply_icon(self)
        pad = {"padx": 22, "pady": (6, 0)}
        ctk.CTkLabel(self, text="Market data keys", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=22, pady=(20, 2))
        ctk.CTkLabel(self, text=("US stocks and ETFs come from Alpaca. A free account gives you a key pair "
                                 "(paper trading keys are fine). Crypto, gold, oil and FX work without keys."),
                     wraplength=510, justify="left", text_color=MUTED).pack(anchor="w", **pad)
        link = ctk.CTkLabel(self, text="Get free keys at alpaca.markets", text_color=ACCENT, cursor="hand2")
        link.pack(anchor="w", **pad)
        link.bind("<Button-1>", lambda _e: webbrowser.open(ALPACA_URL))
        self.key = ctk.CTkEntry(self, placeholder_text="API key ID", width=510)
        self.key.pack(**pad)
        self.secret = ctk.CTkEntry(self, placeholder_text="API secret key", show="•", width=510)
        self.secret.pack(**pad)
        if load_secret("ALPACA_API_KEY_ID"):
            self.key.insert(0, load_secret("ALPACA_API_KEY_ID"))
        if load_secret("ALPACA_API_SECRET_KEY"):
            self.secret.insert(0, load_secret("ALPACA_API_SECRET_KEY"))
        self.status = ctk.CTkLabel(self, text="", text_color=MUTED, wraplength=510, justify="left")
        self.status.pack(anchor="w", padx=22, pady=(10, 0))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=22, pady=(14, 0))
        ctk.CTkButton(row, text="Test connection", width=150, fg_color="transparent", border_width=1,
                      text_color=("#0b0b0b", "#ffffff"), command=self.test).pack(side="left")
        ctk.CTkButton(row, text="Save", width=110, fg_color=ACCENT, command=self.save).pack(side="right")
        folder = ctk.CTkLabel(self, text=f"Data folder: {DATA_DIR}", text_color=MUTED, cursor="hand2")
        folder.pack(anchor="w", padx=22, pady=(22, 0))
        folder.bind("<Button-1>", lambda _e: os.startfile(DATA_DIR))  # noqa: S606

    def _apply(self):
        os.environ["ALPACA_API_KEY_ID"] = self.key.get().strip()
        os.environ["ALPACA_API_SECRET_KEY"] = self.secret.get().strip()

    def test(self):
        self._apply()
        self.status.configure(text="Testing ...", text_color=MUTED)
        self.update_idletasks()
        try:
            from candlecast.sources import AlpacaStockSource  # noqa: PLC0415

            now = pd.Timestamp.now(tz="UTC")
            df = AlpacaStockSource().fetch("SPY", (now - pd.Timedelta(days=10)).to_pydatetime(),
                                           (now - pd.Timedelta(minutes=20)).to_pydatetime(), timeframe="1d")
            self.status.configure(text=f"Connected - received {len(df)} SPY daily bars.", text_color=GOOD)
        except Exception as exc:
            self.status.configure(text=f"Connection failed: {type(exc).__name__}: {exc}"[:300], text_color=BAD)

    def save(self):
        self._apply()
        path = save_secrets({"ALPACA_API_KEY_ID": self.key.get(), "ALPACA_API_SECRET_KEY": self.secret.get()})
        self.status.configure(text=f"Saved to {path}", text_color=GOOD)
        self.app.refresh_key_banner()
        self.after(700, self.destroy)


class AboutDialog(ctk.CTkToplevel):
    def __init__(self, app: "CandleCastApp"):
        super().__init__(app)
        self.title("About CandleCast")
        self.geometry("560x440")
        self.resizable(False, False)
        self.transient(app)
        self.after(150, self.grab_set)
        _apply_icon(self)
        logo = ctk.CTkImage(Image.open(ASSETS / "logo.png"), size=(72, 72))
        ctk.CTkLabel(self, image=logo, text="").pack(pady=(22, 4))
        ctk.CTkLabel(self, text=f"CandleCast {__version__}", font=ctk.CTkFont(size=20, weight="bold")).pack()
        text = ("Probabilistic price forecasts on your desktop, powered by the open-source Kronos foundation "
                "model (MIT licence). Every chart also shows a test forecast made N bars ago, so you can see how "
                "the model did on prices that already happened.\n\n"
                "Not financial advice. In our own backtests (see the project's benchmarks folder) Kronos did not "
                "beat a simple 'no change' guess on price direction; use the bands as an illustration of "
                "uncertainty, not as trading signals.\n\nFree and open source under the MIT licence.")
        ctk.CTkLabel(self, text=text, wraplength=500, justify="left", text_color=MUTED).pack(padx=26, pady=12)
        link = ctk.CTkLabel(self, text="Kronos on GitHub", text_color=ACCENT, cursor="hand2")
        link.pack()
        link.bind("<Button-1>", lambda _e: webbrowser.open(KRONOS_URL))


# ------------------------------------------------------------------ main window

class CandleCastApp(ctk.CTk):
    def __init__(self, settings=None, runner: ForecastRunner | None = None, tickers: list[str] | None = None,
                 autorun: bool = False, timeframe: str | None = None, bars: int | None = None):
        super().__init__()
        self.settings = settings or load_settings()
        self.q: queue.Queue = queue.Queue()
        self.runner = runner or ForecastRunner(self.settings, log=lambda m: self.q.put(("log", m)))
        self.results: list[Result] = []
        self.selected: int | None = None
        self._shown = None
        self._chart_image = None
        self._resize_job = None
        self._file_log = False
        self.title("CandleCast")
        self.geometry("1280x800")
        self.minsize(1080, 700)
        self.after(0, self._maximize)
        _apply_icon(self)
        try:
            self.iconbitmap(default=str(ICON))      # default for every dialog window too
        except Exception:
            pass
        self._build()
        if tickers:
            self.tickers.delete(0, "end")
            self.tickers.insert(0, ", ".join(tickers))
        self.refresh_key_banner()
        if timeframe:
            self.tf.set(timeframe.upper())
            self.on_timeframe()
        if bars:
            self.bars.set(max(timeframes.MIN_PRED, min(timeframes.MAX_PRED, int(bars))))
            self.update_requirement()
        if autorun and tickers:
            self.after(400, self.on_run)
        self.after(120, self._poll)

    def _maximize(self):
        try:
            self.state("zoomed")   # Windows: open maximized so nothing is cut off at any display scaling
        except Exception:
            pass

    # ---------------- layout
    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(self, width=240, corner_radius=0)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_rowconfigure(3, weight=1)
        side.grid_propagate(False)
        brand = ctk.CTkFrame(side, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=18, pady=(20, 10))
        logo = ctk.CTkImage(Image.open(ASSETS / "logo.png"), size=(40, 40))
        ctk.CTkLabel(brand, image=logo, text="").pack(side="left")
        names = ctk.CTkFrame(brand, fg_color="transparent")
        names.pack(side="left", padx=10)
        ctk.CTkLabel(names, text="CandleCast", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(names, text="Kronos price forecasts", font=ctk.CTkFont(size=11), text_color=MUTED).pack(anchor="w")
        ctk.CTkLabel(side, text="RESULTS", font=ctk.CTkFont(size=11, weight="bold"), text_color=MUTED).grid(
            row=2, column=0, sticky="w", padx=20, pady=(14, 4))
        self.result_list = ctk.CTkScrollableFrame(side, fg_color="transparent")
        self.result_list.grid(row=3, column=0, sticky="nsew", padx=10)
        self.empty_hint = ctk.CTkLabel(self.result_list, text="Run a forecast to see\nresults here.",
                                       text_color=MUTED, justify="left")
        self.empty_hint.pack(anchor="w", padx=10, pady=6)
        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.grid(row=4, column=0, sticky="ew", padx=16, pady=16)
        ctk.CTkButton(bottom, text="Settings", fg_color="transparent", border_width=1,
                      text_color=("#0b0b0b", "#ffffff"), command=lambda: SettingsDialog(self)).pack(fill="x")
        ctk.CTkButton(bottom, text="About", fg_color="transparent", border_width=1,
                      text_color=("#0b0b0b", "#ffffff"), command=lambda: AboutDialog(self)).pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(bottom, text="Appearance", text_color=MUTED, font=ctk.CTkFont(size=11)).pack(anchor="w", pady=(14, 2))
        self.mode = ctk.CTkSegmentedButton(bottom, values=["Light", "Dark", "System"], command=self.on_mode)
        self.mode.set("System")
        self.mode.pack(fill="x")

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=18, pady=16)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        self.banner = ctk.CTkFrame(main, fg_color=("#fff4e5", "#3a2a17"), corner_radius=10)
        ctk.CTkLabel(self.banner, text="Add your free Alpaca keys to forecast US stocks and ETFs "
                     "(crypto, gold and FX work without keys).", text_color=("#7a4b00", "#f5c27a")).pack(
            side="left", padx=14, pady=8)
        ctk.CTkButton(self.banner, text="Open settings", width=120, command=lambda: SettingsDialog(self)).pack(
            side="right", padx=10, pady=6)

        bar = ctk.CTkFrame(main, fg_color=CARD, corner_radius=14)
        bar.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        bar.grid_columnconfigure(0, weight=1)
        r1 = ctk.CTkFrame(bar, fg_color="transparent")
        r1.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        r1.grid_columnconfigure(0, weight=1)
        self.tickers = ctk.CTkEntry(r1, placeholder_text="Tickers, e.g. AAPL, NVDA, SPY, BTCUSD", height=38,
                                    font=ctk.CTkFont(size=14))
        self.tickers.insert(0, "AAPL, NVDA, SPY")
        self.tickers.grid(row=0, column=0, sticky="ew")
        self.tickers.bind("<Return>", lambda _e: self.on_run())
        self.run_btn = ctk.CTkButton(r1, text="Forecast", height=38, width=140, fg_color=ACCENT,
                                     font=ctk.CTkFont(size=14, weight="bold"), command=self.on_run)
        self.run_btn.grid(row=0, column=1, padx=(12, 0))
        r2 = ctk.CTkFrame(bar, fg_color="transparent")
        r2.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))
        ctk.CTkLabel(r2, text="Timeframe", text_color=MUTED).pack(side="left")
        self.tf = ctk.CTkSegmentedButton(r2, values=list(TIMEFRAME_LABELS), command=lambda _v: self.on_timeframe())
        self.tf.set("1H")
        self.tf.pack(side="left", padx=(8, 20))
        ctk.CTkLabel(r2, text="Bars", text_color=MUTED).pack(side="left")
        self.bars = ctk.CTkSlider(r2, from_=timeframes.MIN_PRED, to=timeframes.MAX_PRED,
                                  number_of_steps=timeframes.MAX_PRED - timeframes.MIN_PRED, width=220,
                                  command=lambda _v: self.update_requirement())
        self.bars.set(timeframes.DEFAULT_PRED["1h"])
        self.bars.pack(side="left", padx=8)
        self.bars_label = ctk.CTkLabel(r2, text="32", width=36, font=ctk.CTkFont(weight="bold"))
        self.bars_label.pack(side="left")
        ctk.CTkLabel(r2, text="History", text_color=MUTED).pack(side="left", padx=(16, 0))
        self.history = ctk.CTkSlider(r2, from_=timeframes.MIN_HISTORY, to=timeframes.MAX_HISTORY,
                                     number_of_steps=(timeframes.MAX_HISTORY - timeframes.MIN_HISTORY) // 8,
                                     width=160, command=lambda _v: self.update_requirement())
        self.history.set(timeframes.DEFAULT_HISTORY)
        self.history.pack(side="left", padx=8)
        self.history_label = ctk.CTkLabel(r2, text="128", width=36, font=ctk.CTkFont(weight="bold"))
        self.history_label.pack(side="left")
        ctk.CTkLabel(r2, text="Paths", text_color=MUTED).pack(side="left", padx=(20, 6))
        self.paths = ctk.CTkOptionMenu(r2, values=["50", "100", "200"], width=80)
        self.paths.set(str(self.settings.model.sample_paths))
        self.paths.pack(side="left")
        self.rerun = ctk.CTkSwitch(r2, text="Re-run")
        self.rerun.pack(side="left", padx=(20, 0))
        self.req = ctk.CTkLabel(bar, text="", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.req.grid(row=2, column=0, sticky="w", padx=18, pady=(0, 12))

        chart_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=14)
        chart_card.grid(row=2, column=0, sticky="nsew")
        chart_card.grid_columnconfigure(0, weight=1)
        chart_card.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(chart_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        self.chart_title = ctk.CTkLabel(head, text="No forecast selected", font=ctk.CTkFont(size=17, weight="bold"))
        self.chart_title.pack(side="left")
        self.save_btn = ctk.CTkButton(head, text="Save JPG", width=96, fg_color="transparent", border_width=1,
                                      text_color=("#0b0b0b", "#ffffff"), command=self.save_jpg)
        self.save_btn.pack(side="right")
        self.band = ctk.CTkSegmentedButton(head, values=[c.replace("Raw (Kronos)", "Raw") for c in BAND_CHOICES],
                                           command=lambda _v: self.show_selected())
        default = f"{self.settings.chart_band_level}%" if self.settings.chart_band_level else "Raw"
        self.band.set(default if default in self.band.cget("values") else "80%")
        self.band.pack(side="right", padx=(0, 12))
        ctk.CTkLabel(head, text="Band", text_color=MUTED).pack(side="right", padx=(0, 6))
        self.chart_box = ctk.CTkFrame(chart_card, fg_color="transparent")
        self.chart_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))
        self.chart = ctk.CTkLabel(self.chart_box, text="Enter tickers above and press Forecast.", text_color=MUTED)
        self.chart.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.chart_box.bind("<Configure>", self._on_resize)

        cards = ctk.CTkFrame(main, fg_color="transparent")
        cards.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        cards.grid_columnconfigure((0, 1, 2), weight=1, uniform="c")
        self.card_forecast = self._card(cards, 0, "FORECAST")
        self.card_test = self._card(cards, 1, "TEST WINDOW")
        self.card_data = self._card(cards, 2, "DATA")

        status = ctk.CTkFrame(main, fg_color="transparent")
        status.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        status.grid_columnconfigure(1, weight=1)
        self.progress = ctk.CTkProgressBar(status, mode="determinate", height=6, progress_color=ACCENT)
        self.progress.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.progress.set(0)
        self.status = ctk.CTkLabel(status, text="Ready.", text_color=MUTED, anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.logbox = ctk.CTkTextbox(main, height=64, font=ctk.CTkFont(family="Consolas", size=11),
                                     fg_color=CARD, corner_radius=10)
        self.logbox.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        self.logbox.configure(state="disabled")
        self.update_requirement()

    def _card(self, parent, col, title):
        f = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14)
        f.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0 if col == 2 else 6))
        ctk.CTkLabel(f, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=MUTED).pack(
            anchor="w", padx=16, pady=(12, 0))
        big = ctk.CTkLabel(f, text="-", font=ctk.CTkFont(size=22, weight="bold"), anchor="w")
        big.pack(anchor="w", padx=16)
        small = ctk.CTkLabel(f, text="", text_color=MUTED, justify="left", anchor="w", wraplength=330)
        small.pack(anchor="w", padx=16, pady=(0, 12))
        return big, small

    # ---------------- controls
    def current_spec(self) -> custom.Spec:
        return custom.Spec(self.tf.get().lower(), int(round(self.bars.get())), int(round(self.history.get())))

    def on_timeframe(self):
        self.bars.set(timeframes.DEFAULT_PRED[self.tf.get().lower()])
        self.update_requirement()

    def update_requirement(self):
        spec = self.current_spec()
        self.bars_label.configure(text=str(spec.pred_len))
        used = timeframes.history_bars(spec.pred_len, self.settings.model.max_context, spec.history)
        self.history_label.configure(text=str(used))
        self.req.configure(text=requirement_text(spec.timeframe, spec.pred_len, self.settings.model.max_context,
                                                 spec.history)
                           + "  -  same history for any bar count, so longer forecasts extend shorter ones")

    def on_mode(self, value):
        ctk.set_appearance_mode(value.lower())
        self.after(50, self.show_selected)

    def refresh_key_banner(self):
        if has_alpaca_keys():
            self.banner.grid_forget()
        else:
            self.banner.grid(row=0, column=0, sticky="ew", pady=(0, 12))

    def log(self, msg: str):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")
        self.status.configure(text=msg[-160:])
        if self._file_log:
            _append_log(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} UI {msg}")

    # ---------------- run
    def on_run(self):
        if self.runner.busy:
            return
        tickers, invalid = parse_tickers(self.tickers.get())
        if invalid:
            self.log(f"Ignored (not a ticker): {', '.join(invalid)}")
        if not tickers:
            self.log("Enter at least one ticker, e.g. AAPL, MSFT, BTCUSD")
            return
        instruments, unmapped = resolve_tickers(self.settings, tickers)
        if unmapped:
            self.log(f"No data source for: {', '.join(unmapped)} (add them to config/symbol_map.csv)")
        if not instruments:
            return
        spec = self.current_spec()
        n_paths = int(self.paths.get())
        self.runner.busy = True
        self.run_btn.configure(state="disabled", text="Forecasting ...")
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self.log(f"Starting {', '.join(i.ticker for i in instruments)}: {spec.label} bars, {n_paths} paths")
        threading.Thread(target=self._work, args=(instruments, spec, n_paths, bool(self.rerun.get())),
                         daemon=True).start()

    def _work(self, instruments, spec, n_paths, rerun):
        handler = _QueueLog(self.q)
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        try:
            self.q.put(("done", self.runner.run(instruments, spec, n_paths, rerun)))
        except Exception as exc:  # keep the window alive whatever happens
            logging.getLogger(__name__).exception("run failed")
            self.q.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            root.removeHandler(handler)

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                else:
                    self.runner.busy = False
                    self.progress.stop()
                    self.progress.configure(mode="determinate")
                    self.progress.set(0)
                    self.run_btn.configure(state="normal", text="Forecast")
                    if kind == "error":
                        self.log(f"Run failed: {payload}")
                    else:
                        self.set_results(payload)
        except queue.Empty:
            pass
        self.after(120, self._poll)

    # ---------------- results
    def set_results(self, results: list[Result]):
        self.results = results
        for w in self.result_list.winfo_children():
            w.destroy()
        self._result_buttons = []
        for i, res in enumerate(results):
            if res.row:
                sub = f"{res.row['pred_len']} x {timeframes.LABELS[res.row['timeframe']]}"
                sub += "  ·  saved" if res.status == "cached" else ""
            else:
                sub = "failed"
            btn = ctk.CTkButton(self.result_list, text=f"{res.ticker}\n{sub}", anchor="w", height=52,
                                fg_color="transparent", hover_color=("#e8eefb", "#2b3442"),
                                text_color=(("#0b0b0b", "#ffffff") if res.row else BAD),
                                command=lambda i=i: self.select(i))
            btn.pack(fill="x", pady=2)
            self._result_buttons.append(btn)
        ok = sum(r.status != "failed" for r in results)
        self.log(f"Done: {ok} of {len(results)} ready.")
        if results:
            self.select(0)

    def select(self, i: int):
        self.selected = i
        for j, b in enumerate(getattr(self, "_result_buttons", [])):
            b.configure(fg_color=(("#dce8fb", "#27406b") if j == i else "transparent"))
        self.show_selected()

    def theme(self) -> str:
        return "dark" if ctk.get_appearance_mode().lower() == "dark" else "light"

    def band_level(self) -> int:
        v = self.band.get()
        return 0 if v == "Raw" else band_level_from_choice(v)

    def chart_path(self, res: Result, level: int, theme: str, header: bool) -> Path:
        row = res.row
        stamp = pd.Timestamp(row["as_of_bar"]).strftime("%Y%m%dT%H%M")
        name = (f"{res.ticker}_{row['timeframe']}_{row['pred_len']}b_h{row['history_bars']}_{stamp}_{row['n_paths']}p_"
                f"{level or 'raw'}_{theme}{'_full' if header else ''}.png")
        path = self.settings.root / "outputs" / "app_charts" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            custom.render_custom(self.settings, row, res.test, path, level, theme=theme, header=header)
        return path

    def show_selected(self):
        if self.selected is None or self.selected >= len(self.results):
            return
        res = self.results[self.selected]
        if res.row is None:
            self.chart_title.configure(text=f"{res.ticker} - forecast failed")
            self.chart.configure(image=None, text=res.error)
            self._shown = None
            for big, small in (self.card_forecast, self.card_test, self.card_data):
                big.configure(text="-"); small.configure(text="")
            return
        level = self.band_level()
        try:
            path = self.chart_path(res, level, self.theme(), header=False)
        except Exception as exc:
            self.chart.configure(image=None, text=f"Chart unavailable: {type(exc).__name__}: {exc}")
            return
        self._shown = (res, level)
        row, test = res.row, res.test
        self.chart_title.configure(text=f"{res.ticker}  ·  {row['pred_len']} x {timeframes.LABELS[row['timeframe']]} bars")
        self._fill_cards(res, level)
        img = Image.open(path)
        w, h = max(self.chart_box.winfo_width() - 8, 300), max(self.chart_box.winfo_height() - 8, 200)
        scale = min(w / img.width, h / img.height, 1.0)
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        self._chart_image = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        self.chart.configure(image=self._chart_image, text="")

    def _fill_cards(self, res: Result, level: int):
        from candlecast.pipeline import band_for  # noqa: PLC0415

        row, test = res.row, res.test
        last, p10, p50, p90 = row["last_close"], row["end_close_p10"], row["end_close_p50"], row["end_close_p90"]
        tf = row["timeframe"]
        fmt = time_format(tf)
        test_summary = None
        if test and (self.settings.root / test["summary_path"]).exists():
            test_summary = pd.read_parquet(self.settings.root / test["summary_path"])
        band = band_for(self.settings, row["asset_class"], level, test_summary)
        if band:
            k = band["k"]
            lo, hi = p50 - k * (p50 - p10), p50 + k * (p90 - p50)
            rng = f"{band['level']}% range {lo:,.2f} - {hi:,.2f}"
        else:
            rng = f"raw 80% range {p10:,.2f} - {p90:,.2f} (too narrow)"
        end = pd.Timestamp(row["horizon_end"]).tz_convert(NY)
        self.card_forecast[0].configure(text=f"{p50:,.2f}  ({p50 / last - 1:+.1%})")
        approx = (band is not None and (tf, int(row["pred_len"])) != CALIBRATED_SPEC)
        self.card_forecast[1].configure(text=(
            f"Median end close by {end.strftime(fmt)}\n{rng}{' (approx.)' if approx else ''}\n"
            f"Last close {last:,.2f} · P(end higher) {row['prob_up']:.0%}"))
        if test:
            err, naive = test["abs_err_kronos"], test["abs_err_naive"]
            better = err <= naive
            self.card_test[0].configure(text=f"{err:.1%} error", text_color=GOOD if better else BAD)
            cov = band["test_coverage"] if band and band.get("test_coverage") is not None else test.get("cover_bars")
            cutoff = pd.Timestamp(test["cutoff_bar"]).tz_convert(NY)
            self.card_test[1].configure(text=(
                f"vs {naive:.1%} for a 'no change' guess ({'better' if better else 'worse'})\n"
                f"{cov:.0%} of actual closes inside the band · direction {'right' if test['dir_hit'] else 'wrong'}\n"
                f"Made {cutoff.strftime(fmt)} using only data up to then"))
        else:
            self.card_test[0].configure(text="-", text_color=("#0b0b0b", "#ffffff"))
            self.card_test[1].configure(text="No test window.")
        as_of = pd.Timestamp(row["as_of_bar"]).tz_convert(NY)
        self.card_data[0].configure(text=f"{row['history_bars']} bars")
        self.card_data[1].configure(text=(
            f"History used · {row['n_paths']} paths · Kronos-{row['model']}\nLast bar {as_of.strftime(fmt)}\n"
            f"{row.get('data_note', '')}"))

    def save_jpg(self, target: str | None = None) -> str | None:
        if not self._shown:
            self.log("Nothing to save yet - run a forecast first.")
            return None
        res, level = self._shown
        if target is None:
            from tkinter import filedialog  # noqa: PLC0415

            row = res.row
            stamp = pd.Timestamp(row["as_of_bar"]).tz_convert(NY).strftime("%Y%m%d_%H%M")
            name = (f"{res.ticker}_{row['pred_len']}x{timeframes.LABELS[row['timeframe']]}_{stamp}_"
                    f"{f'band{level}' if level else 'rawband'}.jpg")
            target = filedialog.asksaveasfilename(parent=self, title="Save chart as JPG", defaultextension=".jpg",
                                                  initialfile=name, filetypes=[("JPEG image", "*.jpg *.jpeg")])
            if not target:
                return None
        save_as_jpg(self.chart_path(res, level, self.theme(), header=True), target)
        self.log(f"Saved {target}")
        return target

    def _on_resize(self, _e):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self.show_selected)


def main(argv: list[str] | None = None) -> None:
    import sys  # noqa: PLC0415

    _set_windows_app_id()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="CandleCast", description="CandleCast desktop app")
    parser.add_argument("tickers", nargs="*", help="fill these in and start forecasting immediately")
    parser.add_argument("--timeframe", "-t", choices=[t.lower() for t in TIMEFRAME_LABELS], type=str.lower)
    parser.add_argument("--bars", "-n", type=int, help=f"bars to forecast ({timeframes.MIN_PRED}-{timeframes.MAX_PRED})")
    args, _unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    tickers, _ = parse_tickers(" ".join(args.tickers))
    _setup_file_log()
    logging.getLogger(__name__).info("CandleCast %s starting; data folder %s", __version__, DATA_DIR)
    app = CandleCastApp(tickers=tickers or None, autorun=bool(tickers), timeframe=args.timeframe, bars=args.bars)
    app._file_log = True
    app.mainloop()


if __name__ == "__main__":
    main()
