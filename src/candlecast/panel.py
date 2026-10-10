"""Desktop panel: type tickers, choose a bar timeframe and how many bars to forecast, get forecasts.

For each ticker the panel loads its bars at the chosen timeframe (1H / 4H / 1D / 1W) - local database
first, refreshed or downloaded online when needed (candlecast.timeframes) - and makes two Kronos
forecasts of the same length N: the forward forecast of the next N bars, and a test forecast of the last
N bars that already happened (made from only the bars before them), drawn against the actual prices
and scored. History length follows the fixed rule in timeframes.history_bars (8 x N, 64..512-N).
A ticker/timeframe/N whose latest bar was already forecast is shown from the saved files unless
"Re-run" is ticked. Band: calibrated 40-80% bands (or Kronos's raw ones); Save as JPG.

Any US stock/ETF ticker works (Alpaca); crypto/FX/commodities need a row in config/symbol_map.csv
(BTCUSD, ETHUSD, XAUUSD, ... are there). The model is loaded once, on the first run.

Start: ``candlecast panel [TICKER ...]`` or ``.venv\\Scripts\\candlecast-panel.exe [TICKER ...]``.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import pandas as pd

from candlecast import calibration, custom, timeframes
from candlecast.calendars import NY
from candlecast.charts import caption_for_test, time_format
from candlecast.settings import load_settings
from candlecast.symbols import read_symbol_map, resolve

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
NOISY_LOGGERS = ("urllib3", "yfinance", "peewee", "matplotlib", "huggingface_hub", "PIL")
TIMEFRAME_CHOICES = ("1H", "4H", "1D", "1W")
CALIBRATED_SPEC = ("1h", 32)   # the band factors were fitted on 1H x 32-bar forecasts


# ---------------------------------------------------------------- logic (no tkinter; unit-tested)

def parse_tickers(text: str) -> tuple[list[str], list[str]]:
    """Split on commas/spaces/semicolons, upper-case, de-duplicate. Returns (valid, invalid)."""
    valid, invalid = [], []
    for raw in re.split(r"[\s,;]+", text or ""):
        t = raw.strip().upper()
        if not t:
            continue
        (valid if TICKER_RE.match(t) else invalid).append(t)
    return list(dict.fromkeys(valid)), list(dict.fromkeys(invalid))


def resolve_tickers(settings, tickers: list[str]):
    """Instruments for the tickers (symbol map first, else Alpaca US stock); unmapped ones are rejected."""
    df = pd.DataFrame({"ticker": tickers, "exchange": [""] * len(tickers)})
    instruments = resolve(df, read_symbol_map(settings.data.symbol_map_csv))
    ok = [i for i in instruments if i.source is not None]
    return ok, [i.ticker for i in instruments if i.source is None]


BAND_CHOICES = ("40%", "50%", "60%", "70%", "80%", "Raw (Kronos)")


def band_level_from_choice(choice: str) -> int:
    """'60%' -> 60; 'Raw (Kronos)' -> 0 (Kronos's own uncalibrated bands)."""
    return int(choice.rstrip("%")) if choice.endswith("%") else 0


def timeframe_from_choice(choice: str) -> str:
    return choice.lower()


def requirement_text(timeframe: str, pred_len: int, max_context: int = 512, history: int | None = None) -> str:
    """What the forecast needs, in bars and roughly in calendar time (US-stock sessions)."""
    hist = timeframes.history_bars(pred_len, max_context, history)
    total = hist + pred_len
    per = {"1h": (16, "trading days"), "4h": (4, "trading days"), "1d": (21, "months"), "1w": (52, "years")}
    div, unit = per[timeframe]
    return (f"uses {hist} history bars + {pred_len} test bars = {total} {timeframes.LABELS[timeframe]} bars "
            f"(~{total / div:.1f} {unit})")


def describe(row: dict, test: dict | None, band: dict | None = None, calibrated_on: int | None = None) -> str:
    """Plain-text summary of one forecast (and its test window), as shown under the chart.
    ``band``: pipeline.band_for() result for the band level shown on the chart (None = raw bands)."""
    last = row["last_close"]
    tf = row.get("timeframe", "1h")
    fmt = time_format(tf)

    def pct(v):
        return f"{v:,.2f} ({v / last - 1:+.2%})"

    as_of = pd.Timestamp(row["as_of_bar"]).tz_convert(NY)
    start, end = (pd.Timestamp(row[k]).tz_convert(NY) for k in ("horizon_start", "horizon_end"))
    p10, p50, p90 = row["end_close_p10"], row["end_close_p50"], row["end_close_p90"]
    if band is None:
        rng = f"raw Kronos 80% range {p10:,.2f} - {p90:,.2f} (too narrow: held ~40% of past outcomes)"
    else:
        k = band["k"]
        rng = f"calibrated {band['level']}% range {p50 - k * (p50 - p10):,.2f} - {p50 + k * (p90 - p50):,.2f}"
    spec = f"{row['pred_len']} x {timeframes.LABELS.get(tf, tf)} bars"
    lines = [
        f"{row['ticker']} - {spec}, Kronos-{row['model']}, {row['n_paths']} paths, "
        f"{row.get('history_bars', '?')} history bars - last close {last:,.2f} at {as_of.strftime(fmt)}",
        f"Forecast {start.strftime(fmt)} -> {end.strftime(fmt)}: end close median {pct(p50)}, {rng}; "
        f"P(end above last close) {row['prob_up']:.0%}",
    ]
    if test:
        cutoff = pd.Timestamp(test["cutoff_bar"]).tz_convert(NY)
        lines += caption_for_test(test, cutoff, band, tf).replace(" · ", ", ").split("\n")
    else:
        lines.append("No test window for this forecast.")
    note = ("Accuracy note: in 400+ past test forecasts the median line never beat a 'no change' guess - "
            "use the band as a range, not the line as a direction.")
    if band is not None and (tf, int(row["pred_len"])) != CALIBRATED_SPEC:
        note += (f" Band factors were fitted on 1H x 32-bar forecasts, so for {spec} they are approximate - "
                 "check the test window's coverage.")
    if band is not None and calibrated_on is not None and calibrated_on < 100:
        note += f" Calibration for this asset class rests on only {calibrated_on} past forecasts (rough)."
    if row.get("data_note"):
        note += f" Data: {row['data_note']}."
    lines.append(note)
    return "\n".join(lines)


@dataclass
class Result:
    ticker: str
    status: str            # "new", "cached", "failed"
    row: dict | None = None
    test: dict | None = None
    error: str = ""


class ForecastRunner:
    """Runs custom forecasts in a background thread; keeps the Kronos predictor between runs.
    ``make_source(inst)`` (tests) supplies the data source; default: the real one per instrument."""

    def __init__(self, settings, log, make_predictor=None, make_source=None):
        self.settings = settings
        self.log = log
        self._make_predictor = make_predictor
        self._make_source = make_source
        self._predictor = None
        self.busy = False

    def _predictor_once(self):
        if self._predictor is None:
            if self._make_predictor is not None:
                self._predictor = self._make_predictor()
            else:
                from candlecast.kronos_loader import configure_torch_threads, predictor_for  # noqa: PLC0415

                self.log(f"Loading the Kronos-{self.settings.model.name} model (downloaded once the first time, then loaded from disk) ...")
                configure_torch_threads(self.settings.model.torch_threads)
                self._predictor = predictor_for(self.settings.model, offline=getattr(self.settings, "offline", False))
        return self._predictor

    SLOW_AFTER_S = 120   # a normal ticker takes 10-90 s

    def _watch(self, ticker: str):
        """While one ticker runs: tell the user if it is unusually slow, and write every thread's stack to
        logs/slow_run_stacks.log (every 2 min) so a hang can be diagnosed afterwards."""
        import faulthandler  # noqa: PLC0415
        import threading  # noqa: PLC0415

        from candlecast.settings import DATA_DIR  # noqa: PLC0415

        t0 = time.perf_counter()
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(self.SLOW_AFTER_S):
                self.log(f"{ticker}: still working after {time.perf_counter() - t0:.0f}s - slower than usual "
                         "(network or CPU); details are recorded in logs/slow_run_stacks.log")

        threading.Thread(target=heartbeat, daemon=True).start()
        dump = None
        try:
            (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
            dump = open(DATA_DIR / "logs" / "slow_run_stacks.log", "a", encoding="utf-8")  # noqa: SIM115
            dump.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {ticker} started ===\n")
            dump.flush()
            faulthandler.dump_traceback_later(self.SLOW_AFTER_S, repeat=True, file=dump)
        except (OSError, ValueError, RuntimeError):
            pass

        def done():
            stop.set()
            if dump is not None:
                faulthandler.cancel_dump_traceback_later()
                dump.close()
        return done

    def run(self, instruments, spec: custom.Spec, n_paths: int, rerun: bool) -> list[Result]:
        predictor = self._predictor_once()
        con = custom.connect(self.settings.index_db)
        results = []
        try:
            for inst in instruments:
                t0 = time.perf_counter()
                self.log(f"{inst.ticker}: updating prices and forecasting ...")
                unwatch = self._watch(inst.ticker)
                try:
                    source = self._make_source(inst) if self._make_source else None
                    row, test, status = custom.run_one(self.settings, inst, spec, predictor, n_paths, con,
                                                       rerun=rerun, source=source)
                    self.log(f"{inst.ticker} {spec.label}: {'saved forecast reused' if status == 'cached' else 'done'}"
                             f" in {time.perf_counter() - t0:.0f}s ({row.get('data_note', '')})")
                    results.append(Result(inst.ticker, status, row, test))
                except Exception as exc:  # one ticker's failure never stops the others
                    self.log(f"{inst.ticker} {spec.label}: FAILED - {type(exc).__name__}: {exc}")
                    results.append(Result(inst.ticker, "failed", error=f"{inst.ticker}: {type(exc).__name__}: {exc}"))
                finally:
                    unwatch()
            return results
        finally:
            con.close()


def save_as_jpg(png_path, jpg_path, quality: int = 95) -> None:
    """Convert a chart PNG to a JPG file (white background; JPG has no transparency)."""
    from PIL import Image  # noqa: PLC0415

    with Image.open(png_path) as img:
        img.convert("RGB").save(jpg_path, "JPEG", quality=quality, optimize=True)


# ---------------------------------------------------------------- GUI

class _QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__(logging.INFO)
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.q.put(("log", self.format(record)))


class PanelApp:
    def __init__(self, root: tk.Tk, settings=None, runner: ForecastRunner | None = None,
                 tickers: list[str] | None = None, autorun: bool = False):
        self.root = root
        self.settings = settings or load_settings()
        self.q: queue.Queue = queue.Queue()
        self.runner = runner or ForecastRunner(self.settings, log=lambda m: self.q.put(("log", m)))
        self.results: list[Result] = []
        self._photo = None
        self._shown = None
        self._build()
        if tickers:
            self.ticker_var.set(", ".join(tickers))
        if autorun and tickers:
            self.root.after(300, self.on_run)
        self.root.after(150, self._poll)

    # layout
    def _build(self):
        r = self.root
        r.title("Price Forecast - Kronos panel")
        r.geometry("1280x930")
        r.minsize(900, 680)
        top = ttk.Frame(r, padding=(8, 8, 8, 2))
        top.pack(fill="x")
        ttk.Label(top, text="Tickers:").pack(side="left")
        self.ticker_var = tk.StringVar(value="AAPL, NVDA, SPY")
        entry = ttk.Entry(top, textvariable=self.ticker_var, width=32)
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", lambda _e: self.on_run())
        ttk.Label(top, text="Paths:").pack(side="left", padx=(8, 2))
        self.paths_var = tk.IntVar(value=self.settings.model.sample_paths)
        ttk.Spinbox(top, from_=10, to=200, increment=10, textvariable=self.paths_var, width=5).pack(side="left")
        self.rerun_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Re-run", variable=self.rerun_var).pack(side="left", padx=8)
        self.run_btn = ttk.Button(top, text="Run forecast", command=self.on_run)
        self.run_btn.pack(side="left")
        ttk.Button(top, text="Open folder", command=self.open_folder).pack(side="right")

        row2 = ttk.Frame(r, padding=(8, 2, 8, 6))
        row2.pack(fill="x")
        ttk.Label(row2, text="Timeframe:").pack(side="left")
        self.tf_var = tk.StringVar(value="1H")
        tf_box = ttk.Combobox(row2, textvariable=self.tf_var, values=TIMEFRAME_CHOICES, width=5, state="readonly")
        tf_box.pack(side="left", padx=(4, 12))
        tf_box.bind("<<ComboboxSelected>>", lambda _e: self.on_timeframe())
        ttk.Label(row2, text="Bars to forecast (= test bars):").pack(side="left")
        self.bars_var = tk.IntVar(value=timeframes.DEFAULT_PRED["1h"])
        bars_spin = ttk.Spinbox(row2, from_=timeframes.MIN_PRED, to=timeframes.MAX_PRED, increment=1,
                                textvariable=self.bars_var, width=5, command=self.update_requirement)
        bars_spin.pack(side="left", padx=(4, 12))
        bars_spin.bind("<KeyRelease>", lambda _e: self.update_requirement())
        ttk.Label(row2, text="Band:").pack(side="left", padx=(8, 2))
        default = f"{self.settings.chart_band_level}%" if self.settings.chart_band_level else "Raw (Kronos)"
        self.band_var = tk.StringVar(value=default if default in BAND_CHOICES else "80%")
        band_box = ttk.Combobox(row2, textvariable=self.band_var, values=BAND_CHOICES, width=12, state="readonly")
        band_box.pack(side="left")
        band_box.bind("<<ComboboxSelected>>", lambda _e: self.show_selected())
        ttk.Button(row2, text="Save as JPG...", command=self.save_jpg).pack(side="left", padx=(8, 0))
        self.req_var = tk.StringVar()
        ttk.Label(row2, textvariable=self.req_var, foreground="#52514e").pack(side="left", padx=(14, 0))
        self.update_requirement()

        body = ttk.PanedWindow(r, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8)
        left = ttk.Frame(body, width=210)
        ttk.Label(left, text="Results").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=26, activestyle="none", exportselection=False)
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self.show_selected())
        body.add(left, weight=0)
        right = ttk.Frame(body)
        self.summary = tk.Text(right, height=8, wrap="word", font=("Segoe UI", 9), relief="flat")
        self.summary.pack(side="bottom", fill="x", pady=(4, 0))
        # the chart box never resizes to its picture (pack_propagate off); the picture is scaled to the box
        self.chart_box = ttk.Frame(right)
        self.chart_box.pack(side="top", fill="both", expand=True)
        self.chart_box.pack_propagate(False)
        self.chart = ttk.Label(self.chart_box, anchor="center", text="Enter tickers and press Run forecast.")
        self.chart.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._resize_job = None
        self.chart_box.bind("<Configure>", self._on_resize)
        body.add(right, weight=1)

        ttk.Label(r, text="Progress").pack(anchor="w", padx=8)
        self.logbox = tk.Text(r, height=7, wrap="none", font=("Consolas", 9), state="disabled")
        self.logbox.pack(fill="x", padx=8, pady=(0, 8))

    # spec controls
    def current_spec(self) -> custom.Spec:
        tf = timeframe_from_choice(self.tf_var.get())
        try:
            n = int(self.bars_var.get())
        except (tk.TclError, ValueError):
            n = timeframes.DEFAULT_PRED[tf]
        return custom.Spec(tf, max(timeframes.MIN_PRED, min(timeframes.MAX_PRED, n)))

    def on_timeframe(self):
        self.bars_var.set(timeframes.DEFAULT_PRED[timeframe_from_choice(self.tf_var.get())])
        self.update_requirement()

    def update_requirement(self):
        spec = self.current_spec()
        self.req_var.set(requirement_text(spec.timeframe, spec.pred_len, self.settings.model.max_context))

    # actions
    def log(self, msg: str):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def on_run(self):
        if self.runner.busy:
            return
        tickers, invalid = parse_tickers(self.ticker_var.get())
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
        try:
            n_paths = max(10, min(200, int(self.paths_var.get())))
        except (tk.TclError, ValueError):
            n_paths = self.settings.model.sample_paths
        spec = self.current_spec()
        self.bars_var.set(spec.pred_len)
        self.runner.busy = True
        self.run_btn.configure(state="disabled", text="Running ...")
        self.log(f"Starting {', '.join(i.ticker for i in instruments)}: {spec.label} bars, {n_paths} paths; "
                 f"{requirement_text(spec.timeframe, spec.pred_len, self.settings.model.max_context)}.")
        threading.Thread(target=self._work, args=(instruments, spec, n_paths, self.rerun_var.get()),
                         daemon=True).start()

    def _work(self, instruments, spec, n_paths, rerun):
        handler = _QueueLogHandler(self.q)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        try:
            self.q.put(("done", self.runner.run(instruments, spec, n_paths, rerun)))
        except Exception as exc:  # keep the panel alive whatever happens
            logging.getLogger(__name__).exception("run failed")
            self.q.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            root_logger.removeHandler(handler)

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind in ("done", "error"):
                    self.runner.busy = False
                    self.run_btn.configure(state="normal", text="Run forecast")
                    if kind == "error":
                        self.log(f"Run failed: {payload}")
                    else:
                        self.set_results(payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    def set_results(self, results: list[Result]):
        self.results = results
        self.listbox.delete(0, "end")
        for res in results:
            label = {"new": "new", "cached": "saved", "failed": "FAILED"}[res.status]
            spec = (f"{res.row['pred_len']}x{timeframes.LABELS[res.row['timeframe']]}" if res.row else "")
            self.listbox.insert("end", f"{res.ticker:7s} {spec:8s} {label}")
        ok = sum(r.status != "failed" for r in results)
        self.log(f"Done: {ok} of {len(results)} ticker(s) ready. Click a ticker on the left to view it.")
        if results:
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(0)
            self.show_selected()

    def show_selected(self):
        sel = self.listbox.curselection()
        if not sel or sel[0] >= len(self.results):
            return
        res = self.results[sel[0]]
        self.summary.delete("1.0", "end")
        if res.status == "failed":
            self.chart.configure(image="", text=f"{res.ticker}: forecast failed")
            self.summary.insert("1.0", res.error)
            self._shown = None
            return
        level = band_level_from_choice(self.band_var.get())
        try:
            chart_path, band = self.chart_for(res, level)
        except Exception as exc:  # e.g. saved files missing
            self.chart.configure(image="", text=f"{res.ticker}: chart unavailable ({type(exc).__name__}: {exc})")
            self.summary.insert("1.0", describe(res.row, res.test))
            return
        cal = calibration.load(self.settings.root)
        self.summary.insert("1.0", describe(res.row, res.test, band,
                                            calibration.group_size(cal, res.row["asset_class"])))
        self._shown = (res, chart_path, level)
        from PIL import Image, ImageTk  # noqa: PLC0415

        img = Image.open(chart_path)
        w, h = max(self.chart_box.winfo_width(), 200), max(self.chart_box.winfo_height(), 150)
        scale = min(w / img.width, h / img.height, 1.0)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.chart.configure(image=self._photo, text="")

    def chart_for(self, res: Result, level: int):
        """Full-size chart PNG of this forecast at this band level (drawn once, then cached)."""
        from candlecast.pipeline import band_for  # noqa: PLC0415

        row = res.row
        stamp = pd.Timestamp(row["as_of_bar"]).strftime("%Y%m%dT%H%M")
        name = f"{res.ticker}_{row['timeframe']}_{row['pred_len']}b_{stamp}_{row['n_paths']}p_{level or 'raw'}.png"
        path = self.settings.root / "outputs" / "panel_charts" / name
        test_summary = None
        if res.test and (self.settings.root / res.test["summary_path"]).exists():
            test_summary = pd.read_parquet(self.settings.root / res.test["summary_path"])
        band = band_for(self.settings, row["asset_class"], level, test_summary)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            custom.render_custom(self.settings, row, res.test, path, level)
        return path, band

    def save_jpg(self, target: str | None = None) -> str | None:
        """Save the chart currently shown (at the chosen band level) as a JPG; asks for a file name."""
        if not self._shown:
            self.log("Nothing to save yet - run a forecast and select a ticker first.")
            return None
        res, chart_path, level = self._shown
        if target is None:
            from tkinter import filedialog  # noqa: PLC0415

            row = res.row
            stamp = pd.Timestamp(row["as_of_bar"]).tz_convert(NY).strftime("%Y%m%d_%H%M")
            name = (f"{res.ticker}_{row['pred_len']}x{timeframes.LABELS[row['timeframe']]}_{stamp}_"
                    f"{f'band{level}' if level else 'rawband'}.jpg")
            target = filedialog.asksaveasfilename(
                parent=self.root, title="Save chart as JPG", defaultextension=".jpg", initialfile=name,
                filetypes=[("JPEG image", "*.jpg *.jpeg")])
            if not target:
                return None
        save_as_jpg(chart_path, target)
        self.log(f"Saved {target}")
        return target

    def _on_resize(self, _event):
        # re-fit the chart once the window stops changing size (avoid re-scaling on every pixel)
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self.show_selected)

    def open_folder(self):
        folder = self.settings.root / "outputs" / "custom"
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)  # noqa: S606 - Windows: open in Explorer


def main(argv: list[str] | None = None) -> None:
    """``candlecast-panel [TICKER ...]``: tickers given on the command line are filled in and run."""
    import sys  # noqa: PLC0415

    args = sys.argv[1:] if argv is None else argv
    tickers, _ = parse_tickers(" ".join(args))
    root = tk.Tk()
    PanelApp(root, tickers=tickers or None, autorun=bool(tickers))
    root.mainloop()


if __name__ == "__main__":
    main()
