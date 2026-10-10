"""Static PNG forecast chart: recent closes + forecast median with p25-p75 / p10-p90 bands, and
optionally a test forecast: the same forecast made ``pred_len`` bars ago, drawn against the closes
that actually happened since.

Colors follow the reference data-viz palette: neutral ink for history/actual closes, categorical
slot 1 (blue, sequential steps 100 / 200 / 450) for the live forecast, slot 2 (orange) for the test
forecast, recessive grid, text in text tokens. The x-axis is bar-indexed so overnight/weekend gaps
don't squash the plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from candlecast.calendars import NY  # noqa: E402

# Light and dark themes (reference data-viz palette; dark steps are the palette's dark-surface values)
THEMES = {
    "light": {"surface": "#fcfcfb", "text_primary": "#0b0b0b", "text_secondary": "#52514e",
              "grid": "#e4e3df", "history": "#52514e", "band_outer": "#cde2fb", "band_inner": "#9ec5f4",
              "median": "#2a78d6", "test": "#eb6834", "test_band_alpha": 0.18},
    "dark": {"surface": "#1a1a19", "text_primary": "#ffffff", "text_secondary": "#c3c2b7",
             "grid": "#2c2c2a", "history": "#c3c2b7", "band_outer": "#163e6e", "band_inner": "#1e4f8a",
             "median": "#6da7ec", "test": "#d95926", "test_band_alpha": 0.22},
}
# light-theme names kept for older imports
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"


TF_LABEL = {"1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"}


def time_format(timeframe: str) -> str:
    """How a bar time is written: daily/weekly bars have no meaningful clock time."""
    return "%a %Y-%m-%d" if timeframe in ("1d", "1w") else "%a %m/%d %H:%M"


def thin_ticks(starts: list[int], min_gap: int) -> list[int]:
    """Keep period starts at least ``min_gap`` bars apart (labels closer than that collide). Greedy from
    the right, so the most recent period is always labelled."""
    kept: list[int] = []
    for s_ in reversed(starts):
        if not kept or kept[-1] - s_ >= min_gap:
            kept.append(s_)
    return kept[::-1]


def caption_for_test(test_row: dict, cutoff: pd.Timestamp, band: dict | None = None,
                     timeframe: str = "1h") -> str:
    """Two lines: what the test forecast was, and how it did (same metrics as `candlecast score`).
    With ``band`` ({"level", "test_coverage"}) the coverage reported is that of the calibrated band shown."""
    parts = [f"end-close error {test_row['abs_err_kronos']:.1%} (no-change guess {test_row['abs_err_naive']:.1%})"]
    if band is not None and band.get("test_coverage") is not None:
        parts.append(f"{band['test_coverage']:.0%} of actual closes inside its calibrated {band['level']}% band")
    elif test_row.get("cover_bars") is not None:
        parts.append(f"{test_row['cover_bars']:.0%} of actual closes inside its raw 80% band")
    parts.append(f"direction {'right' if test_row['dir_hit'] else 'wrong'}")
    when = cutoff.strftime(time_format(timeframe)) + ("" if timeframe in ("1d", "1w") else " New York")
    return (f"Test (orange): forecast made {when} using only data up to then\n"
            f"How it did: {' · '.join(parts)}")


def render(path: Path, ticker: str, history: pd.DataFrame, summary: pd.DataFrame, horizon: dict,
           model_label: str, n_paths: int, test: dict | None = None, band: dict | None = None,
           timeframe: str = "1h", theme: str = "light", header: bool = True) -> Path:
    """``history``: bars (timestamp, close) shown left of the forecast; ``summary``: bar_summary().
    ``test`` (optional): {"summary": test per-bar quantiles, "cutoff_bar", "cutoff_close", "row": scores};
    drawn only if its cutoff bar is inside ``history``.
    ``band`` (optional): {"level": 40..80, "k": widening factor, "test_coverage": share or None} draws one
    calibrated band (see calibration.py) instead of Kronos's raw 50%/80% bands.
    ``timeframe`` ("1h"/"4h"/"1d"/"1w") sets the title and the date labels on the x-axis.
    ``theme``: "light" or "dark". ``header=False`` leaves out the title/subtitle/test caption (the app
    shows them in its own cards) - the legend always stays."""
    from candlecast import calibration  # noqa: PLC0415

    c = THEMES.get(theme, THEMES["light"])
    n_hist = len(history)
    x_hist = list(range(n_hist))
    x_fc = list(range(n_hist - 1, n_hist + len(summary)))  # start at the last close so lines join
    last = float(history["close"].iloc[-1])

    def joined(col):
        return [last] + summary[col].tolist()

    fig, ax = plt.subplots(figsize=(10, (6.0 if test else 5.2) if header else 4.6), dpi=130)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    if band is None:
        ax.fill_between(x_fc, joined("close_p10"), joined("close_p90"), color=c["band_outer"], linewidth=0,
                        label="80% band (p10-p90)")
        ax.fill_between(x_fc, joined("close_p25"), joined("close_p75"), color=c["band_inner"], linewidth=0,
                        label="50% band (p25-p75)")
    else:
        lo, hi = calibration.band(summary, band["k"])
        ax.fill_between(x_fc, [last] + lo.tolist(), [last] + hi.tolist(), color=c["band_inner"], linewidth=0,
                        label=f"Calibrated {band['level']}% band")

    cutoff_x = None
    if test is not None:
        hits = history.index[history["timestamp"] == pd.Timestamp(test["cutoff_bar"])].tolist()
        if hits:
            cutoff_x = hits[0]
            t = test["summary"]
            c0 = float(test["cutoff_close"])
            x_t = list(range(cutoff_x, cutoff_x + len(t) + 1))
            if band is None:
                t_lo, t_hi, t_label = t["close_p10"], t["close_p90"], "Test 80% band"
            else:
                t_lo, t_hi = calibration.band(t, band["k"])
                t_label = f"Test calibrated {band['level']}% band"
            ax.fill_between(x_t, [c0] + t_lo.tolist(), [c0] + t_hi.tolist(),
                            color=c["test"], alpha=c["test_band_alpha"], linewidth=0, label=t_label)
            ax.plot(x_t, [c0] + t["close_p50"].tolist(), color=c["test"], linewidth=2, label="Test forecast median")
            ax.axvline(cutoff_x, color=c["grid"], linewidth=1.2, zorder=0)
            ax.annotate("test forecast made", (cutoff_x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -2), textcoords="offset points", va="top", ha="left", fontsize=7.5,
                        color=c["text_secondary"])

    # actual closes on top of the test band so the comparison is never hidden
    ax.plot(x_hist, history["close"], color=c["history"], linewidth=2, label="Close (actual)", zorder=3)
    ax.plot(x_fc, joined("close_p50"), color=c["median"], linewidth=2, label="Median forecast", zorder=3)
    ax.axvline(n_hist - 1, color=c["grid"], linewidth=1.2, zorder=0)
    if cutoff_x is not None:
        ax.annotate("now", (n_hist - 1, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                    textcoords="offset points", va="top", ha="left", fontsize=7.5, color=c["text_secondary"])

    # Kronos's own (uncalibrated) median high/low only make sense next to its raw bands
    for key, text in ((("max_high_p50", "median highest high"), ("min_low_p50", "median lowest low"))
                      if band is None else ()):
        y = horizon[key]
        ax.hlines(y, n_hist - 1, x_fc[-1], colors=c["text_secondary"], linewidths=1, linestyles=(0, (4, 3)))
        ax.annotate(f"{text} {y:,.2f}", (x_fc[-1], y), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=c["text_secondary"])

    # date labels at the first bar of each period (day for 1h/4h, week or month for 1d, month or quarter
    # for 1w), thinned so they never collide
    ts = pd.concat([history["timestamp"], summary["timestamp"]], ignore_index=True).dt.tz_convert(NY)
    total = len(ts)
    if timeframe == "1d":
        period, fmt = (ts.dt.strftime("%G-%V"), "%b %d") if total <= 70 else (ts.dt.strftime("%Y-%m"), "%b %Y")
    elif timeframe == "1w":
        period, fmt = (ts.dt.strftime("%Y-%m"), "%b %Y") if total <= 60 else (
            ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str), "%b %Y")
    else:
        period, fmt = ts.dt.date.astype(str), "%a %m/%d"
    starts = [i for i in range(total) if i == 0 or period.iloc[i] != period.iloc[i - 1]]
    starts = thin_ticks(starts, min_gap=max(4, total // 14))
    ax.set_xticks(starts)
    ax.set_xticklabels([ts.iloc[i].strftime(fmt) for i in starts], fontsize=8, color=c["text_secondary"])
    ax.tick_params(axis="y", labelsize=8, colors=c["text_secondary"])
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", color=c["grid"], linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.set_xlim(0, x_fc[-1] + max(6, len(summary) // 3))  # room for the right-hand labels

    top = 0.93
    if header:
        as_of = pd.Timestamp(history["timestamp"].iloc[-1]).tz_convert(NY)
        title = f"{ticker} - {len(summary)} x {TF_LABEL.get(timeframe, timeframe)} bar forecast"
        fig.text(0.012, 0.965, title, ha="left", va="top", fontsize=13,
                 color=c["text_primary"], fontweight="bold")
        as_of_text = as_of.strftime("%a %Y-%m-%d") + ("" if timeframe in ("1d", "1w") else as_of.strftime(" %H:%M New York"))
        fig.text(0.012, 0.912, f"As of {as_of_text} (last close {last:,.2f}) · {model_label}, "
                 f"{n_paths} paths · P(end above last close) {horizon['prob_up']:.0%}",
                 ha="left", va="top", fontsize=8.5, color=c["text_secondary"])
        top = 0.86
        if cutoff_x is not None:
            cutoff = pd.Timestamp(test["cutoff_bar"]).tz_convert(NY)
            fig.text(0.012, 0.878, caption_for_test(test["row"], cutoff, band, timeframe), ha="left", va="top", fontsize=8.5,
                     color=c["text_secondary"])
            top = 0.79
    # legend above the plot so it can never cover prices
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False,
                    ncol=6 if cutoff_x is not None else 4, borderaxespad=0.2, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.tight_layout(rect=(0, 0, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=c["surface"])
    plt.close(fig)
    return path
