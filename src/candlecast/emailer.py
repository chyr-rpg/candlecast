"""Daily forecast e-mail: settings, password storage, SMTP sending, the report itself and the Windows
scheduled task that triggers it.

Mail is sent from the user's own mailbox over SMTP (nothing goes through a CandleCast server, because
there is none). The SMTP server is picked from the address's domain; the password is encrypted with
Windows DPAPI (only this Windows account can decrypt it) and kept in the data folder's .env.
"""

from __future__ import annotations

import base64
import logging
import os
import smtplib
import ssl
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from email.message import EmailMessage
from email.utils import make_msgid
from html import escape
from pathlib import Path

import pandas as pd

from candlecast import timeframes
from candlecast.settings import DATA_DIR, FROZEN, load_prefs, load_secret, save_prefs, save_secrets

log = logging.getLogger(__name__)

PASSWORD_KEY = "CANDLECAST_SMTP_PASSWORD"
TASK_NAME = "CandleCast Daily Email"

# domain -> (server, port, "ssl" | "starttls", where to create an app password)
PROVIDERS = {
    "gmail.com": ("smtp.gmail.com", 587, "starttls", "https://myaccount.google.com/apppasswords"),
    "googlemail.com": ("smtp.gmail.com", 587, "starttls", "https://myaccount.google.com/apppasswords"),
    "outlook.com": ("smtp-mail.outlook.com", 587, "starttls", "https://account.live.com/proofs/AppPassword"),
    "hotmail.com": ("smtp-mail.outlook.com", 587, "starttls", "https://account.live.com/proofs/AppPassword"),
    "live.com": ("smtp-mail.outlook.com", 587, "starttls", "https://account.live.com/proofs/AppPassword"),
    "yahoo.com": ("smtp.mail.yahoo.com", 465, "ssl", "https://login.yahoo.com/account/security"),
    "icloud.com": ("smtp.mail.me.com", 587, "starttls", "https://account.apple.com"),
    "me.com": ("smtp.mail.me.com", 587, "starttls", "https://account.apple.com"),
    "qq.com": ("smtp.qq.com", 465, "ssl", "https://mail.qq.com"),
    "163.com": ("smtp.163.com", 465, "ssl", "https://mail.163.com"),
    "126.com": ("smtp.126.com", 465, "ssl", "https://mail.126.com"),
}


def provider_for(address: str) -> tuple[str, int, str, str | None]:
    """SMTP server for an e-mail address (unknown domains: smtp.<domain>:587 with STARTTLS)."""
    domain = address.rsplit("@", 1)[-1].strip().lower()
    if domain in PROVIDERS:
        return PROVIDERS[domain]
    return f"smtp.{domain}", 587, "starttls", None


@dataclass
class EmailConfig:
    address: str = ""                 # sender mailbox (also the login)
    to: str = ""                      # recipient; empty = same as address
    tickers: list[str] = field(default_factory=list)
    timeframe: str = "1d"
    bars: int = 16
    history: int = timeframes.DEFAULT_HISTORY
    paths: int = 100
    band_level: int = 80
    time: str = "08:00"               # local time, daily
    enabled: bool = False
    server: str = ""                  # optional override "host:port"

    @property
    def recipient(self) -> str:
        return self.to.strip() or self.address.strip()

    def smtp(self) -> tuple[str, int, str]:
        if self.server.strip():
            host, _, port = self.server.strip().partition(":")
            port = int(port or 587)
            return host, port, "ssl" if port == 465 else "starttls"
        host, port, mode, _ = provider_for(self.address)
        return host, port, mode


def load_config(root: Path | None = None) -> EmailConfig:
    raw = load_prefs(root).get("email") or {}
    known = {k: v for k, v in raw.items() if k in EmailConfig.__dataclass_fields__}
    return EmailConfig(**known)


def save_config(cfg: EmailConfig, root: Path | None = None) -> None:
    save_prefs({"email": asdict(cfg)}, root=root)


# ------------------------------------------------------------------ password (Windows DPAPI)

def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    inp = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out = Blob()
    fn = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    if not fn(ctypes.byref(inp), None, None, None, None, 0x1, ctypes.byref(out)):  # 0x1: no UI
        raise OSError("Windows could not " + ("encrypt" if protect else "decrypt") + " the e-mail password")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)


def store_password(password: str, root: Path | None = None) -> None:
    raw = password.replace(" ", "").encode("utf-8")   # app passwords are shown with spaces
    value = ("dpapi:" + base64.b64encode(_dpapi(raw, True)).decode()) if os.name == "nt" else \
        ("plain:" + base64.b64encode(raw).decode())
    save_secrets({PASSWORD_KEY: value}, root=root)


def load_password(root: Path | None = None) -> str | None:
    value = load_secret(PASSWORD_KEY, root=root)
    if not value:
        return None
    kind, _, payload = value.partition(":")
    data = base64.b64decode(payload)
    return (_dpapi(data, False) if kind == "dpapi" else data).decode("utf-8")


# ------------------------------------------------------------------ sending

def send(cfg: EmailConfig, password: str, msg: EmailMessage, timeout: int = 60) -> None:
    host, port, mode = cfg.smtp()
    context = ssl.create_default_context()
    if mode == "ssl":
        server = smtplib.SMTP_SSL(host, port, timeout=timeout, context=context)
    else:
        server = smtplib.SMTP(host, port, timeout=timeout)
    with server:
        if mode != "ssl":
            server.starttls(context=context)
        server.login(cfg.address.strip(), password)
        server.send_message(msg)


def explain_error(exc: Exception, cfg: EmailConfig) -> str:
    """A readable reason for a failed send."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        host = cfg.smtp()[0]
        if "gmail" in host:
            return ("Gmail rejected the login. Gmail only accepts an *app password* here (16 letters, made at "
                    "myaccount.google.com/apppasswords - needs 2-Step Verification), not your normal password.")
        if "outlook" in host:
            return ("Outlook rejected the login. Microsoft has turned off password sign-in for SMTP on most "
                    "personal accounts; a Gmail address is the easiest alternative.")
        return f"{host} rejected the login - check the address and (app) password."
    return f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------------ the report

def _summary_row(settings, res, level: int) -> dict:
    from candlecast.pipeline import band_for  # noqa: PLC0415

    row, test = res.row, res.test
    last, p10, p50, p90 = row["last_close"], row["end_close_p10"], row["end_close_p50"], row["end_close_p90"]
    test_summary = None
    if test and (settings.root / test["summary_path"]).exists():
        test_summary = pd.read_parquet(settings.root / test["summary_path"])
    band = band_for(settings, row["asset_class"], level, test_summary)
    if band:
        k = band["k"]
        rng = f"{p50 - k * (p50 - p10):,.2f} - {p50 + k * (p90 - p50):,.2f} ({band['level']}%)"
    else:
        rng = f"{p10:,.2f} - {p90:,.2f} (raw 80%)"
    out = {"Ticker": row["ticker"], "Last close": f"{last:,.2f}",
           "Median end": f"{p50:,.2f} ({p50 / last - 1:+.1%})", "Range": rng,
           "P(higher)": f"{row['prob_up']:.0%}", "Test window": "-"}
    if test:
        better = test["abs_err_kronos"] <= test["abs_err_naive"]
        out["Test window"] = (f"{test['abs_err_kronos']:.1%} error vs {test['abs_err_naive']:.1%} no-change "
                              f"({'better' if better else 'worse'})")
    return out


def build_report(settings, cfg: EmailConfig, results, chart_dir: Path) -> EmailMessage:
    """E-mail with a summary table and one inline chart per forecast (JPG, also attached)."""
    from candlecast import custom  # noqa: PLC0415
    from candlecast.panel import save_as_jpg  # noqa: PLC0415

    chart_dir.mkdir(parents=True, exist_ok=True)
    label = f"{cfg.bars} x {timeframes.LABELS.get(cfg.timeframe, cfg.timeframe)}"
    ok = [r for r in results if r.row is not None]
    failed = [r for r in results if r.row is None]
    msg = EmailMessage()
    msg["Subject"] = f"CandleCast daily forecast - {', '.join(r.ticker for r in ok) or 'no results'} ({label})"
    msg["From"] = cfg.address.strip()
    msg["To"] = cfg.recipient

    rows = [_summary_row(settings, r, cfg.band_level) for r in ok]
    text = [f"CandleCast forecast, {label} bars, {pd.Timestamp.now():%Y-%m-%d %H:%M}", ""]
    text += [" | ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
    text += [f"FAILED {r.ticker}: {r.error}" for r in failed]
    text += ["", "Shared for learning and information only - not investment advice."]
    msg.set_content("\n".join(text))

    charts = []
    for r in ok:
        png, jpg = chart_dir / f"{r.ticker}.png", chart_dir / f"{r.ticker}.jpg"
        custom.render_custom(settings, r.row, r.test, png, cfg.band_level)
        save_as_jpg(png, jpg, quality=88)
        charts.append((r.ticker, jpg, make_msgid(domain="candlecast.local")))

    th = "padding:6px 10px;text-align:left;border-bottom:1px solid #ddd;font-size:13px"
    head = "".join(f"<th style='{th}'>{escape(k)}</th>" for k in (rows[0] if rows else {}))
    body_rows = "".join("<tr>" + "".join(f"<td style='{th}'>{escape(str(v))}</td>" for v in row.values()) + "</tr>"
                        for row in rows)
    fails = "".join(f"<p style='color:#b42318'>Failed: {escape(r.ticker)} - {escape(str(r.error))}</p>"
                    for r in failed)
    imgs = "".join(f"<h3 style='font-family:Segoe UI,Arial'>{escape(t)}</h3>"
                   f"<img src='cid:{cid[1:-1]}' style='max-width:100%;border:1px solid #eee'>" for t, _, cid in charts)
    html = (f"<div style='font-family:Segoe UI,Arial,sans-serif;color:#222'>"
            f"<h2>CandleCast forecast - {escape(label)} bars</h2>"
            f"<p style='color:#666'>{pd.Timestamp.now():%A %Y-%m-%d %H:%M} - median end close, uncertainty range, "
            f"and how the same setup did on the last {cfg.bars} bars (test window).</p>"
            f"<table style='border-collapse:collapse'><tr>{head}</tr>{body_rows}</table>{fails}{imgs}"
            f"<p style='color:#888;font-size:12px'>Shared for learning and information only - not investment advice. "
            f"Sent by CandleCast from your own mailbox.</p></div>")
    msg.add_alternative(html, subtype="html")
    html_part = msg.get_payload()[1]
    for ticker, jpg, cid in charts:
        html_part.add_related(jpg.read_bytes(), "image", "jpeg", cid=cid, filename=f"{ticker}.jpg")
    return msg


def run_report(settings, cfg: EmailConfig | None = None, password: str | None = None, log_fn=None,
               runner=None) -> str:
    """Forecast the configured tickers and e-mail the result. Returns a one-line status; raises on failure."""
    from candlecast import custom  # noqa: PLC0415
    from candlecast.panel import ForecastRunner, resolve_tickers  # noqa: PLC0415

    cfg = cfg or load_config(settings.root)
    log_fn = log_fn or log.info
    if not cfg.address or "@" not in cfg.address:
        raise ValueError("No e-mail address set (Daily e-mail settings).")
    if not cfg.tickers:
        raise ValueError("No tickers set for the daily e-mail.")
    password = password or load_password(settings.root)
    if not password:
        raise ValueError("No e-mail password saved (Daily e-mail settings).")
    instruments, unmapped = resolve_tickers(settings, cfg.tickers)
    if unmapped:
        log_fn(f"E-mail report: no data source for {', '.join(unmapped)}")
    runner = runner or ForecastRunner(settings, log=log_fn)
    spec = custom.Spec(cfg.timeframe, int(cfg.bars), int(cfg.history))
    results = runner.run(instruments, spec, int(cfg.paths), rerun=False)
    chart_dir = settings.root / "outputs" / "email" / pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    msg = build_report(settings, cfg, results, chart_dir)
    try:
        send(cfg, password, msg)
    except Exception as exc:
        raise RuntimeError(explain_error(exc, cfg)) from exc
    ok = sum(r.row is not None for r in results)
    status = f"E-mail sent to {cfg.recipient}: {ok} of {len(results)} forecasts."
    log_fn(status)
    return status


# ------------------------------------------------------------------ Windows scheduled task

def task_command() -> tuple[str, str]:
    """(program, arguments) that run the e-mail report without a window."""
    if FROZEN:
        return sys.executable, "--email-report"
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    return str(pythonw if pythonw.exists() else exe), "-m candlecast.app --email-report"


def _task_xml(time: str) -> str:
    program, args = task_command()
    hh, mm = (int(x) for x in time.split(":"))
    start = f"2026-01-01T{hh:02d}:{mm:02d}:00"
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>CandleCast: forecast the chosen tickers and e-mail the charts.</Description></RegistrationInfo>
  <Triggers><CalendarTrigger><StartBoundary>{start}</StartBoundary><Enabled>true</Enabled>
    <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger></Triggers>
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <WakeToRun>true</WakeToRun>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author"><Exec><Command>{escape(program)}</Command><Arguments>{escape(args)}</Arguments>
    <WorkingDirectory>{escape(str(DATA_DIR))}</WorkingDirectory></Exec></Actions>
</Task>
"""


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks", *args], capture_output=True, text=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def schedule(time: str) -> None:
    """Create/replace the daily task (runs while you are signed in; catches up after sleep/shutdown)."""
    xml = DATA_DIR / "logs" / "email_task.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    xml.write_text(_task_xml(time), encoding="utf-16")
    r = _schtasks("/Create", "/F", "/TN", TASK_NAME, "/XML", str(xml))
    if r.returncode != 0:
        raise RuntimeError(f"Could not create the scheduled task: {(r.stderr or r.stdout).strip()}")


def unschedule() -> None:
    _schtasks("/Delete", "/F", "/TN", TASK_NAME)


def is_scheduled() -> bool:
    return _schtasks("/Query", "/TN", TASK_NAME).returncode == 0
