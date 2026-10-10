import json
from dataclasses import replace

import pytest

from candlecast import calibration, emailer, kronos_loader, panel, timeframes
from tests.test_custom import fake_model  # noqa: F401  (fixture)
from tests.test_timeframes import FakeSource, daily_frame, tf_settings  # noqa: F401  (fixture)


def test_provider_for_known_and_unknown_domains():
    assert emailer.provider_for("me@gmail.com")[:3] == ("smtp.gmail.com", 587, "starttls")
    assert emailer.provider_for("x@QQ.com")[:3] == ("smtp.qq.com", 465, "ssl")
    assert emailer.provider_for("a@example.org")[:3] == ("smtp.example.org", 587, "starttls")
    cfg = emailer.EmailConfig(address="a@example.org", server="mail.example.org:465")
    assert cfg.smtp() == ("mail.example.org", 465, "ssl")
    assert emailer.EmailConfig(address="a@b.c").recipient == "a@b.c"


def test_config_and_password_roundtrip(tmp_path):
    cfg = emailer.EmailConfig(address="me@gmail.com", tickers=["AAPL", "BTCUSD"], timeframe="4h", bars=12,
                              time="07:30", enabled=True)
    emailer.save_config(cfg, root=tmp_path)
    assert emailer.load_config(tmp_path) == cfg
    emailer.store_password("abcd efgh ijkl mnop", root=tmp_path)
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "abcdefgh" not in env                              # stored encrypted, not in clear text
    assert emailer.load_password(tmp_path) == "abcdefghijklmnop"   # spaces from Google's display removed


def test_task_xml_runs_report_daily():
    xml = emailer._task_xml("07:05")
    assert "T07:05:00" in xml and "--email-report" in xml and "<StartWhenAvailable>true" in xml


def test_offline_mode_never_touches_the_network(tf_settings):  # noqa: F811
    class Boom:
        def fetch(self, *a, **k):
            raise AssertionError("network used in offline mode")

    inst = panel.resolve_tickers(tf_settings, ["AAPL"])[0][0]
    timeframes.write_cache(timeframes.bars_dir(tf_settings, "1d"), "AAPL", daily_frame())
    bars, note = timeframes.load_bars(replace(tf_settings, offline=True), inst, "1d", 50, source=Boom())
    assert len(bars) > 50 and "offline mode" in note


def test_offline_model_not_downloaded_raises(tmp_path, monkeypatch):
    import huggingface_hub

    def no_cache(repo, local_files_only=False, **_):
        if local_files_only:
            raise FileNotFoundError
        raise AssertionError("download attempted offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_cache)
    monkeypatch.setattr(kronos_loader, "MODELS_DIR", tmp_path / "models")
    with pytest.raises(RuntimeError, match="Online once"):
        kronos_loader.local_model_dir("Owner/Model", offline=True)


def test_custom_model_folder_is_used_directly(tmp_path):
    folder = tmp_path / "kronos-large"
    folder.mkdir()
    with pytest.raises(RuntimeError, match="not a Kronos model folder"):
        kronos_loader.local_model_dir(str(folder))
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"0")
    assert kronos_loader.local_model_dir(str(folder), offline=True) == str(folder)


def test_report_is_built_and_sent(tf_settings, fake_model, monkeypatch):  # noqa: F811
    (tf_settings.root / "config").mkdir(exist_ok=True)
    (tf_settings.root / calibration.CALIBRATION_FILE).write_text(json.dumps(
        {"groups": {"us_equity_extended": {"n_forecasts": 600, "k": {str(L): 1 + L / 40 for L in calibration.LEVELS}}}}))
    src = FakeSource(daily_frame())
    runner = panel.ForecastRunner(tf_settings, log=lambda m: None, make_predictor=lambda: object(),
                                  make_source=lambda inst: src)
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["server"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, msg):
            sent["msg"] = msg

    monkeypatch.setattr(emailer.smtplib, "SMTP", FakeSMTP)
    cfg = emailer.EmailConfig(address="me@gmail.com", tickers=["AAPL", "MSFT"], timeframe="1d", bars=16,
                              paths=9)
    status = emailer.run_report(tf_settings, cfg, password="pw", runner=runner, log_fn=lambda m: None)
    msg = sent["msg"]
    assert sent["server"] == ("smtp.gmail.com", 587) and sent["tls"] and sent["login"] == ("me@gmail.com", "pw")
    assert msg["To"] == "me@gmail.com" and "AAPL" in msg["Subject"] and "2 of 2" in status
    html = msg.get_body(preferencelist=("html",)).get_content()
    assert "AAPL" in html and "cid:" in html and "no-change" in html
    images = [p for p in msg.walk() if p.get_content_type() == "image/jpeg"]
    assert len(images) == 2 and len(images[0].get_payload(decode=True)) > 10_000


def test_bad_login_explained():
    import smtplib

    msg = emailer.explain_error(smtplib.SMTPAuthenticationError(535, b"bad"), emailer.EmailConfig(address="a@gmail.com"))
    assert "app password" in msg
