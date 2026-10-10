import json

import pytest

from candlecast import calibration, panel, settings
from tests.test_custom import fake_model  # noqa: F401  (fixture)
from tests.test_timeframes import FakeSource, daily_frame, tf_settings  # noqa: F401  (fixture)


def test_save_secrets_replaces_keys_and_keeps_other_lines(tmp_path):
    (tmp_path / ".env").write_text("OTHER=1\nALPACA_API_KEY_ID=old\n", encoding="utf-8")
    path = settings.save_secrets({"ALPACA_API_KEY_ID": " new ", "ALPACA_API_SECRET_KEY": "s3"}, root=tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["OTHER=1", "ALPACA_API_KEY_ID=new", "ALPACA_API_SECRET_KEY=s3"]
    assert settings.load_secret("ALPACA_API_SECRET_KEY", root=tmp_path) == "s3"


def test_ensure_data_dir_copies_default_config_once(tmp_path):
    settings.ensure_data_dir(tmp_path)
    cfg = tmp_path / "config" / "settings.toml"
    assert cfg.exists() and (tmp_path / "config" / "band_calibration.json").exists()
    cfg.write_text("# user edit\n", encoding="utf-8")
    settings.ensure_data_dir(tmp_path)                       # never overwrites the user's copy
    assert cfg.read_text(encoding="utf-8") == "# user edit\n"


def test_app_window_runs_selects_and_exports(tf_settings, fake_model, tmp_path):  # noqa: F811
    ctk = pytest.importorskip("customtkinter")
    from candlecast import app as appmod

    (tf_settings.root / "config").mkdir(exist_ok=True)
    (tf_settings.root / calibration.CALIBRATION_FILE).write_text(json.dumps(
        {"groups": {"us_equity_extended": {"n_forecasts": 600, "k": {str(L): 1 + L / 40 for L in calibration.LEVELS}}}}))
    src = FakeSource(daily_frame())
    runner = panel.ForecastRunner(tf_settings, log=lambda m: None, make_predictor=lambda: object(),
                                  make_source=lambda inst: src)
    try:
        win = appmod.CandleCastApp(settings=tf_settings, runner=runner)
    except Exception as exc:  # no display available
        pytest.skip(f"no display: {exc}")
    try:
        win.withdraw()
        win.tf.set("1D")
        win.on_timeframe()
        assert win.current_spec() == panel.custom.Spec("1d", 16)
        assert "144 1D bars" in win.req.cget("text")
        insts, _ = panel.resolve_tickers(tf_settings, ["AAPL", "MSFT"])
        win.set_results(runner.run(insts, win.current_spec(), 9, rerun=False))
        win.update()
        assert win._shown is not None and win._shown[1] == 80
        assert "error" in win.card_test[0].cget("text") and "bars" in win.card_data[0].cget("text")
        win.band.set("50%")
        win.show_selected()
        assert win._shown[1] == 50
        win.band.set("Raw")
        win.show_selected()
        assert win._shown[1] == 0 and "raw 80% range" in win.card_forecast[1].cget("text")
        out = win.save_jpg(str(tmp_path / "chart.jpg"))
        from PIL import Image
        with Image.open(out) as im:
            assert im.format == "JPEG" and im.width > 1000
        assert win.set_model("small") and win.settings.model.name == "small"
        assert settings.load_prefs(tf_settings.root)["model"] == "small"
        assert not win.set_model("small") and not win.set_model("large")   # unchanged / not offered
        ctk.set_appearance_mode("dark")
        win.show_selected()                                   # re-renders the dark chart
        assert any(p.name.endswith("_dark.png") for p in (tf_settings.root / "outputs" / "app_charts").iterdir())
    finally:
        ctk.set_appearance_mode("system")
        win.destroy()


def test_prefs_roundtrip(tmp_path):
    assert settings.load_prefs(tmp_path) == {}
    settings.save_prefs({"model": "small"}, root=tmp_path)
    assert settings.save_prefs({"theme": "dark"}, root=tmp_path) == {"model": "small", "theme": "dark"}
    (tmp_path / settings.PREFS_FILE).write_text("not json", encoding="utf-8")
    assert settings.load_prefs(tmp_path) == {}


def test_ensure_console_streams_replaces_missing_streams(monkeypatch):
    import sys

    from candlecast import kronos_loader
    monkeypatch.setattr(sys, "stderr", None)
    kronos_loader.ensure_console_streams()
    sys.stderr.write("progress bar output goes nowhere\n")      # would raise AttributeError before


def test_partial_model_cache_is_not_used(tmp_path, monkeypatch):
    """A crashed download can leave a cache snapshot with only README.md; it must be re-downloaded."""
    import huggingface_hub

    from candlecast import kronos_loader
    partial = tmp_path / "hf_snapshot"
    partial.mkdir()
    (partial / "README.md").write_text("x", encoding="utf-8")
    calls = []

    def fake_snapshot(repo, local_files_only=False, local_dir=None, tqdm_class=None):
        calls.append(local_files_only)
        if local_files_only:
            return str(partial)
        (tmp_path / "out").mkdir(exist_ok=True)
        from pathlib import Path
        Path(local_dir, "config.json").write_text("{}", encoding="utf-8")
        Path(local_dir, "model.safetensors").write_bytes(b"0")
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(kronos_loader, "MODELS_DIR", tmp_path / "models")
    path = kronos_loader.local_model_dir("Owner/Model")
    assert calls == [True, False] and path.endswith("Owner--Model")
    assert kronos_loader.local_model_dir("Owner/Model") == path and calls == [True, False]   # marker reused
