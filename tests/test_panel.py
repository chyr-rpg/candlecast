import json

import pytest

from candlecast import calibration, panel
from tests.test_custom import fake_model  # noqa: F401  (fixture)
from tests.test_timeframes import FakeSource, daily_frame, tf_settings  # noqa: F401  (fixture)


def test_parse_tickers_splits_dedupes_and_rejects_junk():
    valid, invalid = panel.parse_tickers(" aapl, NVDA;spy  brk.b\nAAPL  $$$ ")
    assert valid == ["AAPL", "NVDA", "SPY", "BRK.B"] and invalid == ["$$$"]
    assert panel.parse_tickers("") == ([], [])


def test_resolve_tickers_maps_symbol_map_and_us_stocks(tf_settings):  # noqa: F811
    ok, unmapped = panel.resolve_tickers(tf_settings, ["AAPL", "BTCUSD", "XAUUSD"])
    by = {i.ticker: i for i in ok}
    assert by["AAPL"].source == "alpaca_stock" and by["AAPL"].asset_class == "us_equity_extended"
    assert by["BTCUSD"].asset_class == "crypto" and by["XAUUSD"].source == "yfinance"
    assert unmapped == []


def test_choices_and_requirement_text():
    assert [panel.band_level_from_choice(c) for c in panel.BAND_CHOICES] == [40, 50, 60, 70, 80, 0]
    assert [panel.timeframe_from_choice(c) for c in panel.TIMEFRAME_CHOICES] == ["1h", "4h", "1d", "1w"]
    text = panel.requirement_text("1d", 16)
    assert "128 history bars + 16 test bars = 144 1D bars" in text and "months" in text


def _runner(settings, frame=None):
    logs = []
    src = FakeSource(frame if frame is not None else daily_frame())
    return panel.ForecastRunner(settings, log=logs.append, make_predictor=lambda: object(),
                                make_source=lambda inst: src), logs


def test_runner_and_describe(tf_settings, fake_model):  # noqa: F811
    insts, _ = panel.resolve_tickers(tf_settings, ["AAPL", "MSFT"])
    runner, logs = _runner(tf_settings)
    spec = panel.custom.Spec("1d", 16)
    res = {r.ticker: r for r in runner.run(insts, spec, 9, rerun=False)}
    assert res["AAPL"].status == "new" and res["AAPL"].test is not None
    assert any("AAPL 16 x 1D: done" in m for m in logs)
    text = panel.describe(res["AAPL"].row, res["AAPL"].test)
    assert "16 x 1D bars" in text and "128 history bars" in text and "raw Kronos 80% range" in text
    band = {"level": 60, "k": 2.0, "test_coverage": 0.5}
    text = panel.describe(res["AAPL"].row, res["AAPL"].test, band, calibrated_on=600)
    assert "calibrated 60% range" in text and "50% of actual closes inside its calibrated 60% band" in text
    assert "approximate" in text                                    # not the 1H x 32 calibration spec
    assert runner.run(insts[:1], spec, 9, rerun=False)[0].status == "cached"


def test_panel_window_timeframe_band_and_jpg(tf_settings, fake_model, tmp_path):  # noqa: F811
    tk = pytest.importorskip("tkinter")
    (tf_settings.root / "config").mkdir(exist_ok=True)
    (tf_settings.root / calibration.CALIBRATION_FILE).write_text(json.dumps(
        {"groups": {"us_equity_extended": {"n_forecasts": 600, "k": {str(L): 1 + L / 40 for L in calibration.LEVELS}}}}))
    runner, _ = _runner(tf_settings)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    try:
        root.withdraw()
        app = panel.PanelApp(root, settings=tf_settings, runner=runner)
        app.tf_var.set("1D")
        app.on_timeframe()
        assert app.bars_var.get() == 16 and "144 1D bars" in app.req_var.get()
        insts, _ = panel.resolve_tickers(tf_settings, ["AAPL"])
        app.set_results(runner.run(insts, app.current_spec(), 9, rerun=False))
        root.update()
        app.show_selected()
        assert app._photo is not None and app.listbox.get(0).startswith("AAPL    16x1D")
        assert app._shown[2] == 80 and "_1d_16b_" in app._shown[1].name
        app.band_var.set("50%")
        app.show_selected()
        assert app._shown[2] == 50 and app._shown[1].exists()
        out = app.save_jpg(str(tmp_path / "chart.jpg"))
        from PIL import Image
        with Image.open(out) as im:
            assert im.format == "JPEG" and im.width > 1000
    finally:
        root.destroy()


def test_slow_ticker_warns_and_records_stacks(tmp_path, monkeypatch):
    import time as _time

    from types import SimpleNamespace
    msgs = []
    runner = panel.ForecastRunner.__new__(panel.ForecastRunner)
    runner.settings = SimpleNamespace(root=tmp_path)
    runner.log = msgs.append
    runner.SLOW_AFTER_S = 1
    done = runner._watch("AMZN")
    _time.sleep(2.5)
    done()
    assert any("AMZN: still working" in m for m in msgs)
    text = (tmp_path / "logs" / "slow_run_stacks.log").read_text(encoding="utf-8")
    assert "AMZN started" in text and "File" in text          # thread stacks were written


def test_alpaca_session_gets_default_timeout():
    from candlecast import sources

    seen = {}

    class Session:
        def request(self, method, url, **kwargs):
            seen.update(kwargs)

    class Client:
        _session = Session()

    client = sources._with_timeout(Client())
    client._session.request("GET", "https://example.invalid")
    assert seen["timeout"] == sources.HTTP_TIMEOUT
    assert sources._with_timeout(client) is client                # wrapping twice is harmless
