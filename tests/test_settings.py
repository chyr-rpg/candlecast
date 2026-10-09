import pytest

from candlecast.settings import PROJECT_ROOT, load_secret, load_settings


def test_default_settings_load():
    s = load_settings()
    assert s.model.name == "base"
    assert s.model.max_context == 512
    assert s.watchlist_csv == PROJECT_ROOT / "config" / "watchlist.csv"
    assert s.default_tickers == ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                                 "DIA", "QQQ", "SPY", "XAUUSD", "BTCUSD")
    assert s.bars_dir == PROJECT_ROOT / "data" / "bars" / "1h"
    assert s.horizons["us_equity_extended"] == 32


def test_model_override_and_mini_rejected():
    assert load_settings(model_name="base").model.repo == "NeoQuasar/Kronos-base"
    with pytest.raises(ValueError):
        load_settings(model_name="mini")


def test_load_secret_env_wins_then_dotenv_and_empty_is_unset(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("A_KEY=from_file\nEMPTY_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("EMPTY_KEY", raising=False)
    assert load_secret("A_KEY", root=tmp_path) == "from_file"
    assert load_secret("EMPTY_KEY", root=tmp_path) is None
    assert load_secret("NOT_THERE", root=tmp_path) is None
    monkeypatch.setenv("A_KEY", "from_env")
    assert load_secret("A_KEY", root=tmp_path) == "from_env"
