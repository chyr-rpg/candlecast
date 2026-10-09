"""Loads real Kronos weights from the Hugging Face cache; run ``setup.ps1`` first."""

import numpy as np
import pandas as pd
import pytest

from candlecast.cli import synthetic_bars
from candlecast.kronos_loader import predictor_for
from candlecast.settings import load_settings


@pytest.mark.slow
@pytest.mark.parametrize("name", ["small", "base"])
def test_forecast_shape_and_finiteness(name):
    model = load_settings(model_name=name).model
    bars = synthetic_bars(128)
    y_ts = pd.Series(pd.date_range(bars["timestamp"].iloc[-1] + pd.Timedelta(hours=1), periods=4, freq="h"))
    pred = predictor_for(model).predict(
        df=bars[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=bars["timestamp"],
        y_timestamp=y_ts,
        pred_len=4,
        sample_count=1,
        verbose=False,
    )
    assert list(pred.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert len(pred) == 4
    assert np.isfinite(pred.to_numpy()).all()
    # A 4-hour forecast of a ~100-priced series shouldn't land an order of magnitude away.
    assert (pred["close"] > 50).all() and (pred["close"] < 200).all()
