import json

import pandas as pd
import pytest

from candlecast import calibration


CAL = {"groups": {"us_equity_extended": {"n_forecasts": 600, "k": {"40": 1.0, "80": 3.0}},
                  "other": {"n_forecasts": 40, "k": {"40": 1.5, "80": 5.0}}}}


def test_factor_falls_back_to_other_group_and_missing_is_none(tmp_path):
    assert calibration.factor(CAL, "us_equity_extended", 80) == 3.0
    assert calibration.factor(CAL, "crypto", 80) == 5.0           # no crypto group -> "other"
    assert calibration.group_size(CAL, "crypto") == 40
    assert calibration.factor({"groups": {}}, "crypto", 80) is None
    assert calibration.load(tmp_path) == {"groups": {}}           # no file -> no calibration
    (tmp_path / "config").mkdir()
    (tmp_path / calibration.CALIBRATION_FILE).write_text(json.dumps(CAL))
    assert calibration.load(tmp_path)["groups"]["other"]["k"]["40"] == 1.5


def test_band_widens_each_side_around_the_median_and_coverage_counts_actuals():
    s = pd.DataFrame({"close_p10": [98.0, 96.0], "close_p50": [100.0, 100.0], "close_p90": [101.0, 103.0],
                      "actual_close": [94.5, 108.9]})
    lo, hi = calibration.band(s, 3.0)
    assert lo.tolist() == [94.0, 88.0] and hi.tolist() == [103.0, 109.0]
    assert calibration.coverage(s, 3.0) == 1.0
    assert calibration.coverage(s, 1.0) == 0.0                     # raw band misses both
    assert calibration.coverage(s.drop(columns="actual_close"), 3.0) is None
