from candlecast.charts import thin_ticks


def test_thin_ticks_keeps_evenly_spaced_labels():
    days = list(range(0, 120, 4))                 # 4H equity bars: a new day every 4 bars
    kept = thin_ticks(days, min_gap=8)
    assert kept[-1] == 116 and len(kept) >= 10     # used to collapse to a single label
    assert all(b - a >= 8 for a, b in zip(kept, kept[1:]))
    assert thin_ticks([0, 30, 60], min_gap=8) == [0, 30, 60]
    assert thin_ticks([], min_gap=8) == []
