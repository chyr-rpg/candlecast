import pandas as pd
import pytest

from candlecast.calendars import future_bars, to_model_time


def ny(s):
    return pd.Timestamp(s, tz="America/New_York")


def ny_list(idx):
    return [t.strftime("%a %Y-%m-%d %H:%M") for t in idx.tz_convert("America/New_York")]


def test_equity_friday_evening_rolls_to_monday_premarket():
    got = ny_list(future_bars("us_equity_extended", ny("2026-10-02 19:00"), 18))
    assert got[0] == "Mon 2026-10-05 04:00"
    assert got[15] == "Mon 2026-10-05 19:00"   # 16 bars per full day
    assert got[16] == "Tue 2026-10-06 04:00"


def test_equity_skips_holiday_and_shortens_early_close():
    # Thanksgiving 2026-11-26 closed; 2026-11-27 closes 13:00 => after-hours until 17:00
    got = ny_list(future_bars("us_equity_extended", ny("2026-11-25 19:00"), 14))
    assert got[0] == "Fri 2026-11-27 04:00"
    assert got[12] == "Fri 2026-11-27 16:00"   # 04..16 = 13 bars
    assert got[13] == "Mon 2026-11-30 04:00"


def test_equity_mid_session_continues_same_day():
    got = ny_list(future_bars("us_equity_extended", ny("2026-10-06 10:00"), 2))
    assert got == ["Tue 2026-10-06 11:00", "Tue 2026-10-06 12:00"]


def test_crypto_every_hour_through_weekend():
    got = future_bars("crypto", pd.Timestamp("2026-10-03 23:00", tz="UTC"), 48)
    assert len(got) == 48 and (got.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_fx_closed_friday_17_to_sunday_17():
    got = ny_list(future_bars("fx", ny("2026-10-02 15:00"), 3))
    assert got == ["Fri 2026-10-02 16:00", "Sun 2026-10-04 17:00", "Sun 2026-10-04 18:00"]


def test_commodity_daily_break_and_sunday_18_open():
    assert ny_list(future_bars("commodity", ny("2026-10-06 15:00"), 3)) == [
        "Tue 2026-10-06 16:00", "Tue 2026-10-06 18:00", "Tue 2026-10-06 19:00"]
    assert ny_list(future_bars("commodity", ny("2026-10-02 16:00"), 1)) == ["Sun 2026-10-04 18:00"]


def test_dst_end_has_no_repeated_or_missing_utc_hour():
    # US DST ends 2026-11-01 02:00 local; UTC bars must stay exactly 1h apart
    got = future_bars("fx", pd.Timestamp("2026-11-01 04:00", tz="UTC"), 6)
    assert (got.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_model_time_is_naive_new_york():
    s = to_model_time(pd.DatetimeIndex([pd.Timestamp("2026-07-01 13:00", tz="UTC")]))
    assert s.iloc[0] == pd.Timestamp("2026-07-01 09:00") and s.dt.tz is None


@pytest.mark.parametrize("cls", ["crypto", "fx", "commodity", "us_equity_extended"])
def test_strictly_after_last_bar_and_on_the_hour(cls):
    last = pd.Timestamp("2026-10-07 14:00", tz="UTC")
    got = future_bars(cls, last, 10)
    assert got[0] > last and (got.minute == 0).all() and got.is_monotonic_increasing
