from datetime import datetime

import pandas as pd

from gradboost_pv.inference.utils import (
    calculate_azimuth_and_elevation_angle,
    filter_forecasts_on_sun_elevation,
)


def test_calculate_azimuth_and_elevation_angle():
    """Test calculate_azimuth_and_elevation_angle"""
    datestamps = pd.date_range("2021-06-22 12:00:00", "2021-06-23", freq="5T", tz="UTC")

    s = calculate_azimuth_and_elevation_angle(
        longitude=0, latitude=51, datestamps=datestamps.to_pydatetime()
    )

    assert len(s) == len(datestamps)
    assert "azimuth" in s.columns
    assert "elevation" in s.columns

    # midday sun at 12 oclock on mid summer, middle of the sky, and in london at around 62 degrees
    # https://diamondgeezer.blogspot.com/2017/12/solar-elevation.html
    assert 170 < s["azimuth"][0] < 190
    assert 60 < s["elevation"][0] < 65


def test_filter_forecasts_on_sun_elevation(forecasts):
    # night time
    forecasts[0].forecast_values[0].target_time = datetime(2022, 1, 1)
    forecasts[0].forecast_values[0].expected_power_generation_megawatts = 1
    forecasts[0].forecast_values[0].properties = {"10": 90, "90": 1000}

    # day time
    forecasts[0].forecast_values[1].target_time = datetime(2022, 1, 1, 12)
    forecasts[0].forecast_values[1].expected_power_generation_megawatts = 1
    forecasts[0].forecast_values[1].properties = {"10": 90, "90": 1000}

    forecasts[-1].location.gsp_id = 338

    _ = filter_forecasts_on_sun_elevation(forecasts)

    assert forecasts[0].forecast_values[0].expected_power_generation_megawatts == 0
    assert forecasts[0].forecast_values[0].properties == {"10": 0, "90": 0}
    assert forecasts[0].forecast_values[1].expected_power_generation_megawatts == 1
    assert forecasts[0].forecast_values[1].properties == {"10": 90, "90": 1000}
