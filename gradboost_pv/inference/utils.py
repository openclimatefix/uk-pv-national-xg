""" functions to filter forecasts on the sun """
import logging
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import pvlib
from nowcasting_datamodel.models import ForecastSQL

ELEVATION_LIMIT = 0
logger = logging.getLogger(__name__)


def filter_forecasts_on_sun_elevation(forecasts: List[ForecastSQL]) -> List[ForecastSQL]:
    """
    Filters predictions if the sun elevation is more than X degrees below the horizon

    Args:
        forecasts: The forecast output

    Returns:
        forecast with zeroed out times
    """

    logger.info(f"Filtering forecasts on sun elevation. {ELEVATION_LIMIT=}")

    for forecast in forecasts:
        lat = 51.7612
        lon = -1.2465
        target_times = [
            forecast_value.target_time.replace(tzinfo=timezone.utc)
            for forecast_value in forecast.forecast_values
        ]

        # get a pandas dataframe of of elevation and azimuth positions
        sun_df = calculate_azimuth_and_elevation_angle(
            latitude=lat, longitude=lon, datestamps=target_times
        )

        # check that any elevations are < 'ELEVATION_LIMIT'
        if (sun_df["elevation"] < ELEVATION_LIMIT).sum() > 0:
            logger.debug(f"Got sun angle for {lat} {lon} {target_times}, and some are below zero")

            # loop through target times
            for i in range(len(target_times)):
                sun_for_target_time = sun_df.iloc[i]
                if sun_for_target_time.elevation < ELEVATION_LIMIT:
                    # note sql objects are connected, so we can edit in place
                    forecast.forecast_values[i].expected_power_generation_megawatts = 0

        else:
            logger.debug("No elevations below zero, so need to filter")

    logger.info("Done sun filtering")

    return forecasts


def calculate_azimuth_and_elevation_angle(
    latitude: float, longitude: float, datestamps: list[datetime]
) -> pd.DataFrame:
    """
    Calculation the azimuth angle, and the elevation angle for several datetamps.

    But for one specific osgb location

    More details see:
    https://www.celestis.com/resources/faq/what-are-the-azimuth-and-elevation-of-a-satellite/

    Args:
        latitude: latitude of the pv site
        longitude: longitude of the pv site
        datestamps: list of datestamps to calculate the sun angles. i.e the sun moves from east to
            west in the day.

    Returns: Pandas data frame with the index the same as 'datestamps', with columns of
    "elevation" and "azimuth" that have been calculate.

    """
    # get the solor position
    solpos = pvlib.solarposition.get_solarposition(datestamps, latitude, longitude)

    # extract the information we want
    return solpos[["elevation", "azimuth"]]


def smooth_results(results_df: pd.Series):
    """
    Smooth results by taking the averaging across 3 of data hours

    We take use the weights [25%, 50%, 25%] to smooth the data.
    At the edges we use the last values to fill in the missing data.
    At the edges we could use the edge and next edge,
    but this might lead to some biased results, like pulling the result up if in the morning.

    """

    # use rolling average
    weights = np.array([0.25, 0.5, 0.25])
    smoothed_df = results_df.rolling(window=3, center=True).apply(lambda x: np.sum(weights * x))

    # adjust the first and last values
    smoothed_df.iloc[0] = results_df.iloc[0]
    smoothed_df.iloc[-1] = results_df.iloc[-1]

    # if initial value is zero, keep it zero
    zero_mask = results_df == 0
    smoothed_df[zero_mask] = 0

    return smoothed_df
