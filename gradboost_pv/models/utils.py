from pandas.core.dtypes.common import is_datetime64_dtype
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Union, Tuple


ORDERED_NWP_FEATURE_VARIABLES = [
    "cdcb",
    "lcc",
    "mcc",
    "hcc",
    "sde",
    "hcct",
    "dswrf",
    "dlwrf",
    "h",
    "t",
    "r",
    "dpt",
    "vis",
    "si10",
    "wdir10",
    "prmsl",
    "prate",
]


TRIG_DATETIME_FEATURE_NAMES = [
    "SIN_MONTH",
    "COS_MONTH",
    "SIN_DAY",
    "COS_DAY",
    "SIN_HOUR",
    "COS_HOUR",
]


def _trig_transform(
    values: np.ndarray, period: Union[float, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of values and an upper limit on the values, compute trig decomposition.
    Args:
        values: ndarray of points in the range [0, period]
        period: period of the data
    Returns:
        Decomposition of values into sine and cosine of data with given period
    """

    return np.sin(values * 2 * np.pi / period), np.cos(values * 2 * np.pi / period)


def trigonometric_datetime_transformation(datetimes: npt.ArrayLike) -> np.ndarray:
    """
    Given an iterable of datetimes, returns a trigonometric decomposition on hour, day and month
    Args:
        datetimes: ArrayLike of datetime64 values
    Returns:
        Trigonometric decomposition of datetime into hourly, daily and
        monthly values.
    """
    assert is_datetime64_dtype(
        datetimes
    ), "Data for Trig Decomposition must be np.datetime64 type"

    datetimes = pd.DatetimeIndex(datetimes)
    hour = datetimes.hour.values.reshape(-1, 1) + (
        datetimes.minute.values.reshape(-1, 1) / 60
    )
    day = datetimes.day.values.reshape(-1, 1)
    month = datetimes.month.values.reshape(-1, 1)

    sine_hour, cosine_hour = _trig_transform(hour, 24)
    sine_day, cosine_day = _trig_transform(day, 366)
    sine_month, cosine_month = _trig_transform(month, 12)

    return np.concatenate(
        [sine_month, cosine_month, sine_day, cosine_day, sine_hour, cosine_hour], axis=1
    )


def linear_trend_estimation(data: pd.DataFrame, epsilon=0.01):
    assert all(["x" in data.columns, "y" in data.columns])
    _x, _y = data["x"], data["y"]
    return max(min((1 / ((_x.T @ _x) + epsilon)) * (_x.T @ _y), 10), -10)
