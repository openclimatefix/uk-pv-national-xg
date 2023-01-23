from pandas.core.dtypes.common import is_datetime64_dtype
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Union, Tuple


NWP_VARIABLE_NUM = 17
NWP_STEP_HORIZON = 37
NWP_FPATH = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr/"
)
GSP_FPATH = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"


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

DEFAULT_ROLLING_LR_WINDOW_SIZE = 10


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


def clipped_univariate_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    upper_clip: float = 10,
    lower_clip: float = -10,
    epsilon: float = 0.01,
) -> float:
    return max(min((1 / ((X.T @ X) + epsilon)) * (X.T @ y), upper_clip), lower_clip)


def build_rolling_linear_regression_betas(
    X: Union[pd.Series, pd.DataFrame],
    y: Union[pd.Series, pd.DataFrame],
    window_size: int = DEFAULT_ROLLING_LR_WINDOW_SIZE,
) -> pd.Series:

    assert len(X) == len(y)

    betas = np.nan * np.empty(len(y))
    for n in range(window_size, len(y)):
        _x, _y = (
            X.iloc[(n - window_size) : n].values,
            y.iloc[(n - window_size) : n].values,
        )
        _beta = clipped_univariate_linear_regression(_x, _y)
        betas[n] = _beta

    return pd.Series(data=betas, index=y.index, name="LR_Beta")
