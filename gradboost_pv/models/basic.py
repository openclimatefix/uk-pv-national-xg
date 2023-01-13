import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple

from gradboost_pv.models.utils import (
    trigonometric_datetime_transformation,
    TRIG_DATETIME_FEATURE_NAMES,
    ORDERED_NWP_FEATURE_VARIABLES,
)


def load_local_preprocessed_slice(forecast_horizon_step: int) -> np.ndarray:
    return np.load(
        f"/home/tom/local_data/basic_processed_nwp_data_step_{forecast_horizon_step}.npy"
    )


def build_datasets_from_local(
    processed_nwp_slice: np.ndarray,
    national_gsp: xr.Dataset,
    forecast_horizon: np.timedelta64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(processed_nwp_slice.shape) > 2:
        X = pd.DataFrame(
            data=processed_nwp_slice.reshape(
                processed_nwp_slice.shape[1], processed_nwp_slice.shape[2]
            ).T,
            columns=ORDERED_NWP_FEATURE_VARIABLES,
            index=national_gsp.coords["datetime_gmt"].values,
        )
    else:
        X = pd.DataFrame(
            data=processed_nwp_slice.reshape(
                processed_nwp_slice.shape[0], processed_nwp_slice.shape[1]
            ).T,
            columns=ORDERED_NWP_FEATURE_VARIABLES,
            index=national_gsp.coords["datetime_gmt"].values,
        )
    y = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    )

    # shift y by the step forecast
    y = y.shift(freq=-forecast_horizon).dropna()
    common_index = sorted(pd.DatetimeIndex((set(y.index).intersection(X.index))))
    X, y = X.loc[common_index], y.loc[common_index]

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(
        y.shift(freq=forecast_horizon).index.values
    )
    _X = pd.DataFrame(_X, index=y.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([X, _X], axis=1)

    # add lagged values of GSP PV
    ar_1 = y.shift(freq=(forecast_horizon + np.timedelta64(1, "h")))
    ar_day = y.shift(freq=(forecast_horizon + np.timedelta64(1, "D")))
    ar_1.columns = ["PV_LAG_1HR"]
    ar_day.columns = ["PV_LAG_DAY"]

    # estimate linear trend of the PV
    window_size = 10
    epsilon = 0.01
    y_covariates = y.shift(freq=(forecast_horizon + np.timedelta64(2, "h")))
    y_covariates.columns = ["x"]
    y_target = y.shift(freq=(forecast_horizon + np.timedelta64(1, "h")))
    y_target.columns = ["y"]
    data = pd.concat([y_target, y_covariates], axis=1).dropna()
    _x = data["x"].values
    _y = data["y"].values
    _betas = np.nan * np.empty(len(data))

    for n in range(window_size, len(data)):
        __y = _y[(n - window_size) : n]
        __x = _x[(n - window_size) : n]
        __b = max(min((1 / ((__x.T @ __x) + epsilon)) * (__x.T @ __y), 10), -10)
        _betas[n] = __b

    betas = pd.DataFrame(data=_betas, columns=["AR_Beta"], index=data.index)

    X = pd.concat([X, ar_1, ar_day, betas], axis=1).dropna()
    y = y.loc[X.index]

    return X, y
