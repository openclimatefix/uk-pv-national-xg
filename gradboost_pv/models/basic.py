import numpy as np
import pandas as pd
import xarray as xr
from typing import Iterable, Tuple

from gradboost_pv.models.common import (
    trigonometric_datetime_transformation,
    TRIG_DATETIME_FEATURE_NAMES,
    ORDERED_NWP_FEATURE_VARIABLES,
    build_rolling_linear_regression_betas,
)

AUTO_REGRESSION_TARGET_LAG = np.timedelta64(1, "h")  # to avoid look ahead bias
AUTO_REGRESSION_COVARIATE_LAG = AUTO_REGRESSION_TARGET_LAG + np.timedelta64(1, "h")


def load_local_preprocessed_slice(forecast_horizon_step: int) -> np.ndarray:
    return np.load(
        f"/home/tom/local_data/basic_processed_nwp_data_step_{forecast_horizon_step}.npy"
    )


def build_datasets_from_local(
    processed_nwp_slice: np.ndarray,
    national_gsp: xr.Dataset,
    forecast_horizon: np.timedelta64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    X = pd.DataFrame(
        data=processed_nwp_slice.reshape(
            processed_nwp_slice.shape[0], processed_nwp_slice.shape[1]
        ).T,
        columns=ORDERED_NWP_FEATURE_VARIABLES,
        index=national_gsp.coords["datetime_gmt"].values,
    )
    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    )

    # shift y by the step forecast
    y = gsp.shift(freq=-forecast_horizon).dropna()

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(
        y.shift(freq=forecast_horizon).index.values
    )
    _X = pd.DataFrame(_X, index=y.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([X, _X], axis=1)

    # add lagged values of GSP PV
    ar_2 = gsp.shift(freq=np.timedelta64(2, "h"))
    ar_1 = gsp.shift(freq=np.timedelta64(1, "h"))
    ar_day = gsp.shift(freq=np.timedelta64(1, "D"))
    ar_2.columns = ["PV_LAG_2HR"]
    ar_1.columns = ["PV_LAG_1HR"]
    ar_day.columns = ["PV_LAG_DAY"]

    # estimate linear trend of the PV
    pv_covariates = gsp.shift(
        freq=AUTO_REGRESSION_COVARIATE_LAG
    )  # add lag to PV data to avoid lookahead
    pv_target = y.shift(freq=(AUTO_REGRESSION_TARGET_LAG))
    betas = build_rolling_linear_regression_betas(pv_covariates, pv_target)

    X = pd.concat([X, ar_1, ar_2, ar_day, betas], axis=1).dropna()
    y = y.reindex(X.index).dropna()
    X = X.loc[y.index]

    return X, y


def preprocess_nwp_per_step(
    evaluation_timepoints: Iterable[np.datetime64],
    nwp: xr.Dataset,
    forecast_horizon_step: int,
) -> np.ndarray:
    X = nwp.isel(step=forecast_horizon_step)
    X = X.chunk({"init_time": 1, "variable": 1})
    X = X.mean(dim=["x", "y"])
    X = X.interp(init_time=evaluation_timepoints, method="cubic")

    # compute the transformation - this can take some time
    # approx 10 mins per call.
    X = X.to_array().as_numpy().values
    return X
