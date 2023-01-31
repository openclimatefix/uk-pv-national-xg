import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple

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


def sign_agnostic_modulo(x: int, base: int) -> int:
    """Function that maps all ints to [-base, base]

    Avoids the wrap around that occurs if you do:
        -1 % base = base-1
    Instead returns the following, e.g.
        sign_agnostic_modulo(-28, 25) = -3

    Args:
        x (int): int
        base (int): base for modulo
    """
    if x < 0:
        return -1 * (-1 * x % base)
    else:
        return x % base


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
    ).sort_index(ascending=False)

    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    ).sort_index(ascending=False)

    assert pd.infer_freq(gsp.index) == "-30T"

    # shift y by the step forecast
    y = gsp.shift(freq=forecast_horizon)

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(gsp.index.values)
    _X = pd.DataFrame(_X, index=gsp.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([X, _X], axis=1)

    # add lagged values of GSP PV
    # ar_2 = gsp.shift(freq=np.timedelta64(2, "h"))
    # ar_1 = gsp.shift(freq=np.timedelta64(1, "h"))
    # ar_day = gsp.shift(freq=np.timedelta64(1, "D"))

    NUM_GSP_OBS_ONE_DAY = 48
    NUM_GSP_OBS_BETWEEN_FORECAST = int(forecast_horizon / np.timedelta64(30, "m"))

    ar_day_lag = gsp.shift(
        sign_agnostic_modulo(
            -NUM_GSP_OBS_ONE_DAY
            + sign_agnostic_modulo(NUM_GSP_OBS_BETWEEN_FORECAST, 48),
            48 + 1,
        )
    )

    ar_2_hour_lag = gsp.shift(
        sign_agnostic_modulo(
            -NUM_GSP_OBS_ONE_DAY
            + sign_agnostic_modulo(
                NUM_GSP_OBS_BETWEEN_FORECAST - 4, 48
            ),  # shift by 2 hours
            48 + 1,
        )
    )

    ar_1_hour_lag = gsp.shift(
        sign_agnostic_modulo(
            -NUM_GSP_OBS_ONE_DAY
            + sign_agnostic_modulo(
                NUM_GSP_OBS_BETWEEN_FORECAST - 2, 48
            ),  # shift by 1 hour
            48 + 1,
        )
    )
    ar_2_hour_lag.columns = ["PV_LAG_2HR"]
    ar_1_hour_lag.columns = ["PV_LAG_1HR"]
    ar_day_lag.columns = ["PV_LAG_DAY"]

    # estimate linear trend of the PV
    pv_covariates = gsp.shift(
        freq=AUTO_REGRESSION_COVARIATE_LAG
    )  # add lag to PV data to avoid lookahead
    pv_target = y.shift(freq=(AUTO_REGRESSION_TARGET_LAG))
    betas = build_rolling_linear_regression_betas(pv_covariates, pv_target)

    X = pd.concat([X, ar_1_hour_lag, ar_2_hour_lag, ar_day_lag, betas], axis=1).dropna()
    y = y.reindex(X.index).dropna()
    X = X.loc[y.index]

    return X, y
