import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple

from gradboost_pv.models.common import (
    trigonometric_datetime_transformation,
    TRIG_DATETIME_FEATURE_NAMES,
    build_rolling_linear_regression_betas,
)
from gradboost_pv.preprocessing.region_filtered import DEFAULT_VARIABLES_FOR_PROCESSING


AUTO_REGRESSION_TARGET_LAG = np.timedelta64(1, "h")  # to avoid look ahead bias
AUTO_REGRESSION_COVARIATE_LAG = AUTO_REGRESSION_TARGET_LAG + np.timedelta64(1, "h")


def load_local_preprocessed_slice(
    variable: str, forecast_horizon_step: int, inner: bool = True
) -> np.ndarray:
    if inner:
        return np.load(
            f"/home/tom/local_data/uk_region_mean_var{variable}_step{forecast_horizon_step}.npy"
        )
    else:
        return np.load(
            f"/home/tom/local_data/outer_region_mean_var{variable}_step{forecast_horizon_step}.npy"
        )


def load_all_variable_slices(
    forecast_horizon_step: int, variables: list[str] = DEFAULT_VARIABLES_FOR_PROCESSING
) -> np.ndarray:
    X = list()
    for variable in variables:
        X.append(
            load_local_preprocessed_slice(variable, forecast_horizon_step, inner=True)
        )
        X.append(
            load_local_preprocessed_slice(variable, forecast_horizon_step, inner=False)
        )
    X = np.concatenate(X, axis=0).T
    return X


def build_datasets_from_local(
    processed_nwp_slice: np.ndarray,
    national_gsp: xr.Dataset,
    forecast_horizon: np.timedelta64,
    variables: list[str] = DEFAULT_VARIABLES_FOR_PROCESSING,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    columns = []
    for var in variables:
        columns += [f"{var}_{region}" for region in ["within", "outer"]]

    X = pd.DataFrame(
        data=processed_nwp_slice,
        columns=columns,
        index=national_gsp.coords["datetime_gmt"].values,
    )
    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    )

    _within = [col for col in X.columns if "within" in col]
    _outer = [col for col in X.columns if "outer" in col]
    X_diff = pd.DataFrame(
        data=(X[_within].values - X[_outer].values),
        columns=[col.replace("_within", "_diff") for col in _within],
        index=X.index,
    )

    # shift y by the step forecast
    y = gsp.shift(freq=-forecast_horizon).dropna()

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(
        y.shift(freq=forecast_horizon).index.values
    )
    _X = pd.DataFrame(_X, index=y.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([X, X_diff, _X], axis=1)

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
