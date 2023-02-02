import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation

from gradboost_pv.models.utils import (
    TRIG_DATETIME_FEATURE_NAMES,
    build_lagged_features,
    build_solar_pv_features,
)
from gradboost_pv.preprocessing.region_filtered import DEFAULT_VARIABLES_FOR_PROCESSING


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
    add_noise: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    columns = []
    for var in variables:
        columns += [f"{var}_{region}" for region in ["within", "outer"]]

    X = pd.DataFrame(
        data=processed_nwp_slice,
        columns=columns,
        index=national_gsp.coords["datetime_gmt"].values,
    ).sort_index(ascending=False)

    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    ).sort_index(ascending=False)

    _within = [col for col in X.columns if "within" in col]
    _outer = [col for col in X.columns if "outer" in col]
    X_diff = pd.DataFrame(
        data=(X[_within].values - X[_outer].values),
        columns=[col.replace("_within", "_diff") for col in _within],
        index=X.index,
    )

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(
        gsp.index.shift(freq=forecast_horizon).sort_values(ascending=False).values
    )
    _X = pd.DataFrame(_X, index=gsp.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([X, X_diff, _X], axis=1).sort_index(ascending=False).dropna()

    solar_variables = build_solar_pv_features(
        gsp.index.shift(freq=forecast_horizon).sort_values(ascending=False)
    )
    solar_variables.index = gsp.index

    # shift y by the step forecast
    y = gsp.shift(freq=-forecast_horizon)

    # add lagged values of GSP PV
    pv_autoregressive_lags = build_lagged_features(gsp, forecast_horizon)

    X = pd.concat(
        [
            X,
            pv_autoregressive_lags,
            solar_variables,
        ],
        axis=1,
    ).dropna()

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    if add_noise:
        # use random noise as a benchmark for uninformative features
        # used only in model analysis/benchmarking

        noise = pd.DataFrame(
            columns=["RANDOM_NOISE"], data=np.random.randn(len(X)), index=X.index
        )

        X = pd.concat([X, noise], axis=1)

    return X, y
