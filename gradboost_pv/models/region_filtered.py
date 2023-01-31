import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple
from pvlib.location import Location

from gradboost_pv.models.common import (
    trigonometric_datetime_transformation,
    TRIG_DATETIME_FEATURE_NAMES,
)
from gradboost_pv.preprocessing.region_filtered import DEFAULT_VARIABLES_FOR_PROCESSING


AUTO_REGRESSION_TARGET_LAG = np.timedelta64(1, "h")  # to avoid look ahead bias
AUTO_REGRESSION_COVARIATE_LAG = AUTO_REGRESSION_TARGET_LAG + np.timedelta64(1, "h")

LATITUDE_UK_SOUTH_CENTER = 52.80111
LONGITUDE_UK_SOUTH_CENTER = -1.0967
DEFAULT_UK_SOUTH_LOCATION = Location(
    LATITUDE_UK_SOUTH_CENTER, LONGITUDE_UK_SOUTH_CENTER
)


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


def build_solar_pv_features(
    times_of_forecast: pd.DatetimeIndex, location: Location = DEFAULT_UK_SOUTH_LOCATION
) -> pd.DataFrame:
    """Build PV/Solar features given times and a location.

    Args:
        times_of_forecast (pd.DatetimeIndex): times at which to compute solar data.
        location (Location, optional): Location for computation.
        Defaults to DEFAULT_UK_SOUTH_LOCATION.

    Returns:
        pd.DataFrame: A DataFrame with various solar position / clear sky features.
        Indexed by forecast times
    """
    clear_sky = location.get_clearsky(times_of_forecast)[["ghi", "dni"]]
    solar_position = location.get_solarposition(times_of_forecast)[
        ["zenith", "elevation", "azimuth", "equation_of_time"]
    ]
    solar_variables = pd.concat([clear_sky, solar_position], axis=1)
    return solar_variables


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
    NUM_GSP_OBS_BETWEEN_FORECAST = int(forecast_horizon / np.timedelta64(30, "m"))

    ar_day_lag = gsp.shift(
        freq=np.timedelta64(int(24 - ((NUM_GSP_OBS_BETWEEN_FORECAST / 2) % 24)), "h")
    )

    ar_2_hour_lag = gsp.shift(
        freq=np.timedelta64(
            int(24 - ((NUM_GSP_OBS_BETWEEN_FORECAST / 2 - 2) % 24)), "h"
        )
    )

    ar_1_hour_lag = gsp.shift(
        freq=np.timedelta64(
            int(24 - ((NUM_GSP_OBS_BETWEEN_FORECAST / 2 - 1) % 24)), "h"
        )
    )

    pv_autoregressive_lags = (
        pd.concat([ar_day_lag, ar_1_hour_lag, ar_2_hour_lag], axis=1)
        .sort_index(ascending=False)
        .dropna()
    )
    pv_autoregressive_lags.columns = ["PV_LAG_DAY", "PV_LAG_1HR", "PV_LAG_2HR"]

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
