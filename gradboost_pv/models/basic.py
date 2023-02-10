"""Basic Model with single point downsampling"""
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation

from gradboost_pv.models.utils import TRIG_DATETIME_FEATURE_NAMES, build_lagged_features


def load_local_preprocessed_slice(forecast_horizon_step: int) -> pd.DataFrame:
    """TODO - remove"""
    return pd.read_pickle(
        f"/home/tom/local_data/basic_processed_nwp_data_step_{forecast_horizon_step}.pickle"
    )


def build_datasets_from_local(
    processed_nwp_slice: pd.DataFrame,
    national_gsp: xr.Dataset,
    forecast_horizon: np.timedelta64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates features for region-masked model.

    Args:
        processed_nwp_slice (pd.DataFrame): Processed NWP data performed at an earlier stage
        national_gsp (xr.Dataset): National GSP PV data
        forecast_horizon (np.timedelta64): forecast horizon for features

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X and y
    """

    X = processed_nwp_slice.sort_index(ascending=False)

    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    ).sort_index(ascending=False)

    assert pd.infer_freq(gsp.index) == "-30T"

    # shift y by the step forecast
    y = gsp.shift(freq=-forecast_horizon)

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(
        gsp.index.shift(freq=forecast_horizon).sort_values(ascending=False).values
    )
    _X = pd.DataFrame(_X, index=gsp.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([X, _X], axis=1).sort_index(ascending=False).dropna()

    pv_autoregressive_lags = build_lagged_features(gsp, forecast_horizon)

    X = pd.concat([X, pv_autoregressive_lags], axis=1).dropna()
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    return X, y
