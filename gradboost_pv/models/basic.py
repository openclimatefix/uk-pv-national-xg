import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation

from gradboost_pv.models.utils import (
    TRIG_DATETIME_FEATURE_NAMES,
    ORDERED_NWP_FEATURE_VARIABLES,
    build_lagged_features,
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
