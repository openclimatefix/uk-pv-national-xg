from typing import Iterable
import xarray as xr
import numpy as np


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
