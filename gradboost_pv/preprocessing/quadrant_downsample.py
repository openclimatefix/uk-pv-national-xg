from typing import Iterable
import xarray as xr
import numpy as np

DOWNSAMPLE_TO_NUM_POINTS = 2


def preprocess_nwp_per_step(
    eval_timepoints: Iterable[np.datetime64], nwp_data: xr.Dataset, fh: int
):
    X = nwp_data.isel(step=fh)
    X = X.chunk({"init_time": 1, "variable": 1})
    X = X.coarsen(
        dim={
            "x": int(548 / DOWNSAMPLE_TO_NUM_POINTS),
            "y": int(704 / DOWNSAMPLE_TO_NUM_POINTS),
        },
        boundary="exact",
    ).mean()
    X = X.interp(init_time=eval_timepoints, method="cubic")

    # compute the transformation - this can take some time
    # around 10 mins per timestep
    _X = X.to_array().as_numpy()

    return _X
