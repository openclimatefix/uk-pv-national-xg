from typing import Iterable
import numpy as np
import xarray as xr
import itertools
import multiprocessing as mp
import sys

sys.path.append("home/tom/dev/gradboost_pv/")
from gradboost_pv.preprocessing.geospatial import (
    get_eso_uk_multipolygon,
    generate_polygon_mask,
)

FORECAST_HORIZONS = range(2)
VARIABLES = range(2)
NWP_PATH = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr/"
)
GSP_PATH = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"


def build_region_masked_covariates(
    eval_timepoints: Iterable[np.datetime64],
    nwp_data: xr.Dataset,
    fh: int,
    geospatial_mask: np.ndarray,
) -> np.ndarray:
    X = nwp_data.isel(step=fh).to_array()
    X = (geospatial_mask * X).mean(dim=["x", "y"])
    X = X.interp(init_time=eval_timepoints, method="linear")
    return X.as_numpy()


def main():
    """
    Temporary Script to preprocess NWP data (for overnight)
    NOT PRODUCTION!!
    """
    gsp = xr.open_zarr(GSP_PATH)
    nwp = xr.open_zarr(NWP_PATH)
    nwp = nwp.chunk({"init_time": 2, "variable": 2, "step": 1})

    evaluation_timeseries = (
        gsp.coords["datetime_gmt"]
        .where(
            (gsp["datetime_gmt"] >= nwp.coords["init_time"].values[0])
            & (gsp["datetime_gmt"] <= nwp.coords["init_time"].values[-1]),
            drop=True,
        )
        .values
    )

    uk_poly = get_eso_uk_multipolygon()
    uk_mask = generate_polygon_mask(
        nwp.coords["x"].values, nwp.coords["y"].values, uk_poly
    )

    params = [(evaluation_timeseries, nwp, _x, uk_mask) for _x in FORECAST_HORIZONS]
    print(params[0])

    results = list()
    with mp.Pool(10) as pool:
        for result in pool.starmap(build_region_masked_covariates, params):
            results.append(result)

    for idx, _params in enumerate(params):
        _, _, fh, _ = _params
        X = results[idx]
        np.save(f"/home/tom/local_data/testing_uk_filter_v2_fh{fh}.npy", X)


if __name__ == "__main__":
    main()
