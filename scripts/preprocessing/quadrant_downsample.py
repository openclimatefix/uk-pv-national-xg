from typing import Iterable
import numpy as np
import xarray as xr


nwp_path = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr/"
)
gsp_path = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"
forecast_horizons = range(15, 36)

DOWNSAMPLE_TO_NUM_POINTS = 2


def build_basic_covariates(
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


def main():
    """
    Temporary Script to preprocess NWP data (for overnight)
    NOT PRODUCTION!!
    """
    gsp = xr.open_zarr(gsp_path)
    nwp = xr.open_zarr(nwp_path)

    evalutation_timeseries = (
        gsp.coords["datetime_gmt"]
        .where(
            (gsp["datetime_gmt"] >= nwp.coords["init_time"].values[0])
            & (gsp["datetime_gmt"] <= nwp.coords["init_time"].values[-1]),
            drop=True,
        )
        .values
    )

    for fh in forecast_horizons:
        # could be multiprocessed, but I am running overnight anyway
        X = build_basic_covariates(evalutation_timeseries, nwp, fh)
        np.save(
            f"/home/tom/local_data/geospatial_dsample_processed_nwp_data_step_{fh}.npy",
            X,
        )
        print(f"Completed processing of data for step: {fh}")


if __name__ == "__main__":
    main()
