import numpy as np
import xarray as xr
import itertools

from gradboost_pv.preprocessing.geospatial import NWPUKRegionMaskedDatasetBuilder

FORECAST_HORIZONS = range(37)
VARIABLES = [
    "dswrf",
    "hcct",
    "lcc",
    "t",
    "sde",
    "wdir10",
]

NWP_PATH = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr/"
)
GSP_PATH = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"


def main():
    """
    Script to preprocess NWP data, overnight
    """
    gsp = xr.open_zarr(GSP_PATH)
    nwp = xr.open_zarr(NWP_PATH, chunks={"step": 1, "variable": 1, "init_time": 50})

    evaluation_timeseries = (
        gsp.coords["datetime_gmt"]
        .where(
            (gsp["datetime_gmt"] >= nwp.coords["init_time"].values[0])
            & (gsp["datetime_gmt"] <= nwp.coords["init_time"].values[-1]),
            drop=True,
        )
        .values
    )

    dataset_builder = NWPUKRegionMaskedDatasetBuilder(
        nwp,
        evaluation_timeseries,
    )

    iter_params = list(itertools.product(VARIABLES, FORECAST_HORIZONS))
    for var, step in iter_params:
        uk_region, outer_region = dataset_builder.build_region_masked_covariates(
            var, step
        )
        np.save(
            f"/home/tom/local_data/uk_region_mean_var{var}_step{step}.npy", uk_region
        )
        np.save(
            f"/home/tom/local_data/outer_region_mean_var{var}_step{step}.npy",
            outer_region,
        )
        print(
            f"Completed UK + Outer Region Feature Extraction for var: {var}, step: {step}"
        )


if __name__ == "__main__":
    main()
