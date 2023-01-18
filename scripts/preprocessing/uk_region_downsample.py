from argparse import ArgumentParser
import numpy as np
import xarray as xr
import itertools
from typing import Tuple

from gradboost_pv.preprocessing.region_filtered import (
    NWPUKRegionMaskedDatasetBuilder,
    DEFAULT_VARIABLES_FOR_PROCESSING,
)
from gradboost_pv.models.common import NWP_STEP_HORIZON, NWP_FPATH, GSP_FPATH
from gradboost_pv.utils.logger import getLogger


logger = getLogger("uk-region-filter-nwp-data")


FORECAST_HORIZONS = range(NWP_STEP_HORIZON)


def parse_args():
    parser = ArgumentParser(
        description="Script to bulk process NWP xarray data for later use in simple ML model."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save collated data."
    )
    args = parser.parse_args()
    return args


def _build_local_save_path(path_to_dir, forecast_horizon, variable) -> Tuple[str, str]:
    return (
        f"{path_to_dir}/uk_region_inner_variable_{variable}_step_{forecast_horizon}.npy",
        f"{path_to_dir}/uk_region_outer_variable_{variable}_step_{forecast_horizon}.npy",
    )


def main():
    """
    Script to preprocess NWP data, overnight
    """

    args = parse_args()

    gsp = xr.open_zarr(GSP_FPATH)
    nwp = xr.open_zarr(NWP_FPATH)
    nwp = nwp.chunk({"step": 1, "variable": 1, "init_time": 50})

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

    iter_params = list(
        itertools.product(DEFAULT_VARIABLES_FOR_PROCESSING, FORECAST_HORIZONS)
    )
    for var, step in iter_params:
        uk_region, outer_region = dataset_builder.build_region_masked_covariates(
            var, step
        )

        inner_fpath, outer_fpath = _build_local_save_path(args.save_dir, step, var)

        np.save(inner_fpath, uk_region)
        np.save(outer_fpath, outer_region)

        logger.info(
            f"Completed UK + Outer Region Feature Extraction for var: {var}, step: {step}"
        )


if __name__ == "__main__":
    main()
