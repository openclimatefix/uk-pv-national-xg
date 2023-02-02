from argparse import ArgumentParser
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import itertools
from typing import Tuple

from gradboost_pv.preprocessing.region_filtered import (
    NWPUKRegionMaskedDatasetBuilder,
    DEFAULT_VARIABLES_FOR_PROCESSING,
)
from gradboost_pv.models.utils import NWP_STEP_HORIZON, NWP_FPATH, GSP_FPATH
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

    # if we consider all nwp together the polygon mask is too big to fit in memory
    # instead, iterate over each year in the dataset and save locally

    years = pd.DatetimeIndex(nwp.init_time.values).year.unique().values
    date_years = [dt.datetime(year=year, month=1, day=1) for year in years]

    for i in range(len(years) - 1):
        year = years[i]
        start_datetime, end_datetime = date_years[i], date_years[i + 1]
        _nwp = nwp.sel(init_time=slice(start_datetime, end_datetime))

        evaluation_timeseries = (
            gsp.coords["datetime_gmt"]
            .where(
                (gsp["datetime_gmt"] >= _nwp.coords["init_time"].values[0])
                & (gsp["datetime_gmt"] <= _nwp.coords["init_time"].values[-1]),
                drop=True,
            )
            .values
        )

        # save the time points of our dataset
        np.save(f"{args.save_dir}/{year}/eval_timepoints.npy", evaluation_timeseries)

        dataset_builder = NWPUKRegionMaskedDatasetBuilder(
            _nwp,
            evaluation_timeseries,
        )

        iter_params = list(
            itertools.product(DEFAULT_VARIABLES_FOR_PROCESSING, FORECAST_HORIZONS)
        )
        for var, step in iter_params:
            uk_region, outer_region = dataset_builder.build_region_masked_covariates(
                var, step
            )

            inner_fpath, outer_fpath = _build_local_save_path(
                f"{args.save_dir}/{year}", step, var
            )

            np.save(inner_fpath, uk_region)
            np.save(outer_fpath, outer_region)

            logger.info(
                f"({i}/{len(years)}) Completed UK + Outer Region Feature Extraction for "
                f"var: {var}, step: {step}"
            )


if __name__ == "__main__":
    main()
