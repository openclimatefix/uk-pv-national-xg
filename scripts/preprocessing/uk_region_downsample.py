"""Script for processing raw NWP data"""
import datetime as dt
import itertools
from argparse import ArgumentParser

import pandas as pd
import xarray as xr

from gradboost_pv.models.region_filtered import build_local_save_path
from gradboost_pv.models.utils import GSP_FPATH, NWP_FPATH, NWP_STEP_HORIZON
from gradboost_pv.preprocessing.region_filtered import (
    DEFAULT_VARIABLES_FOR_PROCESSING,
    NWPUKRegionMaskedDatasetBuilder,
)
from gradboost_pv.utils.logger import getLogger

logger = getLogger("uk-region-filter-nwp-data")


FORECAST_HORIZONS = range(NWP_STEP_HORIZON)


def parse_args():
    """Parse command line arguments.

    Returns:
        args: Returns arguments
    """
    parser = ArgumentParser(
        description="Script to bulk process NWP xarray data for later use in simple ML model."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save collated data."
    )
    args = parser.parse_args()
    return args


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

        # time points to interpolate our nwp data onto.
        evaluation_timeseries = (
            gsp.coords["datetime_gmt"]
            .where(
                (gsp["datetime_gmt"] >= _nwp.coords["init_time"].values[0])
                & (gsp["datetime_gmt"] <= _nwp.coords["init_time"].values[-1]),
                drop=True,
            )
            .values
        )

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

            inner_fpath, outer_fpath = build_local_save_path(
                f"{args.save_dir}/{year}", step, var, year
            )

            uk_region.to_pickle(inner_fpath)
            outer_region.to_pickle(outer_fpath)

            logger.info(
                f"({i}/{len(years)}) Completed UK + Outer Region Feature Extraction for "
                f"var: {var}, step: {step}"
            )


if __name__ == "__main__":
    main()
