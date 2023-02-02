import numpy as np
import xarray as xr
from argparse import ArgumentParser

from gradboost_pv.models.utils import NWP_FPATH, GSP_FPATH, NWP_STEP_HORIZON
from gradboost_pv.preprocessing.basic import preprocess_nwp_per_step
from gradboost_pv.utils.logger import getLogger


logger = getLogger("basic-process-nwp-data")


def parse_args():
    parser = ArgumentParser(
        description="Script to bulk process NWP xarray data for later use in simple ML model."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save collated data."
    )
    args = parser.parse_args()
    return args


def _build_local_save_path(path_to_dir: str, forecast_horizon: int) -> str:
    return f"{path_to_dir}/basic_nwp_preprocessed_step_{forecast_horizon}.npy"


def main():
    """
    Bulk process NWP data to single pointwise entry.
    """

    args = parse_args()

    gsp = xr.open_zarr(GSP_FPATH)
    nwp = xr.open_zarr(NWP_FPATH)

    evalutation_timeseries = (
        gsp.coords["datetime_gmt"]
        .where(
            (gsp["datetime_gmt"] >= nwp.coords["init_time"].values[0])
            & (gsp["datetime_gmt"] <= nwp.coords["init_time"].values[-1]),
            drop=True,
        )
        .values
    )  # create a subset at same interval as GSP (30 mins) but over the timespan of NWP

    for forecast_horizon in range(NWP_STEP_HORIZON):
        # could be multiprocessed, but I am running overnight anyway
        X = preprocess_nwp_per_step(evalutation_timeseries, nwp, forecast_horizon)
        np.save(
            _build_local_save_path(args.save_dir, forecast_horizon),
            X,
        )
        logger.info(f"Completed processing of data for step: {forecast_horizon}")


if __name__ == "__main__":
    main()
