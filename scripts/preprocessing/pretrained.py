from argparse import ArgumentParser
import xarray as xr
from torchvision.models import resnet101
from ocf_datapipes.load.nwp.nwp import OpenNWPIterDataPipe
import pandas as pd
from math import ceil

from gradboost_pv.preprocessing.pretrained import ProcessNWPPretrainedIterDataPipe
from gradboost_pv.models.utils import (
    NWP_FPATH,
    NWP_STEP_HORIZON,
    NWP_VARIABLE_NUM,
)
from gradboost_pv.utils.logger import getLogger


logger = getLogger("pretrained-process-nwp-data")


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
    return f"{path_to_dir}/pretrained_nwp_processed_step_{forecast_horizon}.pkl"


def main():
    """Bulk Processing NWP data with a pretrained CNN, saving the output locally."""
    args = parse_args()

    # init base NWP datapipe
    nwp = xr.open_zarr(NWP_FPATH)

    BATCH_SIZE = 1_000

    # init base NWP datapipe
    base_nwp_dpipe = OpenNWPIterDataPipe(NWP_FPATH)
    model = resnet101(pretrained=True)

    # for each forecast horizon, proprocess and save the data locally.
    for step in range(NWP_STEP_HORIZON):
        process_nwp_dpipe = ProcessNWPPretrainedIterDataPipe(
            base_nwp_dpipe,
            model,
            step,
            batch_size=BATCH_SIZE,
            interpolate=False,  # cheaper to interpolate afterwards
        )

        results = []
        count = 0
        for tstamp, data in process_nwp_dpipe:
            results.append((tstamp, data))
            count += 1

            if count >= ceil(len(nwp.init_time) / BATCH_SIZE):
                # only run the preprocessing through one cycle of data
                break

        # format the results into a format to save and use later

        # generate column names of our data
        columns = list()
        for variable in nwp.coords["variable"].values:
            for idx in range(5):
                columns.append(f"{variable}_{idx}")

        # put it all into a DataFrame to save locally and load later.
        # interpolate our pretrained features to 1/2 hourly interval, same
        # as out GSP target.
        results = pd.concat(
            [
                pd.DataFrame(
                    res[1].reshape(len(res[0]), NWP_VARIABLE_NUM * 5),
                    columns=columns,
                    index=res[0],
                )
                .reindex(pd.date_range(start=res[0][0], end=res[0][-1], freq="0.5H"))
                .interpolate(method="linear", axis=0)
                for res in results
            ]
        ).sort_index()

        results.to_pickle(_build_local_save_path(args.save_dir, step))
        logger.info(f"Complete Pretrained Proprocessing for step: {step}")


if __name__ == "__main__":
    main()
