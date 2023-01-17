import xarray as xr
from torchvision.models import resnet101
from ocf_datapipes.load.nwp.nwp import OpenNWPIterDataPipe
import pandas as pd
from math import ceil

from gradboost_pv.models.pretrained import ProcessNWPPretrainedIterDataPipe
from gradboost_pv.models.common import (
    NWP_FPATH,
    GSP_FPATH,
    NWP_STEP_HORIZON,
    NWP_VARIABLE_NUM,
)


if __name__ == "__main__":
    gsp = xr.open_zarr(GSP_FPATH)
    nwp = xr.open_zarr(NWP_FPATH)

    BATCH_SIZE = 1_000

    # get a common 30 minute interval timeseries for GSP
    evaluation_timeseries = (
        gsp.coords["datetime_gmt"]
        .where(
            (gsp["datetime_gmt"] >= nwp.coords["init_time"].values[0])
            & (gsp["datetime_gmt"] <= nwp.coords["init_time"].values[-1]),
            drop=True,
        )
        .values
    )

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

        results.to_pickle(
            f"/home/tom/local_data/pretrained_nwp_processing_step_{step}.pkl"
        )

        print(f"Complete Pretrained Proprocessing for step: {step}")
