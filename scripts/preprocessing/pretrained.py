import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import torch
from torchvision.models import resnet101
from ocf_datapipes.load.nwp.nwp import OpenNWPIterDataPipe
import pandas as pd

from typing import Callable, Iterable, Optional, Tuple, Union, Iterator
import datetime as dt
from math import ceil, floor


PRETRAINED_OUTPUT_DIMS = 1_000
NWP_VARIABLE_NUM = 17
NWP_STEP_HORIZON = 37
NWP_FPATH = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr/"
)
GSP_FPATH = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"


@functional_datapipe("process_nwp_pretrained")
class ProcessNWPPretrainedIterDataPipe(IterDataPipe):
    def __init__(
        self,
        base_nwp_datapipe: IterDataPipe,
        pretrained_model: Callable[[torch.Tensor], torch.Tensor],
        step: int,
        batch_size: int = 50,
        interpolate: bool = False,
        interpolation_timepoints: Optional[
            Iterable[Union[dt.datetime, np.datetime64]]
        ] = None,
    ) -> None:
        if interpolate:
            assert (
                interpolation_timepoints is not None
            ), "Must provide points for interpolation."
        self.source_datapipe = base_nwp_datapipe
        self.step = step
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.interpolate = interpolate
        self.interpolation_timepoints = interpolation_timepoints

    def process_for_pretrained_input(self, nwp_batch: xr.DataArray) -> torch.Tensor:
        nwp_batch = nwp_batch.to_numpy()
        n = nwp_batch.shape[0]  # batch size for all but the last batch
        nwp_batch = nwp_batch.reshape(NWP_VARIABLE_NUM * n, 64, 68)
        nwp_batch = np.tile(nwp_batch, (3, 1, 1, 1)).reshape(
            NWP_VARIABLE_NUM * n, 3, 64, 68
        )
        nwp_batch = torch.from_numpy(np.nan_to_num(nwp_batch))
        return nwp_batch

    def downsample_pretrained_output(self, model_output: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            output = torch.softmax(model_output, 1)
            output = np.split(output, range(200, PRETRAINED_OUTPUT_DIMS, 200), axis=1)
            n = torch.linalg.norm(output[0], axis=1).shape[0]
            dsampled_output = torch.concat(
                [torch.linalg.norm(x, axis=1).reshape(1, n) for x in output]
            ).T
        return dsampled_output.numpy()

    def __iter__(self) -> Iterator[Tuple[pd.DatetimeIndex, np.ndarray]]:
        for nwp in self.source_datapipe:
            nwp = nwp.isel(step=self.step)  # select the horizon we want
            if self.interpolate:
                nwp = nwp.interp(
                    init_time_utc=self.interpolation_timepoints, method="linear"
                )
            for batch_idx in range(
                1, ceil(len(nwp.init_time_utc) / self.batch_size) + 1
            ):
                batch = nwp.isel(
                    init_time_utc=slice(
                        self.batch_size * (batch_idx - 1), self.batch_size * batch_idx
                    )
                )
                batch = batch.coarsen(
                    dim={
                        "x_osgb": floor(548 / 64),
                        "y_osgb": floor(704 / 64),
                    },
                    boundary="trim",
                ).mean()
                time_slice = batch.coords["init_time_utc"].values
                with torch.no_grad():
                    batch = self.process_for_pretrained_input(batch)
                    output = self.pretrained_model(batch)
                    output = self.downsample_pretrained_output(output)
                yield (time_slice, output)


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
                    res[1].reshape(len(res[0]), 17 * 5), columns=columns, index=res[0]
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
