import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import torch
import pickle
from torchvision.models import resnet101
from ocf_datapipes.load.nwp.nwp import OpenNWPIterDataPipe

from typing import Callable, Iterable, Optional, Union
from scipy.ndimage import zoom
import datetime as dt


PRETRAINED_OUTPUT_DIMS = 1_000
NWP_VARIABLE_NUM = 17
NWP_FPATH = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr/"
)
GSP_FPATH = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"
NUM_OBS = 35_000


def _downsample_and_process_for_pretrained_input(
    nwp_data_time_slice: xr.DataArray,
) -> torch.Tensor:
    nwp_data_time_slice = nwp_data_time_slice.as_numpy().values
    nwp_data_time_slice = np.nan_to_num(nwp_data_time_slice)
    nwp_data_time_slice = np.tile(
        zoom(nwp_data_time_slice, (1, 64 / 704, 64 / 548), order=1), (3, 1, 1, 1)
    ).reshape(17, 3, 64, 64)
    return torch.from_numpy(nwp_data_time_slice)


def _downsample_pretrained_output(model_output: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = torch.softmax(model_output, 1)
        output = np.split(output, range(200, PRETRAINED_OUTPUT_DIMS, 200), axis=1)
        dsampled_output = torch.concat(
            [torch.linalg.norm(x, axis=1).reshape(1, NWP_VARIABLE_NUM) for x in output]
        ).T
        assert dsampled_output.shape == (NWP_VARIABLE_NUM, 5)
        return dsampled_output.numpy().flatten()


@functional_datapipe("process_nwp_pretrained")
class ProcessNWPPretrainedIterDataPipe(IterDataPipe):
    def __init__(
        self,
        base_nwp_datapipe: IterDataPipe,
        step: int,
        pretrained_model: Callable[[torch.Tensor], torch.Tensor],
        interpolate: bool = False,
        interpolation_timepoints=Optional[Iterable[Union[dt.datetime, np.datetime64]]],
    ) -> None:
        if interpolate:
            assert (
                interpolation_timepoints is not None
            ), "Must provide points for interpolation."
        self.source_datapipe = base_nwp_datapipe
        self.step = step
        self.pretrained_model = pretrained_model
        self.interpolation_timepoints = interpolation_timepoints

    def __iter__(self):
        for nwp in self.source_datapipe:
            nwp = nwp.isel(step=self.step)  # select the horizon we want
            nwp.interp(
                init_time_utc=self.interpolation_timepoints, method="cubic"
            )  # interpolate to perscribed points
            for time, nwp_by_init_time in nwp.groupby("init_time_utc"):
                # at each time point, pass the nwp data to pretrained model
                data = _downsample_and_process_for_pretrained_input(nwp_by_init_time)
                data = self.pretrained_model(data)
                data = _downsample_pretrained_output(data)
                yield (time, data)


if __name__ == "__main__":
    gsp_path = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"
    gsp = xr.open_zarr(gsp_path)
    interp_timepoints = gsp["datetime_gmt"].values[
        (gsp["datetime_gmt"] >= np.datetime64("2020-01-01"))
        & (gsp["datetime_gmt"] < np.datetime64("2022-01-01"))
    ]
    base_nwp_dpipe = OpenNWPIterDataPipe(NWP_FPATH)
    model = resnet101(pretrained=True)
    for step in range(1):
        process_nwp_dpipe = ProcessNWPPretrainedIterDataPipe(
            base_nwp_dpipe,
            step,
            model,
            interpolate=True,
            interpolation_timepoints=interp_timepoints,
        )
        results = []
        count = 0
        for tstamp, data in process_nwp_dpipe:
            results.append((tstamp, data))

        with open(
            f"/home/tom/local_data/pretrained_nwp_processing_step_{step}.pkl", "wb"
        ) as f:
            pickle.dump(results, f)
