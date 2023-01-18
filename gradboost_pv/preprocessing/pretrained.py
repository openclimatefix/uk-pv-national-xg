import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import torch
import pandas as pd
from typing import Callable, Iterable, Optional, Tuple, Union, Iterator
import datetime as dt
from math import ceil, floor

from gradboost_pv.models.common import NWP_VARIABLE_NUM

AUTO_REGRESSION_TARGET_LAG = np.timedelta64(1, "h")  # to avoid look ahead bias
AUTO_REGRESSION_COVARIATE_LAG = AUTO_REGRESSION_TARGET_LAG + np.timedelta64(1, "h")
PRETRAINED_OUTPUT_DIMS = 1_000


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
