import asyncio
from dataclasses import dataclass
from typing import Iterator, Callable

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@dataclass(frozen=True)
class LiveInferenceData:
    datetime: np.datetime64
    nwp_slice: xr.Dataset
    gsp_slice: xr.Dataset


@dataclass(frozen=True)
class ModelOutput:
    datetime_of_forecast_utc: np.datetime64
    datetime_of_target_utc: np.datetime64
    forecast_kw: float


@functional_datapipe("mock_datafeed")
class MockDataFeed(IterDataPipe):
    def __init__(self, nwp_data: xr.Dataset, gsp_data: xr.Dataset) -> None:
        self.data_feed = self.create_data_feed(nwp_data, gsp_data)

    def create_data_feed(
        self, nwp: xr.Dataset, gsp: xr.Dataset
    ) -> list[LiveInferenceData]:
        """Create a tuple of individual time-slices for coupled nwp and gsp.

        Since the nwp and gsp are sampled at different frequencies, we realign
        them so that at the tick of one data, we also have the most recent data
        of the other source at hand. This is meant to simulate iteratively
        reading the most recent elements a database at given intervals.

        Args:
            nwp (xr.Dataset): NWP dataset with "init_time" np.datetime64 coords
            gsp (xr.Dataset): GSP PV data with "datetime_gmt" np.datetime64 coords
        """

        tseries_nwp = nwp.coords["init_time"].values
        tseries_gsp = gsp.coords["datetime_gmt"].values
        assert (
            tseries_gsp[0] == tseries_nwp[0]
        ), "Datasets must start at the same point in time"

        tseries = np.sort(
            np.unique(np.concatenate((tseries_gsp, tseries_nwp), axis=0))
        ).tolist()

        return [
            LiveInferenceData(
                t,
                nwp.sel(init_time=np.datetime64(t, "ns"), method="ffill"),
                gsp.sel(datetime_gmt=np.datetime64(t, "ns"), method="ffill"),
            )
            for t in tseries
        ]

    def __iter__(self) -> Iterator[LiveInferenceData]:
        for datapoint in self.data_feed:
            yield datapoint


@functional_datapipe("nationalboost_model_inference")
class NationalBoostModelInference(IterDataPipe):
    def __init__(
        self,
        model_loader: Callable[[], Callable[[pd.DataFrame], ModelOutput]],
        data_feed: IterDataPipe,
        data_processor: Callable[[LiveInferenceData], pd.DataFrame],
    ) -> None:
        self.model_loader = model_loader
        self.model = self.model_loader()
        self.data_feed = data_feed
        self.data_processor = data_processor

    def __iter__(self) -> Iterator[ModelOutput]:
        for data in self.data_feed:
            processed = self.data_processor(data)
            if processed is not None:  # model may need a warm up period to start
                output: ModelOutput = self.model(processed)
                yield output
