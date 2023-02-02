from dataclasses import dataclass
from typing import Iterator

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@dataclass
class DataInput:
    nwp: xr.Dataset
    gsp: xr.Dataset
    forecast_intitation_datetime_utc: np.datetime64


@functional_datapipe("mock_datafeed")
class MockDataFeed(IterDataPipe):
    def __init__(self, nwp_data: xr.Dataset, national_gsp_data: xr.Dataset) -> None:
        self.nwp = nwp_data
        self.national_gsp_data = national_gsp_data

    def initialise(self):
        self.data_feed = self.create_data_feed(self.nwp, self.national_gsp_data)

    def create_data_feed(
        self,
        nwp: xr.Dataset,
        gsp: xr.Dataset,
        gsp_added_history: np.timedelta64 = np.timedelta64(24, "h"),
    ) -> list[DataInput]:
        """Create a tuple of individual time-slices for coupled nwp and gsp.

        Since the nwp and gsp are sampled at different frequencies, we realign
        them so that at the tick of one data, we also have the most recent data
        of the other source at hand. This is meant to simulate iteratively
        reading the most recent elements a database at given intervals.

        Args:
            nwp (xr.Dataset): NWP dataset with "init_time" np.datetime64 coords
            gsp (xr.Dataset): GSP PV data with "datetime_gmt" np.datetime64 coords
            gsp_added_history (np.timedelta64): Instead of one obs per timestamp, supply
            a rolling history of pv data.
        """

        tseries_nwp = nwp.coords["init_time"].values
        tseries_gsp = gsp.coords["datetime_gmt"].values
        assert (
            tseries_nwp[0] == tseries_gsp[0]
        ), "Datasets must start at the same point in time"

        tseries = np.sort(
            np.unique(
                np.concatenate(
                    (
                        tseries_gsp[tseries_gsp >= tseries_gsp[0] + gsp_added_history],
                        tseries_nwp[tseries_nwp >= tseries_nwp[0] + gsp_added_history],
                    ),
                    axis=0,
                )
            )
        )

        return [
            DataInput(
                nwp.sel(init_time=np.datetime64(t, "ns"), method="ffill"),
                gsp.sel(
                    datetime_gmt=slice(
                        np.datetime64(t, "ns") - gsp_added_history,
                        np.datetime64(t, "ns"),
                    )
                ),
                forecast_intitation_datetime_utc=t,
            )
            for t in tseries
        ]

    def __iter__(self) -> Iterator[DataInput]:
        assert self.data_feed is not None
        for datapoint in self.data_feed:
            yield datapoint
