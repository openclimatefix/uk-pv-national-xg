"""Datafeeds for model inference"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import xarray as xr
from ocf_datapipes.production.xgnational import xgnational_production
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@dataclass
class DataInput:
    """Input representation from Database read."""

    nwp: xr.Dataset
    gsp: xr.Dataset
    forecast_intitation_datetime_utc: np.datetime64


@functional_datapipe("mock_datafeed")
class MockDataFeed(IterDataPipe):
    """Mock Data Feed to simulate reading from a database of NWP and GSP values"""

    def __init__(self, nwp_data: xr.Dataset, national_gsp_data: xr.Dataset) -> None:
        """Setup the mock data feed, pre-initalisation

        Args:
            nwp_data (xr.Dataset): Dataset of NWP variables
            national_gsp_data (xr.Dataset): Dataset of GSP PV data
        """
        self.nwp = nwp_data
        self.national_gsp_data = national_gsp_data

    def initialise(self):
        """Create a data feed."""
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
        assert tseries_nwp[0] == tseries_gsp[0], "Datasets must start at the same point in time"

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
        """Iteratively supply data from data feed.

        Yields:
            Iterator[DataInput]: Data, as read from a database
        """
        assert self.data_feed is not None
        for datapoint in self.data_feed:
            yield datapoint


@functional_datapipe("production_datafeed")
class ProductionDataFeed(IterDataPipe):
    """DataPipe reading NWP and GSP values from ocf_datapipes function"""

    def __init__(self, path_to_configuration_file: Union[str, Path]) -> None:
        """Datafeed."""
        self.path_to_configuration_file = path_to_configuration_file

    def __iter__(self) -> Iterator[DataInput]:
        """Returns a single observation of NWP data and 24 hours of GSP data.

        Yields:
            Iterator[DataInput]: Input data for model feature generation.
        """
        data = xgnational_production(self.path_to_configuration_file)
        yield DataInput(
            nwp=data["nwp"], gsp=data["gsp"], forecast_intitation_datetime_utc=np.datetime64("now")
        )
