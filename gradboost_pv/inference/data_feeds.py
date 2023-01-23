from typing import Dict, Iterator

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("mock_datafeed")
class MockDataFeed(IterDataPipe):
    def __init__(self, nwp_data: xr.Dataset, national_gsp_data: xr.Dataset) -> None:
        self.data_feed = self.create_data_feed(nwp_data, national_gsp_data)

    def create_data_feed(
        self,
        nwp: xr.Dataset,
        gsp: xr.Dataset,
        pv_shift: np.timedelta64 = np.timedelta64(1, "D"),
    ) -> list[Dict[str, xr.Dataset]]:
        """Create a tuple of individual time-slices for coupled nwp and gsp.

        Since the nwp and gsp are sampled at different frequencies, we realign
        them so that at the tick of one data, we also have the most recent data
        of the other source at hand. This is meant to simulate iteratively
        reading the most recent elements a database at given intervals.

        There are autoregressive features for PV data in downstrem model, so we
        take the most recent day's worth of data.

        Args:
            nwp (xr.Dataset): NWP dataset with "init_time" np.datetime64 coords
            gsp (xr.Dataset): GSP PV data with "datetime_gmt" np.datetime64 coords
        """

        tseries_nwp = nwp.coords["init_time"].values
        tseries_gsp = gsp.coords["datetime_gmt"].values
        assert (
            tseries_nwp[0] == tseries_gsp[0] + pv_shift
        ), "Datasets must start at the same point in time (PV shift backed by 1 day)"

        tseries = np.sort(
            np.unique(
                np.concatenate(
                    (tseries_gsp[tseries_gsp > tseries_nwp[0]], tseries_nwp), axis=0
                )
            )
        ).tolist()

        return [
            {
                "nwp": nwp.sel(init_time=np.datetime64(t, "ns"), method="ffill"),
                "gsp": gsp.sel(
                    datetime_gmt=slice(
                        np.datetime64(t, "ns") - pv_shift,
                        np.datetime64(t, "ns"),
                    )
                ),
            }
            for t in tseries
        ]

    def __iter__(self) -> Iterator[Dict[str, xr.Dataset]]:
        for datapoint in self.data_feed:
            yield datapoint
