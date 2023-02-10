""" Configure and load test NWP/PV data """
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

PATH_TO_NWP_TEST_DATASET = (
    Path(__file__).parents[1] / "data" / "test" / "nwp_single_observation.zarr"
)


@pytest.fixture(scope="session")
def nwp_single_observation() -> xr.Dataset:
    """Loads a sample NWP [x, y] observation from file.

    Returns:
        xr.Dataset: NWP Observation.
    """
    # open a single observation of nwp variable at a specific forecast horizon
    nwp = xr.open_zarr(PATH_TO_NWP_TEST_DATASET)

    return nwp


@pytest.fixture(scope="session")
def mock_gsp_data() -> pd.DataFrame:
    """Build Mock PV Dataset

    Returns:
        pd.DataFrame: Mock PV Dataset with 'target' column
    """
    datetimes = pd.DatetimeIndex(
        pd.date_range(
            np.datetime64("2020-02-02T00:00:00"),
            np.datetime64("2020-02-03T00:00:00"),
            freq="30T",
        )
    )

    data = np.arange(0, len(datetimes))
    mock_gsp_data = pd.DataFrame(
        index=datetimes, data=data, columns=["target"]
    ).sort_index(ascending=False)
    return mock_gsp_data
