import numpy as np
import xarray as xr


def test_nwp_single_observation(nwp_single_observation: xr.Dataset):
    """Load in a single observation shipped with code, check processing works
    for this specific observation. Validate it is the observation we would expect.

    Args:
        nwp_single_observation (_type_): xr.Dataset
    """
    # check only one observation, no other dimesions than spatial
    assert nwp_single_observation.dims == {"x": 548, "y": 704}

    # check all the slicing details are correct
    assert nwp_single_observation.coords["variable"] == np.array("t", dtype="<U1")
    assert nwp_single_observation.coords["init_time"] == np.datetime64(
        "2020-01-04T18:00:00.000000000"
    )
    assert nwp_single_observation.coords["step"] == np.timedelta64(18_000_000_000_000, "ns")
