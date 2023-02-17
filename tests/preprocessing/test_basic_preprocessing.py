import numpy as np
import xarray as xr

from gradboost_pv.preprocessing.basic import _process_nwp


def test_basic_preprocessor(nwp_single_observation: xr.Dataset):
    processed_nwp_obs = _process_nwp(nwp_single_observation).to_array().values

    assert np.isclose(processed_nwp_obs, np.array([281.43784]), 1e-6)
