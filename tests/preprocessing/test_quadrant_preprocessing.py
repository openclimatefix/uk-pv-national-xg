import numpy as np
import xarray as xr

from gradboost_pv.preprocessing.quadrant_downsample import _process_nwp


def test_quadrant_preprocessor(nwp_single_observation: xr.Dataset):
    expected_result = np.array([[282.4833, 280.58603], [282.632, 280.05002]], dtype=np.float32)

    nwp = _process_nwp(nwp_single_observation)
    nwp = nwp.to_array().values.reshape(2, 2)

    assert np.allclose(expected_result, nwp, 1e-6)
