from typing import Dict, Tuple

import numpy as np
import xarray as xr
from xgboost import XGBRegressor

from gradboost_pv.models.common import NWP_FPATH, GSP_FPATH
from gradboost_pv.inference.data_feeds import MockDataFeed
from gradboost_pv.inference.models import (
    NationalBoostInferenceModel,
    NationalPVModelConfig,
    Prediction,
    Hour,
)
from gradboost_pv.utils.logger import getLogger

logger = getLogger(__name__)
MOCK_DATE_RANGE = slice(
    np.datetime64("2020-08-02T00:00:00"), np.datetime64("2020-08-09T00:00:00")
)


def _load_model_from_local(forecast_hour_ahead: Hour) -> XGBRegressor:
    _model = XGBRegressor()
    _model.load_model(
        f"/home/tom/dev/gradboost_pv/data/uk_region_model_step_{forecast_hour_ahead}.model"
    )
    return _model


def _load_nwp() -> xr.Dataset:
    return xr.open_zarr(NWP_FPATH)


def _load_gsp() -> xr.Dataset:
    return xr.open_zarr(GSP_FPATH).sel(gsp_id=0)


def _dump_results(predictions: Dict[Hour, Prediction]) -> None:
    logger.info(predictions)


def create_date_range_slice(
    nwp: xr.Dataset,
    gsp: xr.Dataset,
    datetime_range: slice = MOCK_DATE_RANGE,
) -> Tuple[xr.Dataset, xr.Dataset]:
    return nwp.sel(init_time=datetime_range), gsp.sel(datetime_gmt=datetime_range)


def main():
    nwp, gsp = _load_nwp(), _load_gsp()
    nwp, gsp = create_date_range_slice(nwp, gsp)

    data_feed = MockDataFeed(nwp, gsp)

    config = NationalPVModelConfig(
        "mock_inference", overwrite_read_datetime_at_inference=False
    )
    model = NationalBoostInferenceModel(
        config, _load_model_from_local, nwp.coords["x"].values, nwp.coords["y"].values
    )

    model.initialise()

    for _data in data_feed:
        resp = model(_data)
        _dump_results(resp)


if __name__ == "__main__":
    main()
