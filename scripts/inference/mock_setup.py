from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
from xgboost import XGBRegressor

from gradboost_pv.models.utils import NWP_FPATH, GSP_FPATH
from gradboost_pv.models.utils import load_nwp_coordinates
from gradboost_pv.inference.data_feeds import MockDataFeed
from gradboost_pv.inference.models import (
    NationalBoostInferenceModel,
    NationalPVModelConfig,
    Hour,
)
from gradboost_pv.inference.run import (
    MockDatabaseConnection,
    NationalBoostModelInference,
)

MOCK_DATE_RANGE = slice(
    np.datetime64("2020-08-02T00:00:00"), np.datetime64("2020-08-04T00:00:00")
)
DEFAULT_PATH_TO_MOCK_DATABASE = (
    Path(__file__).parents[2] / "data" / "mock_inference_database.p"
)


def parse_args():
    parser = ArgumentParser(
        description="Script to run mock inference of NationalPV model using data from GCP."
    )
    parser.add_argument(
        "--path_to_database",
        type=str,
        required=False,
        default=DEFAULT_PATH_TO_MOCK_DATABASE,
    )
    args = parser.parse_args()
    return args


def load_model_from_local(forecast_hour_ahead: Hour) -> XGBRegressor:
    _model = XGBRegressor()
    _model.load_model(
        f"/home/tom/dev/gradboost_pv/data/uk_region_model_step_{forecast_hour_ahead}.model"
    )
    return _model


def load_nwp() -> xr.Dataset:
    return xr.open_zarr(NWP_FPATH)


def load_gsp() -> xr.Dataset:
    return xr.open_zarr(GSP_FPATH).sel(gsp_id=0)


def create_date_range_slice(
    nwp: xr.Dataset,
    gsp: xr.Dataset,
    datetime_range: slice = MOCK_DATE_RANGE,
) -> Tuple[xr.Dataset, xr.Dataset]:
    return nwp.sel(init_time=datetime_range), gsp.sel(datetime_gmt=datetime_range)


def main(path_to_database: Path):
    # load data to feed into mock data feed
    nwp, gsp = load_nwp(), load_gsp()
    nwp, gsp = create_date_range_slice(nwp, gsp)

    data_feed = MockDataFeed(nwp, gsp)
    data_feed.initialise()

    # load in our national pv model
    x, y = load_nwp_coordinates()
    config = NationalPVModelConfig(
        "mock_inference", overwrite_read_datetime_at_inference=False
    )
    model = NationalBoostInferenceModel(
        config, load_model_from_local, nwp.coords["x"].values, nwp.coords["y"].values
    )
    model.initialise()

    # create a mock database to write to
    database_conn = MockDatabaseConnection(path_to_database, overwrite_database=True)

    inference_pipeline = NationalBoostModelInference(model, data_feed, database_conn)
    inference_pipeline.run()

    # open and print the database to console
    database_conn = MockDatabaseConnection(path_to_database, overwrite_database=False)
    database_conn.connect()
    print(database_conn.database.data)


if __name__ == "__main__":
    args = parse_args()
    main(args.path_to_database)
