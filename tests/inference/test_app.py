from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from xgboost import XGBRegressor

import gradboost_pv
from gradboost_pv.inference.data_feeds import DataInput
from gradboost_pv.inference.models import (
    NationalBoostInferenceModel,
    NationalPVModelConfig,
    Prediction,
)
from gradboost_pv.inference.run import NationalBoostModelInference
from gradboost_pv.models.utils import load_nwp_coordinates

PATH_TO_TEST_DATA_DIRECTORY = Path(gradboost_pv.__file__).parents[1] / "data" / "test"
PATH_TO_SAMPLE_NWP = PATH_TO_TEST_DATA_DIRECTORY / "sample_prod_nwp.zarr"
PATH_TO_SAMPLE_GSP = PATH_TO_TEST_DATA_DIRECTORY / "sample_prod_gsp.zarr"


@pytest.fixture
def sample_prod_nwp_data() -> xr.Dataset:
    """Loads a sample NWP [x, y] observation from file.

    This data looks like the (processed) NWP data from prod aws.
    Returns:
        xr.Dataset: NWP Observation.
    """
    nwp = xr.open_zarr(PATH_TO_SAMPLE_NWP)
    return nwp


@pytest.fixture
def sample_prod_gsp_data() -> xr.Dataset:
    """Loads a sample GSP observation from file.

    This data looks like the prod GSP data read from database.
    Returns:
        xr.Dataset: GSP Observations.
    """
    gsp = xr.open_zarr(PATH_TO_SAMPLE_GSP)
    return gsp


@functional_datapipe("mock_production_datafeed")
class MockProdDataPipe(IterDataPipe):
    """Mock Datapipe for inference testing"""

    def __init__(self, nwp_data: xr.Dataset, gsp_data: xr.Dataset) -> None:
        self.nwp = nwp_data
        self.gsp = gsp_data
        self.forecast_time = np.datetime64("2023-02-21T11:28:43")  # hardcoded

    def __iter__(self) -> Iterator[DataInput]:
        yield DataInput(
            nwp=self.nwp, gsp=self.gsp, forecast_intitation_datetime_utc=self.forecast_time
        )


@pytest.fixture
def mock_prod_datafeed(sample_prod_nwp_data, sample_prod_gsp_data) -> MockProdDataPipe:
    """Instantiate a Mock datapipe for inference testing."""
    return MockProdDataPipe(sample_prod_nwp_data, sample_prod_gsp_data)


class MockTestingDatabase:
    """Mock database for inference testing"""

    def __init__(self) -> None:
        """Initialise empty database"""
        self.outputs = list()

    def write(self, data):
        """Write prediction data to database"""
        self.outputs.append(data)


class MockTestingDatabaseConnection:
    """Mock database connection for inference testing"""

    def __init__(self) -> None:
        """Initialise connection to new database"""
        self.database = MockTestingDatabase()

    def __enter__(self) -> MockTestingDatabase:
        """Open database connection"""
        return self.database

    def __exit__(self, type, value, traceback) -> None:
        """Close connection"""
        return


@pytest.fixture
def mock_testing_database_connection() -> MockTestingDatabaseConnection:
    """Instantiate a mock testing database for inference testing"""
    return MockTestingDatabaseConnection()


@pytest.fixture
def nwp_coords():
    """Return x, y coordinates of NWP data for region-masking"""
    return load_nwp_coordinates()


@pytest.fixture
def testing_inference_model(
    model_config: NationalPVModelConfig, nwp_coords, mock_model
) -> NationalBoostInferenceModel:
    """Set up model for inference pipeline testing

    Args:
        model_config (NationalPVModelConfig): configuration for testing
        nwp_coords (_type_): coordinates loaded as fixture
        s3_client (_type_): s3 client loaded as fixture

    Returns:
        NationalBoostInferenceModel: model object for inference pipeline
    """
    x, y = nwp_coords

    def _model_loader_by_hour(hour) -> XGBRegressor:
        return mock_model

    model = NationalBoostInferenceModel(model_config, _model_loader_by_hour, x, y)
    model.initialise()
    return model


# @pytest.mark.skip("Currently no access to AWS")
def test_inference(testing_inference_model, mock_prod_datafeed, mock_testing_database_connection):
    """Test for model inference pipeline

    Args:
        testing_inference_model (_type_): Initialised model
        mock_prod_datafeed (_type_): Datafeed of NWP and GSP data
        mock_testing_database_connection (_type_): Connection to output database
    """
    pipeline = NationalBoostModelInference(
        model=testing_inference_model,
        data_feed=mock_prod_datafeed,
        database_connection=mock_testing_database_connection,
    )

    pipeline.run()

    results = mock_testing_database_connection.database.outputs
    assert len(results) == 1, "Should only be computing one batch of predictions"
    assert all(
        [
            isinstance(x, Prediction)
            for x in list(mock_testing_database_connection.database.outputs[0].values())
        ]
    ), "Did not create predictions for the model"
