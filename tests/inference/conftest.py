"""Generate s3 client, and model configs for testing"""
from pathlib import Path

import pytest
from xgboost import XGBRegressor

from gradboost_pv.inference.models import NationalPVModelConfig
from gradboost_pv.models.s3 import build_object_name, create_s3_client, load_model

MODEL_FORECAST_HORIZON_HOUR = 5
PATH_TO_TEST_MODEL_CONFIG = Path(__file__).parents[1] / "configs" / "test_model_config.yaml"


@pytest.fixture
def mock_model():
    """Loads a model at example forecast horizon"""

    model = XGBRegressor()

    # 33 features
    xtrain = [[1] * 33]
    ytrain = [1]

    model.fit(xtrain, ytrain)

    return model


@pytest.fixture
def model_config() -> NationalPVModelConfig:
    """Loads test model configuration from yaml file."""
    return NationalPVModelConfig.load_from_yaml(PATH_TO_TEST_MODEL_CONFIG)


@pytest.fixture
def model_config_path() -> Path:
    """configuration yaml file."""
    return PATH_TO_TEST_MODEL_CONFIG
