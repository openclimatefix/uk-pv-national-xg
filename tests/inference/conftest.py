"""Generate s3 client, and model configs for testing"""
from pathlib import Path

import pytest

from gradboost_pv.inference.models import NationalPVModelConfig
from gradboost_pv.models.s3 import build_object_name, create_s3_client, load_model

MODEL_FORECAST_HORIZON_HOUR = 5
PATH_TO_TEST_MODEL_CONFIG = Path(__file__).parents[1] / "configs" / "test_model_config.yaml"


@pytest.fixture
def s3_client():
    """Loads s3 client using env var keys"""
    client = create_s3_client()
    return client


@pytest.fixture
def model(s3_client):
    """Loads a model at example forecast horizon"""
    model = load_model(s3_client, build_object_name(MODEL_FORECAST_HORIZON_HOUR))
    return model


@pytest.fixture
def model_config() -> NationalPVModelConfig:
    """Loads test model configuration from yaml file."""
    return NationalPVModelConfig.load_from_yaml(PATH_TO_TEST_MODEL_CONFIG)
