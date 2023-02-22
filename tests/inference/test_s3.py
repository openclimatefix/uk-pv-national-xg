from xgboost import XGBRegressor
import pytest


@pytest.mark.skip('Currently no access to AWS')
def test_client(s3_client):
    """Check client finds buckets"""
    resp = s3_client.list_buckets()
    assert len(resp["Buckets"]) > 0


@pytest.mark.skip('Currently no access to AWS')
def test_model(model):
    """Check loaded model is XGBoost model."""
    assert isinstance(model, XGBRegressor)
