from xgboost import XGBRegressor


def test_client(s3_client):
    """Check client finds buckets"""
    resp = s3_client.list_buckets()
    assert len(resp["Buckets"]) > 0


def test_model(model):
    """Check loaded model is XGBoost model."""
    assert isinstance(model, XGBRegressor)
