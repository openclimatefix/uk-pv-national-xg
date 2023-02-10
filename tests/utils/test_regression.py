import numpy as np
import pandas as pd

from gradboost_pv.models.utils import (
    build_rolling_linear_regression_betas,
    clipped_univariate_linear_regression,
)


def test_clipped_univariate_regression_upper_clip():
    n = 10
    upper_clip_value = 2.0

    input_X = np.arange(n)
    input_y = np.arange(0, (upper_clip_value + 1) * n, upper_clip_value + 1)
    expected_result = upper_clip_value
    result = clipped_univariate_linear_regression(
        input_X, input_y, upper_clip=upper_clip_value, epsilon=0.0
    )

    assert np.isclose(result, expected_result, rtol=1e-6)


def test_clipped_univariate_regression_lower_clip():
    n = 10
    lower_clip_value = 2.0

    input_X = np.arange(n)
    input_y = np.arange(0, -(lower_clip_value + 1) * n, -(lower_clip_value + 1))
    expected_result = lower_clip_value
    result = clipped_univariate_linear_regression(
        input_X, input_y, lower_clip=lower_clip_value, epsilon=0.0
    )

    assert np.isclose(result, expected_result, rtol=1e-6)


def test_clipped_univariate_regression_normal():
    n = 10

    input_X = np.arange(n)
    input_y = np.arange(n)
    expected_result = 1.0
    result = clipped_univariate_linear_regression(input_X, input_y, epsilon=0.0)

    assert np.isclose(result, expected_result, rtol=1e-6)


def test_rolling_univariate_regression():
    X = pd.Series(index=np.arange(10), data=np.arange(10))
    y = pd.Series(index=np.arange(10), data=np.arange(0, 30, 3))

    rolling_results = pd.Series(index=np.arange(3, 10), data=3.0)

    betas = build_rolling_linear_regression_betas(
        X,
        y,
        window_size=3,
        regression_function=lambda x, y: clipped_univariate_linear_regression(
            x, y, epsilon=0.0
        ),
    )

    assert len(betas) == 10
    betas = betas.dropna()
    assert len(betas) == 7
    assert np.allclose(betas, rolling_results, rtol=1e-6)
