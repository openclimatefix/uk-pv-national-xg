"""Functions for model training"""
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
)
from xgboost import XGBRegressor

ALPHA = np.array([0.1, 0.5, 0.9])

DEFFAULT_HYPARAM_CONFIG = {
    "objective": "reg:quantileerror",
    "booster": "gbtree",
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "colsample_bytree": 0.85,
    "early_stopping_rounds": None,
    "gamma": 0,
    "gpu_id": -1,
    "grow_policy": "depthwise",
    "importance_type": None,
    "interaction_constraints": "",
    "learning_rate": 0.005,
    "max_bin": 256,
    "max_cat_threshold": 64,
    "max_depth": 80,
    "max_leaves": 0,
    "min_child_weight": 5,
    "n_estimators": 1250,
    "n_jobs": -1,
    "num_parallel_tree": 1,
    "predictor": "auto",
    "random_state": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "sampling_method": "uniform",
    "scale_pos_weight": 1,
    "subsample": 0.65,
    "tree_method": "hist",
    "quantile_alpha": ALPHA,
    "validate_parameters": 1,
    "verbosity": 1,
}


@dataclass
class ExperimentSummary:
    """Object for storing basic model train/test results"""

    pinball_train_loss: float
    pinball_test_loss: float
    pinball_train_10_percentile_loss: float
    pinball_test_10_percentile_loss: float
    pinball_train_90_percentile_loss: float
    pinball_test_90_percentile_loss: float
    mae_train_loss: float
    mae_test_loss: float
    model: XGBRegressor


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Class for json dumping a dataclass object to file.

    Stolen from:
    https://stackoverflow.com/questions/51286748/
    make-the-python-json-encoder-support-pythons-new-dataclasses
    """

    def default(self, o):
        """Default encoding"""
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def run_experiment(
    X: pd.DataFrame,
    y: pd.DataFrame,
    booster_hyperparam_config: dict = DEFFAULT_HYPARAM_CONFIG,
    save_errors_locally: bool = False,
    errors_local_save_file: Optional[Union[Path, str]] = None,
) -> ExperimentSummary:
    """Trains and tests XGBoost Regression model.

    Args:
        X (pd.DataFrame): X - Features
        y (pd.DataFrame): y - target, normalised PV data
        booster_hyperparam_config (dict, optional): Model hyperparams,
        defaults to DEFFAULT_HYPARAM_CONFIG.
        save_errors_locally (bool, optional): Defaults to False.
        errors_local_save_file (Optional[Union[Path, str]], optional): Defaults to None.

    Returns:
        ExperimentSummary: Object storing some basic fit/evalutation stats + model.
    """

    if save_errors_locally:
        assert errors_local_save_file is not None

    # use 2020 as training period and 2021 as test
    X_train, y_train = X.loc[X.index < "2021-01-01"], y.loc[y.index < "2021-01-01"]
    X_test, y_test = X.loc[X.index >= "2021-01-01"], y.loc[y.index >= "2021-01-01"]

    model = XGBRegressor(**booster_hyperparam_config)
    model.fit(X_train, y_train)

    y_pred_test, y_pred_train = model.predict(X_test), model.predict(X_train)

    train_pinballs = []
    test_pinballs = []
    for idx, alpha in enumerate(ALPHA):
        y_pred_train_alpha = y_pred_train[:, idx]
        y_pred_test_alpha = y_pred_test[:, idx]
        train_pinball, test_pinball = mean_pinball_loss(
            y_train, y_pred_train_alpha, alpha=alpha
        ), mean_pinball_loss(y_test, y_pred_test_alpha, alpha=alpha)
        train_pinballs.append(train_pinball)
        test_pinballs.append(test_pinball)
    y_pred_train = y_pred_train[:, 1]
    y_pred_test = y_pred_test[:, 1]
    train_mae, test_mae = mean_absolute_error(y_train, y_pred_train), mean_absolute_error(
        y_test, y_pred_test
    )

    if save_errors_locally:
        errors_test = pd.DataFrame(
            data=np.concatenate(
                [
                    (y_test.values - y_pred_test.reshape(-1, 1)) ** 2,
                    np.abs(y_test.values - y_pred_test.reshape(-1, 1)),
                ],
                axis=1,
            ),
            columns=["test_mse", "test_mae"],
            index=y_test.index,
        )
        errors_train = pd.DataFrame(
            data=np.concatenate(
                [
                    (y_train.values - y_pred_train.reshape(-1, 1)) ** 2,
                    np.abs(y_train.values - y_pred_train.reshape(-1, 1)),
                ],
                axis=1,
            ),
            columns=["train_mse", "train_mae"],
            index=y_train.index,
        )

        errors = pd.concat([errors_train, errors_test], axis=1)
        errors.to_pickle(errors_local_save_file)

    return ExperimentSummary(
        train_pinballs[1],
        test_pinballs[1],
        train_pinballs[0],
        test_pinballs[0],
        train_pinballs[2],
        test_pinballs[2],
        train_mae,
        test_mae,
        model,  # just save the last trained model for nwp
    )


def plot_loss_metrics(results_by_step: dict[int, ExperimentSummary]):
    """Convenience function for plotting loss metrics over forecast horizons"""
    title_mapping = {
        "MAE Median Train": lambda x: x.mae_train_loss,
        "MAE Median Test": lambda x: x.mae_test_loss,
    }
    title_mapping2 = {
        "0.5 Train": lambda x: x.pinball_train_loss,
        "0.5 Test": lambda x: x.pinball_test_loss,
        "0.1 Train": lambda x: x.pinball_train_10_percentile_loss,
        "0.1 Test": lambda x: x.pinball_test_10_percentile_loss,
        "0.9 Train": lambda x: x.pinball_train_90_percentile_loss,
        "0.9 Test": lambda x: x.pinball_test_90_percentile_loss,
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    for idx, title in enumerate(title_mapping.keys()):
        # Put each metric on a different row
        col = idx
        data = pd.Series({step: title_mapping[title](r) for step, r in results_by_step.items()})
        axes[0][col].scatter(data.index, data.values)
        axes[0][col].set_title(title)
        axes[0][col].set_xlabel("Forecast Horizon (Hours from init_time_utc)")
    for idx, title in enumerate(title_mapping2.keys()):
        if "Test" in title:
            continue
        # Put each metric on a different row
        data = pd.Series({step: title_mapping2[title](r) for step, r in results_by_step.items()})
        axes[1][0].scatter(data.index, data.values, label=title.split(" ")[0].strip())
        # Add title to legend for axis

    axes[1][0].set_title("Pinball Losses Train")
    axes[1][0].set_xlabel("Forecast Horizon (Hours from init_time_utc)")
    axes[1][0].legend()
    for idx, title in enumerate(title_mapping2.keys()):
        if "Train" in title:
            continue
        # Put each metric on a different row
        col = idx
        data = pd.Series({step: title_mapping2[title](r) for step, r in results_by_step.items()})
        axes[1][1].scatter(data.index, data.values, label=title.split(" ")[0].strip())
    axes[1][1].set_title("Pinball Losses Test")
    axes[1][1].set_xlabel("Forecast Horizon (Hours from init_time_utc)")
    axes[1][1].legend()

    plt.show()


def plot_feature_importances(
    results_by_step: dict[int, ExperimentSummary], forecast_horizons=[1, 12, 24, 34]
):
    """Convenience function for plotting feature importances over several forecast horizons"""
    fig, axes = plt.subplots(2, len(forecast_horizons), figsize=(28, 18))

    for param_idx, param in enumerate(["weight", "gain"]):
        for idx, fh in enumerate(forecast_horizons):
            data = pd.DataFrame.from_dict(
                results_by_step[fh].model.get_booster().get_score(importance_type=param),
                orient="index",
            ).sort_values(by=0, ascending=False)
            axes[param_idx][idx].bar(range(len(data)), data.values.flatten())
            axes[param_idx][idx].set_xticks(range(len(data)))
            axes[param_idx][idx].set_xticklabels(data.index, rotation=90)
            axes[param_idx][idx].set_title(f"Feat. Importance: {param}, Forecast-Horizon: {fh}")

    plt.show()
