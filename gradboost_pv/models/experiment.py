from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import dataclasses
import json
import numpy as np
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union
from pathlib import Path

DEFFAULT_HYPARAM_CONFIG = {
    "objective": "reg:squarederror",
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
    "max_depth": 100,
    "max_leaves": 0,
    "min_child_weight": 5,
    "n_estimators": 1_500,
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
    "validate_parameters": 1,
    "verbosity": 1,
}


@dataclass
class ExperimentSummary:
    mse_train_loss: float
    mse_test_loss: float
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

    if save_errors_locally:
        assert errors_local_save_file is not None

    # use 2020 as training period and 2021 as test
    X_train, y_train = X.loc[X.index < "2021-01-01"], y.loc[y.index < "2021-01-01"]
    X_test, y_test = X.loc[X.index >= "2021-01-01"], y.loc[y.index >= "2021-01-01"]

    model = XGBRegressor(**booster_hyperparam_config)
    model.fit(X_train, y_train)

    y_pred_test, y_pred_train = model.predict(X_test), model.predict(X_train)
    train_mse, test_mse = mean_squared_error(y_train, y_pred_train), mean_squared_error(
        y_test, y_pred_test
    )
    train_mae, test_mae = mean_absolute_error(
        y_train, y_pred_train
    ), mean_absolute_error(y_test, y_pred_test)

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
        train_mse,
        test_mse,
        train_mae,
        test_mae,
        model,  # just save the last trained model for nwp
    )


def plot_loss_metrics(results_by_step: dict[int, ExperimentSummary]):
    title_mapping = {
        "MAE Train": lambda x: x.mae_train_loss,
        "MAE Test": lambda x: x.mae_test_loss,
        "MSE Train": lambda x: x.mse_train_loss,
        "MSE Test": lambda x: x.mse_test_loss,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for idx, title in enumerate(title_mapping.keys()):
        row = int(idx > 1)
        col = idx % 2
        data = pd.Series(
            {step: title_mapping[title](r) for step, r in results_by_step.items()}
        )
        axes[row][col].scatter(data.index, data.values)
        axes[row][col].set_title(title)
        axes[row][col].set_xlabel("Forecast Horizon (Hours from init_time_utc)")

    plt.show()


def plot_feature_importances(
    results_by_step: dict[int, ExperimentSummary], forecast_horizons=[1, 12, 24, 34]
):
    assert len(forecast_horizons) == 4, "Plotting built for 4 forecast horizons!!"
    fig, axes = plt.subplots(2, len(forecast_horizons), figsize=(28, 18))

    for param_idx, param in enumerate(["weight", "gain"]):
        for idx, fh in enumerate(forecast_horizons):
            data = pd.DataFrame.from_dict(
                results_by_step[fh]
                .model.get_booster()
                .get_score(importance_type=param),
                orient="index",
            ).sort_values(by=0, ascending=False)
            axes[param_idx][idx].bar(range(len(data)), data.values.flatten())
            axes[param_idx][idx].set_xticks(range(len(data)))
            axes[param_idx][idx].set_xticklabels(data.index, rotation=90)
            axes[param_idx][idx].set_title(
                f"Feat. Importance: {param}, Forecast-Horizon: {fh}"
            )

    plt.show()
