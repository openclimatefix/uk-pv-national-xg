"""Functions for model training"""
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ocf_ml_metrics.evaluation.evaluation import evaluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
)
from xgboost import XGBRegressor

"""
Dump from training on 2021 and testing on 2020

```
0
[15:48:15] WARNING: /home/jacob/Development/xgboost/src/learner.cc:753:
Parameters: { "scale_pos_weight" } are not used.

Percentile: 0.1, train pinball: 0.00282, percentile count: 0.53132 non-night percentile count: 0.11743
Percentile: 0.5, train pinball: 0.005, percentile count: 0.74744 non-night percentile count: 0.52492
Percentile: 0.9, train pinball: 0.0038, percentile count: 0.92279 non-night percentile count: 0.85077
Number of positive values for actual - 90th percentile: 3084/17568
Number of positive values for actual - 10th percentile: 6861/17568
Number of positive values for actual - median: 4916/17568
Number of positive values for actual - 90th percentile: 2994/8008
Number of positive values for actual - 10th percentile: 6377/8008
Number of positive values for actual - median: 4620/8008
Median test MAE: 0.01


1
[15:49:16] WARNING: /home/jacob/Development/xgboost/src/learner.cc:753:
Parameters: { "scale_pos_weight" } are not used.

Percentile: 0.1, train pinball: 0.00388, percentile count: 0.53286 non-night percentile count: 0.12085
Percentile: 0.5, train pinball: 0.0074, percentile count: 0.75453 non-night percentile count: 0.54109
Percentile: 0.9, train pinball: 0.00658, percentile count: 0.92312 non-night percentile count: 0.85024
Number of positive values for actual - 90th percentile: 3394/17568
Number of positive values for actual - 10th percentile: 6769/17568
Number of positive values for actual - median: 5056/17568
Number of positive values for actual - 90th percentile: 3329/8008
Number of positive values for actual - 10th percentile: 6281/8008
Number of positive values for actual - median: 4762/8008
Median test MAE: 0.0148
[INFO][2023-05-15 03:50:41] : Trained model for 1 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0011104914595641766, pinball_test_loss=0.007401900459080166, pinball_train_10_percentile_loss=0.0016597589380497257, pinball_test_10_percentile_loss=0.0038831984857985538, pinball_train_90_percentile_loss=0.0007042405500994957, pinball_test_90_percentile_loss=0.006581151385691185, mae_train_loss=0.002220982919128353, mae_test_loss=0.014803800918160331, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
2

```
"""

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
    forecast_hour: int = 0,
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
    X_train = X_train.dropna()
    X_test = X_test.dropna()
    y_train = y_train.fillna(0.0)
    y_test = y_test.fillna(0.0)
    print(X_test.keys())
    print(y_test.keys())
    print(len(X_test))
    print(y_test)
    print(X_test)
    print(y_test.index)
    exit()
    print(len(y_test))
    print(len(X_train))
    print(len(y_train))
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

    percentile_counts = []
    non_night_percentiles = []
    for idx, alpha in enumerate(ALPHA):
        y_pred_test_alpha = y_pred_train[:, idx]
        percentile_counts.append(np.sum(y_train["target"].values < y_pred_test_alpha))
        non_night_percentiles.append(
            np.sum(
                (y_train["target"].values < y_pred_test_alpha) & (y_train["target"].values > 0.01)
            )
        )
    # Get percentage of total test data that is below each percentile
    percentile_counts = np.array(percentile_counts) / len(y_train["target"].values)
    non_night_percentiles = np.array(non_night_percentiles) / len(
        y_train[y_train["target"].values > 0.01]["target"].values
    )
    # Print out each of the pinball test losses and percentile counts
    for idx, alpha in enumerate(ALPHA):
        print(
            f"Percentile: {alpha}, train pinball: {np.round(train_pinballs[idx], 5)}, percentile count: {np.round(percentile_counts[idx], 5)} "
            f"non-night percentile count: {np.round(non_night_percentiles[idx], 5)}"
        )
    print(
        f"Number of positive values for actual - 90th percentile: {np.sum((y_test['target'].values - y_pred_test[:, 2]) > 0.)}/{len(y_test['target'].values)}"
    )
    print(
        f"Number of positive values for actual - 10th percentile: {np.sum((y_test['target'].values - y_pred_test[:, 0]) > 0.)}/{len(y_test['target'].values)}"
    )
    print(
        f"Number of positive values for actual - median: {np.sum((y_test['target'].values - y_pred_test[:, 1]) > 0.)}/{len(y_test['target'].values)}"
    )
    percentile_counts = []
    non_night_percentiles = []
    for idx, alpha in enumerate(ALPHA):
        y_pred_test_alpha = y_pred_test[:, idx]
        percentile_counts.append(np.sum(y_test["target"].values < y_pred_test_alpha))
        non_night_percentiles.append(
            np.sum((y_test["target"].values < y_pred_test_alpha) & (y_test["target"].values > 0.01))
        )
    # Get percentage of total test data that is below each percentile
    percentile_counts = np.array(percentile_counts) / len(y_test["target"].values)
    non_night_percentiles = np.array(non_night_percentiles) / len(
        y_test[y_test["target"].values > 0.01]["target"].values
    )
    # Print out each of the pinball test losses and percentile counts
    for idx, alpha in enumerate(ALPHA):
        print(
            f"Percentile: {alpha}, test pinball: {np.round(test_pinballs[idx], 5)}, percentile count: {np.round(percentile_counts[idx], 5)} "
            f"non-night percentile count: {np.round(non_night_percentiles[idx], 5)}"
        )
    # Now plot the predictions vs the actuals
    xx = list(range(y_test.shape[0]))
    percent_90 = y_pred_test[:, 2]
    plt.plot(xx, y_test["target"].values, label="Actual")
    plt.plot(xx, percent_90, label="90th Percentile")
    plt.legend()
    plt.title(f"{forecast_hour} Actual vs 90th Percentile Prediction")
    plt.savefig(f"{forecast_hour}_actual_vs_90th_percentile.png")
    plt.cla()
    plt.clf()
    plt.close()
    plt.plot(xx, y_test["target"].values - percent_90)
    plt.title(f"{forecast_hour} Actual minus 90th Percentile Prediction")
    plt.savefig(f"{forecast_hour}_actual_minus_90th_percentile.png")
    plt.cla()
    plt.clf()
    plt.close()
    xx = list(range(1000))
    plt.plot(xx, y_test["target"].values[:1000], label="Actual")
    plt.plot(xx, percent_90[:1000], label="90th Percentile")
    plt.legend()
    plt.title(f"{forecast_hour} Actual vs 90th Percentile Prediction first 1000")
    plt.savefig(f"{forecast_hour}_actual_vs_90th_percentile_first_1000.png")
    plt.cla()
    plt.clf()
    plt.close()
    plt.plot(xx, y_test["target"].values[:1000] - percent_90[:1000])
    plt.title(f"{forecast_hour} Actual minus 90th Percentile Prediction first 1000")
    plt.savefig(f"{forecast_hour}_actual_minus_90th_percentile_first_1000.png")
    plt.cla()
    plt.clf()
    plt.close()
    xx = list(range(200))
    plt.plot(xx, y_test["target"].values[:200], label="Actual")
    plt.plot(xx, percent_90[:200], label="90th Percentile")
    plt.legend()
    plt.title(f"{forecast_hour} Actual vs 90th Percentile Prediction first 200")
    plt.savefig(f"{forecast_hour}_actual_vs_90th_percentile_first_200.png")
    plt.cla()
    plt.clf()
    plt.close()
    # Plot the prediction minus the actual

    # Daytime only
    y_pred_test_daytime = y_pred_test[y_test["target"].values > 0.01]
    y_test_daytime = y_test[y_test["target"].values > 0.01]
    print(
        f"Number of positive values for actual - 90th percentile: {np.sum((y_test_daytime['target'].values - y_pred_test_daytime[:, 2]) > 0.)}/{len(y_test_daytime['target'].values)}"
    )
    print(
        f"Number of positive values for actual - 10th percentile: {np.sum((y_test_daytime['target'].values - y_pred_test_daytime[:, 0]) > 0.)}/{len(y_test_daytime['target'].values)}"
    )
    print(
        f"Number of positive values for actual - median: {np.sum((y_test_daytime['target'].values - y_pred_test_daytime[:, 1]) > 0.)}/{len(y_test_daytime['target'].values)}"
    )
    xx = list(range(y_test_daytime.shape[0]))
    percent_90 = y_pred_test_daytime[:, 2]
    plt.plot(xx, y_test_daytime["target"].values, label="Actual")
    plt.plot(xx, percent_90, label="90th Percentile")
    plt.legend()
    plt.title(f"{forecast_hour} Actual vs 90th Percentile Prediction Daytime")
    plt.savefig(f"{forecast_hour}_actual_vs_90th_percentile_daytime.png")
    plt.cla()
    plt.clf()
    plt.close()
    plt.plot(xx, y_test_daytime["target"].values - percent_90)
    plt.title(f"{forecast_hour} Actual minus 90th Percentile Prediction Daytime")
    plt.savefig(f"{forecast_hour}_actual_minus_90th_percentile_daytime.png")
    plt.cla()
    plt.clf()
    plt.close()
    xx = list(range(1000))
    plt.plot(xx, y_test_daytime["target"].values[:1000], label="Actual")
    plt.plot(xx, percent_90[:1000], label="90th Percentile")
    plt.legend()
    plt.title(f"{forecast_hour} Actual vs 90th Percentile Prediction Daytime first 1000")
    plt.savefig(f"{forecast_hour}_actual_vs_90th_percentile_first_1000_daytime.png")
    plt.cla()
    plt.clf()
    plt.close()
    xx = list(range(200))
    plt.plot(xx, y_test_daytime["target"].values[:200], label="Actual")
    plt.plot(xx, percent_90[:200], label="90th Percentile")
    plt.legend()
    plt.title(f"{forecast_hour} Actual vs 90th Percentile Prediction Daytime first 200")
    plt.savefig(f"{forecast_hour}_actual_vs_90th_percentile_first_200_daytime.png")
    plt.cla()
    plt.clf()
    plt.close()

    y_pred_train = y_pred_train[:, 1]
    y_pred_test = y_pred_test[:, 1]
    train_mae, test_mae = mean_absolute_error(y_train, y_pred_train), mean_absolute_error(
        y_test, y_pred_test
    )
    print(f"Median test MAE: {np.round(test_mae, 5)}")

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

    # Create a dataframe
    # Need to calculate the lat/lon of center of UK
    # Need to calculate datetimes from sin/cos of month, day, and hour, then add forecast_horizon to get target time

    capacity_mw = 13500  # Roughly
    actual_pv_outturn_mw = y_test["target"].to_numpy() * capacity_mw
    predicted_pv_outturn_mw = y_pred_test * capacity_mw  # Median value
    t0_datetime = X_test.index
    target_datetime = y_test.index
    national_ids = np.repeat(national_id, len(t0_datetime))
    latitudes = np.repeat(55.3781, len(t0_datetimes))  # 55.3781
    longitudes = np.repeat(-3.4360, len(t0_datetimes))  # -3.4360
    t0_actual_pv_outturn_mw = X_test["PV_LAG_1HR"].to_numpy()  # Relatively close to current outturn
    capacitys = np.repeat(capacity_mw, len(t0_datetimes))

    # Create dataframe from numpy arrays from above
    results_df = pd.DataFrame(
        national_ids,
        latitudes,
        longitudes,
        t0_datetime,
        target_datetime,
        actual_pv_outturn_mw,
        predicted_pv_outturn_mw,
        t0_actual_pv_outturn_mw,
        capacitys,
        columns=[
            "id",
            "latitude",
            "longitude",
            "t0_datetime_utc",
            "target_datetime_utc",
            "actual_pv_outturn_mw",
            "forecast_pv_outturn_mw",
            "t0_actual_pv_outturn_mw",
            "capacity_mwp",
        ],
    )

    evaluation_results = evaluation(results_df, "national_xg")
    evaluation_results.to_pickle(f"evaluation_results_{forecast_hour}.pkl")
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
