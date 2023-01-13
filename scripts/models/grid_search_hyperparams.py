import numpy as np
import pandas as pd
import xarray as xr
from pandas.core.dtypes.common import is_datetime64_dtype
import numpy.typing as npt
from typing import Union, Tuple, List
import pickle
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


nwp_path = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr/"
)
gsp_path = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"

TRIG_DATETIME_FEATURE_NAMES = [
    "SIN_MONTH",
    "COS_MONTH",
    "SIN_DAY",
    "COS_DAY",
    "SIN_HOUR",
    "COS_HOUR",
]


def _get_path_to_uk_region_data_data(
    variable: str, forecast_horizon: int, inner: bool
) -> str:
    if inner:
        return f"/home/tom/local_data/uk_region_mean_var{variable}_step{forecast_horizon}.npy"
    else:
        return f"/home/tom/local_data/outer_region_mean_var{variable}_step{forecast_horizon}.npy"


def _trig_transform(
    values: np.ndarray, period: Union[float, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of values and an upper limit on the values, compute trig decomposition.
    Args:
        values: ndarray of points in the range [0, period]
        period: period of the data
    Returns:
        Decomposition of values into sine and cosine of data with given period
    """

    return np.sin(values * 2 * np.pi / period), np.cos(values * 2 * np.pi / period)


def trigonometric_datetime_transformation(datetimes: npt.ArrayLike) -> np.ndarray:
    """
    Given an iterable of datetimes, returns a trigonometric decomposition on hour, day and month
    Args:init_time=30,
        datetimes: ArrayLike of datetime64 values
    Returns:
        Trigonometric decomposition of datetime into hourly, daily and
        monthly values.
    """
    assert is_datetime64_dtype(
        datetimes
    ), "Data for Trig Decomposition must be np.datetime64 type"

    datetimes = pd.DatetimeIndex(datetimes)
    hour = datetimes.hour.values.reshape(-1, 1) + (
        datetimes.minute.values.reshape(-1, 1) / 60
    )
    day = datetimes.day.values.reshape(-1, 1)
    month = datetimes.month.values.reshape(-1, 1)

    sine_hour, cosine_hour = _trig_transform(hour, 24)
    sine_day, cosine_day = _trig_transform(day, 366)
    sine_month, cosine_month = _trig_transform(month, 12)

    return np.concatenate(
        [sine_month, cosine_month, sine_day, cosine_day, sine_hour, cosine_hour], axis=1
    )


DEFAULT_UK_REGION_NWP_VARS = ["dswrf", "hcct", "t", "lcc", "sde"]


# def build_datasets_from_local(
#     eval_timeseries: pd.DatetimeIndex,
#     step: int,
#     variables: list[str] = DEFAULT_UK_REGION_NWP_VARS,
#     nan_to_zero: bool = False,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:

#     X = list()
#     for var in variables:
#         X.append(np.load(_get_path_to_uk_region_data_data(var, step, True)))
#         X.append(np.load(_get_path_to_uk_region_data_data(var, step, False)))
#     X = np.concatenate(X, axis=0).T
#     X = np.nan_to_num(X) if nan_to_zero else X

#     columns = []
#     for var in variables:
#         columns += [f"{var}_{region}" for region in ["within", "outer"]]

#     X = pd.DataFrame(data=X, columns=columns, index=eval_timeseries)
#     y = pd.DataFrame(
#         gsp["generation_mw"] / gsp["installedcapacity_mwp"],
#         index=eval_timeseries,
#         columns=["target"],
#     )

#     # shift y by the step forecast
#     shift = nwp.step.values[step]
#     y = y.shift(freq=-shift).dropna()
#     common_index = sorted(pd.DatetimeIndex((set(y.index).intersection(X.index))))

#     X, y = X.loc[common_index], y.loc[common_index]

#     # add datetime methods for the point at which we are forecasting e.g. now + step
#     _X = trigonometric_datetime_transformation(
#         y.shift(freq=nwp.step.values[step]).index.values
#     )
#     _X = pd.DataFrame(_X, index=y.index, columns=TRIG_DATETIME_FEATURE_NAMES)
#     X = pd.concat([X, _X], axis=1)

#     # add lagged values of GSP PV
#     ar_1 = y.shift(freq=-(shift + np.timedelta64(1, "h")))
#     ar_day = y.shift(freq=-(shift + np.timedelta64(1, "D")))
#     ar_1.columns = ["PV_LAG_1HR"]
#     ar_day.columns = ["PV_LAG_DAY"]

#     # estimate linear trend of the PV
#     window_size = 10
#     epsilon = 0.01
#     y_covariates = y.shift(freq=-(shift + np.timedelta64(2, "h")))
#     y_covariates.columns = ["x"]
#     y_target = y.shift(freq=-(shift + np.timedelta64(1, "h")))
#     y_target.columns = ["y"]
#     data = pd.concat([y_target, y_covariates], axis=1).dropna()
#     _x = data["x"].values
#     _y = data["y"].values
#     _betas = np.nan * np.empty(len(data))

#     for n in range(window_size, len(data)):
#         __y = _y[(n - window_size) : n]
#         __x = _x[(n - window_size) : n]
#         __b = max(min((1 / ((__x.T @ __x) + epsilon)) * (__x.T @ __y), 10), -10)
#         _betas[n] = __b

#     betas = pd.DataFrame(data=_betas, columns=["AR_Beta"], index=data.index)

#     X = pd.concat([X, ar_1, ar_day, betas], axis=1).dropna()
#     y = y.loc[X.index]

#     return X, y


def build_ts_data_cv_splitting(
    X: pd.DataFrame, n_splits: int, val_size: int
) -> List[Tuple]:
    X_tests = range(0, len(X) - val_size, int((len(X) - val_size) / n_splits))
    prev_idx = 0
    indicies = list()

    for idx in X_tests[1:]:
        indicies.append((list(range(prev_idx, idx)), list(range(idx, idx + val_size))))

    return indicies


NON_SEARCH_PARAMS = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "early_stopping_rounds": None,
    "gamma": 0,
    "gpu_id": -1,
    "importance_type": None,
    "interaction_constraints": "",
    "max_bin": 256,
    "max_cat_threshold": 64,
    "max_leaves": 0,
    "n_jobs": -1,
    "num_parallel_tree": 1,
    "predictor": "auto",
    "random_state": 0,
    "sampling_method": "uniform",
    "scale_pos_weight": 1,
    "tree_method": "hist",
    "validate_parameters": 1,
    "verbosity": 1,
}


PARAM_GRID = {
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "max_depth": [50, 100, 300],
    "min_child_weight": [5, 50],
    "n_estimators": [300, 1_000, 2_000],
    "subsample": [0.65, 0.8, 0.95],
    "reg_alpha": [0, 1, 5],
    "reg_lambda": [0, 1, 5],
    "grow_policy": ["depthwise", "leafwise"],
    "gamma": [0, 5],
    "colsample_bytree": [0.75, 0.9],
}


def main():
    def build_datasets_from_local(
        eval_timeseries: pd.DatetimeIndex,
        step: int,
        variables: list[str] = DEFAULT_UK_REGION_NWP_VARS,
        nan_to_zero: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        X = list()
        for var in variables:
            X.append(np.load(_get_path_to_uk_region_data_data(var, step, True)))
            X.append(np.load(_get_path_to_uk_region_data_data(var, step, False)))
        X = np.concatenate(X, axis=0).T
        X = np.nan_to_num(X) if nan_to_zero else X

        columns = []
        for var in variables:
            columns += [f"{var}_{region}" for region in ["within", "outer"]]

        X = pd.DataFrame(data=X, columns=columns, index=eval_timeseries)
        y = pd.DataFrame(
            gsp["generation_mw"] / gsp["installedcapacity_mwp"],
            index=eval_timeseries,
            columns=["target"],
        )

        # shift y by the step forecast
        shift = nwp.step.values[step]
        y = y.shift(freq=-shift).dropna()
        common_index = sorted(pd.DatetimeIndex((set(y.index).intersection(X.index))))

        X, y = X.loc[common_index], y.loc[common_index]

        # add datetime methods for the point at which we are forecasting e.g. now + step
        _X = trigonometric_datetime_transformation(
            y.shift(freq=nwp.step.values[step]).index.values
        )
        _X = pd.DataFrame(_X, index=y.index, columns=TRIG_DATETIME_FEATURE_NAMES)
        X = pd.concat([X, _X], axis=1)

        # add lagged values of GSP PV
        ar_1 = y.shift(freq=-(shift + np.timedelta64(1, "h")))
        ar_day = y.shift(freq=-(shift + np.timedelta64(1, "D")))
        ar_1.columns = ["PV_LAG_1HR"]
        ar_day.columns = ["PV_LAG_DAY"]

        # estimate linear trend of the PV
        window_size = 10
        epsilon = 0.01
        y_covariates = y.shift(freq=-(shift + np.timedelta64(2, "h")))
        y_covariates.columns = ["x"]
        y_target = y.shift(freq=-(shift + np.timedelta64(1, "h")))
        y_target.columns = ["y"]
        data = pd.concat([y_target, y_covariates], axis=1).dropna()
        _x = data["x"].values
        _y = data["y"].values
        _betas = np.nan * np.empty(len(data))

        for n in range(window_size, len(data)):
            __y = _y[(n - window_size) : n]
            __x = _x[(n - window_size) : n]
            __b = max(min((1 / ((__x.T @ __x) + epsilon)) * (__x.T @ __y), 10), -10)
            _betas[n] = __b

        betas = pd.DataFrame(data=_betas, columns=["AR_Beta"], index=data.index)

        X = pd.concat([X, ar_1, ar_day, betas], axis=1).dropna()
        y = y.loc[X.index]

        return X, y

    gsp = xr.open_zarr(gsp_path)
    nwp = xr.open_zarr(nwp_path)

    evaluation_timeseries = (
        gsp.coords["datetime_gmt"]
        .where(
            (gsp["datetime_gmt"] >= nwp.coords["init_time"].values[0])
            & (gsp["datetime_gmt"] <= nwp.coords["init_time"].values[-1]),
            drop=True,
        )
        .values
    )

    gsp = gsp.sel(datetime_gmt=evaluation_timeseries, gsp_id=0)
    X, y = build_datasets_from_local(
        evaluation_timeseries, 30
    )  # choose forecast horizon 30 for the CV
    _cv = build_ts_data_cv_splitting(X, 5, 2_000)

    gsearch = GridSearchCV(
        XGBRegressor(**NON_SEARCH_PARAMS),
        param_grid=PARAM_GRID,
        scoring="neg_mean_absolute_error",
        verbose=1,
        cv=_cv,
    )

    results = gsearch.fit(X, y)  # run the grid search

    # save the results
    scores = pd.concat(
        [
            pd.DataFrame(results.cv_results_["params"]),
            pd.DataFrame(results.cv_results_["mean_test_score"], columns=["NegMAE"]),
        ],
        axis=1,
    )

    scores.to_pickle("/home/tom/local_data/xgboost_uk_region_grid_search_results.p")
    pickle.dump(
        gsearch.best_estimator_,
        open("/home/tom/local_data/xgboost_uk_region_best_model.p", "wb"),
    )


if __name__ == "__main__":
    main()
