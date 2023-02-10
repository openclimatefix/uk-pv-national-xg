"""Script for Hyperparameter gridsearch"""
import pickle
from argparse import ArgumentParser
from typing import List, Tuple

import pandas as pd
import xarray as xr
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from gradboost_pv.models.region_filtered import build_datasets_from_local, load_all_variable_slices
from gradboost_pv.models.utils import GSP_FPATH, NWP_FPATH

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


def build_ts_data_cv_splitting(X: pd.DataFrame, n_splits: int, val_size: int) -> List[Tuple]:
    """Timeseries Cross Validation Splitter"""
    X_tests = range(0, len(X) - val_size, int((len(X) - val_size) / n_splits))
    prev_idx = 0
    indicies = list()

    for idx in X_tests[1:]:
        indicies.append((list(range(prev_idx, idx)), list(range(idx, idx + val_size))))

    return indicies


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description="Script for gridsearching model hyperparameters")
    parser.add_argument(
        "--path_to_processed_nwp",
        type=str,
        required=True,
        help="Directory to save collated data.",
    )
    args = parser.parse_args()
    return args


def main(path_to_processed_nwp: str):
    """Runs gridsearch on region-masked based model,

    Logic can be extended to other models also.

    Args:
        path_to_processed_nwp (str): Path for processed NWP-region masked data
    """
    gsp = xr.open_zarr(GSP_FPATH)
    nwp = xr.open_zarr(NWP_FPATH)

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

    step = 24
    X = load_all_variable_slices(path_to_processed_nwp, step)
    X, y = build_datasets_from_local(X, gsp, nwp.coords["step"].values[step])
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

    scores.to_pickle("/home/tom/local_data/xgboost_uk_region_grid_search_results.pickle")
    pickle.dump(
        gsearch.best_estimator_,
        open("/home/tom/local_data/xgboost_uk_region_best_model.pickle", "wb"),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.path_to_processed_nwp)
