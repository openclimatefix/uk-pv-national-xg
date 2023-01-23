from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import time

import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBRegressor

from gradboost_pv.preprocessing.region_filtered import (
    get_eso_uk_multipolygon,
    generate_polygon_mask,
)
from gradboost_pv.models.common import (
    NWP_VARIABLE_NUM,
    NWP_STEP_HORIZON,
    ORDERED_NWP_FEATURE_VARIABLES,
    clipped_univariate_linear_regression,
    DEFAULT_ROLLING_LR_WINDOW_SIZE,
)
from gradboost_pv.models.region_filtered import AUTO_REGRESSION_COVARIATE_LAG


DEFAULT_INSTALLED_CAPACITY_MWP = 13861.203


@dataclass(frozen=True)
class Covariates:
    covariates: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    datetime_of_model_inference_utc: np.datetime64
    datetime_of_target_utc: np.datetime64
    forecast_kw: float


@dataclass
class NationalPVModelConfig:
    name: str
    forecast_horizon_hours: list[int] = range(36)


class BaseInferenceModel(ABC):
    """Abstract model for trained NationalUK PV Prediciton"""

    def __init__(self, config: NationalPVModelConfig) -> None:
        self._config = config

    @property
    def get_config(self):
        return self._config

    @abstractmethod
    def initialise(self):
        pass

    @abstractmethod
    def covariate_transform(self, data) -> Covariates:
        pass

    def predict(self, data):
        X = self.covariate_transform(data)
        return self.predict_from_covariates(X)

    @abstractmethod
    def predict_from_covariates(self, covariates: Covariates) -> Prediction:
        pass

    def __call__(self, data) -> Prediction:
        return self.predict(data)


class NationalBoostInferenceModel(BaseInferenceModel):
    def __init__(
        self,
        config: NationalPVModelConfig,
        model_loader: Callable[[int], XGBRegressor],
        nwp_x_coords: np.ndarray,
        nwp_y_coords: np.ndarray,
    ) -> None:
        self.model_loader = model_loader
        self.nwp_x_coords = nwp_x_coords
        self.nwp_y_coords = nwp_y_coords
        super().__init__(config)

    def initialise(self):
        # load models for each time step from disk.
        self.meta_model = self.load_meta_model()

        # get uk-region mask/polygon
        self.mask = self.load_mask()

    def load_meta_model(self) -> Dict[int, XGBRegressor]:
        return {
            forecast_horizon_hour: self.load_model_per_forecast_horizon(
                forecast_horizon_hour
            )
            for forecast_horizon_hour in self._config.forecast_horizon_hours
        }

    def load_model_per_forecast_horizon(
        self, forecast_horizon_hours: int
    ) -> XGBRegressor:
        return self.model_loader(forecast_horizon_hours)

    def load_mask(self) -> xr.DataArray:
        uk_polygon = get_eso_uk_multipolygon()
        mask = generate_polygon_mask(self.nwp_x_coords, self.nwp_y_coords, uk_polygon)
        mask = xr.DataArray(
            np.tile(mask.T, (NWP_VARIABLE_NUM, NWP_STEP_HORIZON, 1, 1)),
            dims=["variable", "step", "x", "y"],
        )
        return mask

    def covariate_transform(
        self,
        data: Dict[str, xr.Dataset],
        time_of_inference_utc: Optional[np.datetime64] = None,
    ) -> Covariates:
        _start = time.perf_counter()
        #  NWP Processing
        nwp_inner = (
            xr.where(~self.mask.isnull(), data["nwp"], np.nan)
            .mean(dim=["x", "y"])
            .to_array()
            .values.reshape(NWP_VARIABLE_NUM, NWP_STEP_HORIZON)
            .T
        )
        _end = time.perf_counter()
        print(f"inner mask took {_end - _start}")

        _start = time.perf_counter()
        nwp_outer = (
            xr.where(self.mask.isnull(), data["nwp"], np.nan)
            .mean(dim=["x", "y"])
            .to_array()
            .values.reshape(NWP_VARIABLE_NUM, NWP_STEP_HORIZON)
            .T
        )
        _end = time.perf_counter()
        print(f"outer mask took {_end - _start}")

        # cast to pandas

        _start = time.perf_counter()

        nwp_inner = pd.DataFrame(
            data=nwp_inner,
            index=range(NWP_STEP_HORIZON),
            columns=[f"{var}_within" for var in ORDERED_NWP_FEATURE_VARIABLES],
        )

        nwp_outer = pd.DataFrame(
            data=nwp_outer,
            index=range(NWP_STEP_HORIZON),
            columns=[f"{var}_outer" for var in ORDERED_NWP_FEATURE_VARIABLES],
        )

        nwp_diff = pd.DataFrame(
            data=(nwp_inner.values - nwp_outer.values),
            columns=[col.replace("_within", "_diff") for col in nwp_inner],
            index=nwp_inner.index,
        )

        X = pd.concat([nwp_inner, nwp_outer, nwp_diff], axis=1)

        _end = time.perf_counter()
        print(f"pandas casting took: {_end - _start}")

        # process PV/GSP data
        gsp = pd.DataFrame(
            data["gsp"]["generation_mw"] / data["gsp"]["installedcapacity_mwp"],
            index=data["gsp"].coords["datetime_gmt"].values,
            columns=["target"],
        )

        pv_by_hour = gsp.resample("1H").max()

        pv_2hr = pd.concat(
            [pv_by_hour.shift(2).iloc[[-2, -1]], pv_by_hour.iloc[2:], pv_by_hour[:12]],
            axis=0,
        ).reset_index(drop=True)
        pv_1hr = pd.concat(
            [pv_by_hour.shift(1).iloc[[-1]], pv_by_hour.iloc[1:], pv_by_hour[:12]],
            axis=0,
        ).reset_index(drop=True)
        pv_1day = pv_1day = pd.concat(
            (pv_by_hour.iloc[:-1], pv_by_hour.iloc[:13]), axis=0
        ).reset_index(drop=True)

        X["PV_LAG_2HR"] = pv_2hr
        X["PV_LAG_1HR"] = pv_1hr
        X["PV_LAG_DAY"] = pv_1day

        _start = time.perf_counter()

        # get autoregressive beta on short term pv data
        pv_covariates = gsp.shift(freq=AUTO_REGRESSION_COVARIATE_LAG)
        beta = clipped_univariate_linear_regression(
            pv_covariates.values[-DEFAULT_ROLLING_LR_WINDOW_SIZE:].flatten(),
            gsp.values[-DEFAULT_ROLLING_LR_WINDOW_SIZE:].flatten(),
        )

        X["LR_Beta"] = beta

        _end = time.perf_counter()

        print(f"LR beta took {_end - _start}")

        assert X.shape == (37, NWP_VARIABLE_NUM * 3 + 4)

        cov = Covariates(
            covariates=X,
            metadata={
                "installedcapacity_mwp": data["gsp"]
                .isel(datetime_gmt=-1)["installedcapacity_mwp"]
                .values.item(),
            }
        )
        if time_of_inference_utc is not None:
            cov.metadata["time_of_inference_utc"] = time_of_inference_utc,
        return cov

    def predict_from_covariates(self, covariates: Covariates) -> Dict[int, Prediction]:
        X = covariates.covariates.loc[self._config.forecast_horizon_hours]
        predictions = {
            forecast_horizon_hour: self.meta_model[forecast_horizon_hour].predict(
                X.loc[forecast_horizon_hour]
            )
            for forecast_horizon_hour in self._config.forecast_horizon_hours
        }

        predictions = {
            hour: self.process_model_output(hour, model_output, covariates.metadata)
            for hour, model_output in predictions.items()
        }

        return predictions

    def process_model_output(
        self,
        forecast_horizon_hours: int,
        forecast: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Prediction:

        if "time_of_inference_utc" not in metadata:
            time_of_inference = np.datetime64("now")  # needs to be in UTC
        else:
            time_of_inference = metadata["time_of_inference_utc"]

        if "installedcapacity_mwp" not in metadata:
            capacity = DEFAULT_INSTALLED_CAPACITY_MWP
        else:
            capacity = metadata["installedcapacity_mwp"]

        return Prediction(
            time_of_inference,
            time_of_inference + np.timedelta64(forecast_horizon_hours, "h"),
            forecast[0] * capacity,
        )
