from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBRegressor
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation

from gradboost_pv.preprocessing.region_filtered import (
    get_eso_uk_multipolygon,
    generate_polygon_mask,
)
from gradboost_pv.models.utils import TRIG_DATETIME_FEATURE_NAMES
from gradboost_pv.utils.logger import getLogger
from gradboost_pv.models.region_filtered import (
    build_solar_pv_features,
)
from gradboost_pv.preprocessing.region_filtered import DEFAULT_VARIABLES_FOR_PROCESSING
from gradboost_pv.inference.data_feeds import DataInput


DEFAULT_PATH_TO_UK_REGION_MASK = Path(__file__).parents[2] / "data/uk_region_mask.npy"
DEFAULT_MODEL_COVARIATES = [
    "dswrf_within",
    "dswrf_outer",
    "hcct_within",
    "hcct_outer",
    "lcc_within",
    "lcc_outer",
    "t_within",
    "t_outer",
    "sde_within",
    "sde_outer",
    "wdir10_within",
    "wdir10_outer",
    "dswrf_diff",
    "hcct_diff",
    "lcc_diff",
    "t_diff",
    "sde_diff",
    "wdir10_diff",
    "SIN_MONTH",
    "COS_MONTH",
    "SIN_DAY",
    "COS_DAY",
    "SIN_HOUR",
    "COS_HOUR",
    "PV_LAG_DAY",
    "PV_LAG_1HR",
    "PV_LAG_2HR",
    "ghi",
    "dni",
    "zenith",
    "elevation",
    "azimuth",
    "equation_of_time",
]

# store models and results by hours ahead
Hour = int


@dataclass(frozen=True)
class Covariates:
    covariates: pd.DataFrame
    installed_capacity_mwp_at_inference_time: float
    inference_datetime_utc: np.datetime64


@dataclass(frozen=True)
class Prediction:
    datetime_of_model_inference_utc: np.datetime64
    datetime_of_target_utc: np.datetime64
    forecast_kw: float


def _load_default_nwp_variables() -> list[str]:
    return DEFAULT_VARIABLES_FOR_PROCESSING


def _load_default_forecast_horizons() -> list[Hour]:
    return list(range(37))


def _load_default_model_covariates() -> list[str]:
    return DEFAULT_MODEL_COVARIATES


@dataclass
class NationalPVModelConfig:
    # used for logging
    name: str

    # local path to mask
    path_to_uk_region_mask: Path = DEFAULT_PATH_TO_UK_REGION_MASK

    # period of observation for nwp and gsp at each evaluation
    gsp_data_history: np.timedelta64 = np.timedelta64(24, "h")
    gsp_data_frequency: str = "30T"

    # hours ahead to forecast
    forecast_horizon_hours: list[Hour] = field(
        default_factory=_load_default_forecast_horizons
    )
    # subset of NWP variables used by model
    nwp_variables: list[str] = field(default_factory=_load_default_nwp_variables)

    # required features for our model to run
    required_model_covariates: list[str] = field(
        default_factory=_load_default_model_covariates
    )

    # whether we mark the time of inference as the read time of database
    # or the machine's clock time
    # not overwriting allows for backfilling data with a mock data feed
    overwrite_read_datetime_at_inference: bool = True

    # clip near 0 forecast values to 0
    clip_near_zero_predictions: bool = True
    clip_near_zero_value_kw: float = 50.0


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
    def predict_from_covariates(self, covariates: Covariates) -> Dict[Hour, Prediction]:
        pass

    def __call__(self, data) -> Dict[Hour, Prediction]:
        return self.predict(data)


class NationalBoostInferenceModel(BaseInferenceModel):
    """Model Object for NationalPV Forecast."""

    def __init__(
        self,
        config: NationalPVModelConfig,
        model_loader: Callable[[Hour], XGBRegressor],
        nwp_x_coords: np.ndarray,
        nwp_y_coords: np.ndarray,
    ) -> None:
        self.model_loader = model_loader
        self.nwp_x_coords = nwp_x_coords
        self.nwp_y_coords = nwp_y_coords
        super().__init__(config)
        self.logger = getLogger(self._config.name)

    def initialise(self):
        self.logger.debug("Initialising model.")
        # load models for each time step from disk.
        self.meta_model = self.load_meta_model()

        # get uk-region mask/polygon
        self.mask = self.load_mask()

    def load_meta_model(self) -> Dict[Hour, XGBRegressor]:
        return {
            forecast_horizon_hour: self.load_model_per_forecast_horizon(
                forecast_horizon_hour
            )
            for forecast_horizon_hour in self._config.forecast_horizon_hours
        }

    def load_model_per_forecast_horizon(
        self, forecast_horizon_hour: Hour
    ) -> XGBRegressor:
        return self.model_loader(forecast_horizon_hour)

    def load_mask(self) -> xr.DataArray:
        try:
            mask = np.load(self._config.path_to_uk_region_mask)
            self.logger.debug("Loaded region mask from local.")

        except FileNotFoundError:
            self.logger.info("Downloading region mask.")
            # couldn't find locally, download instead
            uk_polygon = get_eso_uk_multipolygon()
            mask = generate_polygon_mask(
                self.nwp_x_coords, self.nwp_y_coords, uk_polygon
            )

        mask = xr.DataArray(
            np.tile(
                mask.T,
                (
                    len(self._config.nwp_variables),
                    len(self._config.forecast_horizon_hours),
                    1,
                    1,
                ),
            ),
            dims=["variable", "step", "x", "y"],
        )
        return mask

    def check_incoming_data(self, data: DataInput) -> None:
        """Runs some basic operations to check that we have received the
        data required for our model to run.

        This is a basic pass and not a definitive santitization of the data.
        Args:
            data (Dict[str, xr.Dataset]): nwp and gsp data from datafeed.
        """
        # GSP PV data is 30 min intervals for 24 hours (inclusive)
        assert (
            pd.infer_freq(data.gsp.coords["datetime_gmt"].values)
            == self._config.gsp_data_frequency
        )
        assert len(data.gsp.coords["datetime_gmt"].values) == (2 * 24 + 1)

        # check that the variables we would like are available
        assert set(self._config.nwp_variables).issubset(
            data.nwp.coords["variable"].values
        )

    def covariate_transform(self, data: DataInput) -> Covariates:
        """Transform raw nwp and gsp data to features for model inference.

        This function is analogous to gradboost_pv.models.region_filtered.build_dataset_from_slice
        except that it is operating on a single batch/observation rather than generating
        features for an entire dataset.
        In addition - for the training dataset, the region masking
        occurs offline in a prerequisite preprocessing step. For this live inference version,
        we perform the processing step live.

        TODO - create a method linking the two methods so there does not have to two places
        to update the same logic.

        Args:
            data (Dict[str, xr.Dataset]): Data from NWP and PV databases.
            time_of_inference_utc (Optional[np.datetime64], optional): Time of inference.

        Returns:
            Covariates: Object collating the features used by the model
        """
        # basic data checking
        self.check_incoming_data(data)

        _nwp = data.nwp.sel(variable=self._config.nwp_variables)
        _nwp = _nwp.isel(step=self._config.forecast_horizon_hours)

        nwp_inner = (
            xr.where(~self.mask.isnull(), _nwp, np.nan)
            .mean(dim=["x", "y"])
            .to_array()
            .values.reshape(len(_nwp.coords["variable"]), len(_nwp.coords["step"]))
            .T
        )

        nwp_outer = (
            xr.where(self.mask.isnull(), _nwp, np.nan)
            .mean(dim=["x", "y"])
            .to_array()
            .values.reshape(len(_nwp.coords["variable"]), len(_nwp.coords["step"]))
            .T
        )

        # cast to pandas
        nwp_inner = pd.DataFrame(
            data=nwp_inner,
            index=self._config.forecast_horizon_hours,
            columns=[f"{var}_within" for var in _nwp.coords["variable"].values],
        )

        nwp_outer = pd.DataFrame(
            data=nwp_outer,
            index=self._config.forecast_horizon_hours,
            columns=[f"{var}_outer" for var in _nwp.coords["variable"].values],
        )

        nwp_diff = pd.DataFrame(
            data=(nwp_inner.values - nwp_outer.values),
            columns=[col.replace("_within", "_diff") for col in nwp_inner],
            index=nwp_inner.index,
        )

        # process PV/GSP data
        gsp = pd.DataFrame(
            data.gsp["generation_mw"].values / data.gsp["installedcapacity_mwp"].values,
            index=data.gsp.coords["datetime_gmt"].values,
            columns=["target"],
        ).sort_index(ascending=False)

        forecast_times = pd.DatetimeIndex(
            [
                data.forecast_intitation_datetime_utc + np.timedelta64(step, "h")
                for step in self._config.forecast_horizon_hours
            ]
        )

        _X = trigonometric_datetime_transformation(forecast_times)
        _X = pd.DataFrame(
            data=_X,
            index=self._config.forecast_horizon_hours,
            columns=TRIG_DATETIME_FEATURE_NAMES,
        )

        X = pd.concat([nwp_inner, nwp_outer, nwp_diff, _X], axis=1)

        solar_variables = build_solar_pv_features(forecast_times)
        solar_variables.index = self._config.forecast_horizon_hours

        pv_autoregressive_lags = list()
        for step in self._config.forecast_horizon_hours:
            lagged_data = pd.concat(
                [
                    gsp.shift(freq=np.timedelta64(24 - (step % 24), "h"))
                    .loc[gsp.index[0]]
                    .rename("PV_LAG_DAY"),
                    gsp.shift(freq=np.timedelta64(24 - ((step - 2) % 24), "h"))
                    .loc[gsp.index[0]]
                    .rename("PV_LAG_2HR"),
                    gsp.shift(freq=np.timedelta64(24 - ((step - 1) % 24), "h"))
                    .loc[gsp.index[0]]
                    .rename("PV_LAG_1HR"),
                ],
                axis=1,
            )
            lagged_data.index = [step]
            pv_autoregressive_lags.append(lagged_data)

        pv_autoregressive_lags = pd.concat(pv_autoregressive_lags)

        X = pd.concat([X, pv_autoregressive_lags, solar_variables], axis=1)

        assert X.shape == (
            len(self._config.forecast_horizon_hours),
            len(self._config.required_model_covariates),
        )
        assert sorted(X.columns.tolist()) == sorted(
            self._config.required_model_covariates
        )

        inference_time = (
            data.forecast_intitation_datetime_utc
            if not self._config.overwrite_read_datetime_at_inference
            else np.datetime64("now")
        )

        # reorder covariates, XGBoost requires inference/training design matrix
        # to have the same order (it doesn't store column names internally)
        # see https://github.com/dmlc/xgboost/issues/636
        # 1 entire day lost on this "feature" :-)
        cov = Covariates(
            covariates=X[self._config.required_model_covariates],
            installed_capacity_mwp_at_inference_time=data.gsp.sel(
                datetime_gmt=data.gsp.datetime_gmt.max().values
            )["installedcapacity_mwp"].values.item(),
            inference_datetime_utc=inference_time,
        )
        return cov

    def predict_from_covariates(self, covariates: Covariates) -> Dict[Hour, Prediction]:
        X = covariates.covariates.loc[self._config.forecast_horizon_hours]
        predictions = {
            forecast_horizon_hour: self.meta_model[forecast_horizon_hour].predict(
                X.loc[[forecast_horizon_hour]]
            )[0]
            for forecast_horizon_hour in self._config.forecast_horizon_hours
        }

        predictions = {
            hour: self.process_model_output(
                hour,
                model_output,
                covariates.installed_capacity_mwp_at_inference_time,
                covariates.inference_datetime_utc,
            )
            for hour, model_output in predictions.items()
        }

        return predictions

    def process_model_output(
        self,
        forecast_horizon_hours: Hour,
        forecast: float,
        pv_capacity_mwp: float,
        inference_datetime: np.datetime64,
    ) -> Prediction:

        pv_amount = forecast * pv_capacity_mwp

        if self._config.clip_near_zero_predictions:
            pv_amount = (
                pv_amount if pv_amount > self._config.clip_near_zero_value_kw else 0.0
            )

        return Prediction(
            inference_datetime,
            inference_datetime + np.timedelta64(forecast_horizon_hours, "h"),
            pv_amount,
        )
