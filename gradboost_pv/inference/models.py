"""Models used for inference"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation
from xgboost import XGBRegressor

from gradboost_pv.inference.data_feeds import DataInput
from gradboost_pv.models.utils import (
    TRIG_DATETIME_FEATURE_NAMES,
    build_lagged_features,
    build_solar_pv_features,
)
from gradboost_pv.preprocessing.region_filtered import (
    DEFAULT_VARIABLES_FOR_PROCESSING,
    _process_nwp,
    generate_polygon_mask,
    process_eso_uk_multipolygon,
    query_eso_geojson,
)
from gradboost_pv.utils.logger import getLogger

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
    """Dataclass for storing processed data ready for model usage"""

    covariates: pd.DataFrame
    installed_capacity_mwp_at_inference_time: float
    inference_datetime_utc: np.datetime64


@dataclass(frozen=True)
class Prediction:
    """Wrapper for model output"""

    datetime_of_model_inference_utc: np.datetime64
    datetime_of_target_utc: np.datetime64
    forecast_kw: float


def _load_default_nwp_variables() -> list[str]:
    """Default factory for NWP variables used"""
    return DEFAULT_VARIABLES_FOR_PROCESSING


def _load_default_forecast_horizons() -> list[Hour]:
    """Default factory for hours forecasted"""
    return list(range(37))


def _load_default_model_covariates() -> list[str]:
    """Factory for expected model covariates"""
    return DEFAULT_MODEL_COVARIATES


@dataclass
class NationalPVModelConfig:
    """Config class used to supply the inference model with setup information"""

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

    # clip near zero forecast values to zero
    clip_near_zero_predictions: bool = True
    clip_near_zero_value_kw: float = 50.0


class BaseInferenceModel(ABC):
    """Abstract Inference Model"""

    def __init__(self, config: NationalPVModelConfig) -> None:
        """Abstract model for trained NationalUK PV Prediciton"""
        self._config = config

    @property
    def get_config(self):
        """Return model configuration"""
        return self._config

    @abstractmethod
    def initialise(self):
        """Perform any model initialisation steps needed"""
        pass

    @abstractmethod
    def covariate_transform(self, data) -> Covariates:
        """Transform raw data from database into model features"""
        pass

    def predict(self, data):
        """Call model prediction from raw database data."""
        X = self.covariate_transform(data)
        return self.predict_from_covariates(X)

    @abstractmethod
    def predict_from_covariates(self, covariates: Covariates) -> Dict[Hour, Prediction]:
        """Call model prediction from generated features."""
        pass

    def __call__(self, data) -> Dict[Hour, Prediction]:
        """Call model inference/prediction"""
        return self.predict(data)


class NationalBoostInferenceModel(BaseInferenceModel):
    """NationalBoost model based on uk-region masked NWP processing"""

    def __init__(
        self,
        config: NationalPVModelConfig,
        model_loader: Callable[[Hour], XGBRegressor],
        nwp_x_coords: np.ndarray,
        nwp_y_coords: np.ndarray,
    ) -> None:
        """Model Object for NationalPV Inference"""

        self.model_loader = model_loader
        self.nwp_x_coords = nwp_x_coords
        self.nwp_y_coords = nwp_y_coords
        super().__init__(config)
        self.logger = getLogger(self._config.name)

    def initialise(self):
        """Load model and region mask"""
        self.logger.debug("Initialising model.")
        # load models for each time step from disk.
        self.meta_model = self.load_meta_model()

        # get uk-region mask/polygon
        self.mask = self.load_mask()

    def load_meta_model(self) -> Dict[Hour, XGBRegressor]:
        """Loads model for each forecast horizon

        Returns:
            Dict[Hour, XGBRegressor]: dict of models indexed by forecast horizon
        """
        return {
            forecast_horizon_hour: self.load_model_per_forecast_horizon(
                forecast_horizon_hour
            )
            for forecast_horizon_hour in self._config.forecast_horizon_hours
        }

    def load_model_per_forecast_horizon(
        self, forecast_horizon_hour: Hour
    ) -> XGBRegressor:
        """Wrapper for model loading"""
        return self.model_loader(forecast_horizon_hour)

    def load_mask(self) -> xr.DataArray:
        """Loads UK-region mask into memory.

        First attempts to load from disk, otherwise downloads form
        national grid.

        Returns:
            xr.DataArray: Mask of UK region on NWP coordinate grid.
        """
        try:
            mask = np.load(self._config.path_to_uk_region_mask)
            self.logger.debug("Loaded region mask from local.")

        except FileNotFoundError:
            self.logger.info("Downloading region mask.")
            # couldn't find locally, download instead
            uk_polygon = query_eso_geojson()
            uk_polygon = process_eso_uk_multipolygon(uk_polygon)
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
        """Basic data sanitation

        Runs some basic operations to check that we have received the
        data required for our model to run.This is a basic pass and
        not a definitive santitization of the data.

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

        # use region-masked nwp processing
        nwp_inner, nwp_outer = _process_nwp(_nwp, self.mask)

        # cast to pandas
        nwp_inner = pd.DataFrame(
            data=nwp_inner.to_array()
            .values.reshape(len(_nwp.coords["variable"]), len(_nwp.coords["step"]))
            .T,
            index=self._config.forecast_horizon_hours,
            columns=[f"{var}_within" for var in _nwp.coords["variable"].values],
        )

        nwp_outer = pd.DataFrame(
            data=nwp_outer.to_array()
            .values.reshape(len(_nwp.coords["variable"]), len(_nwp.coords["step"]))
            .T,
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

        # build lagged features for each forecast horizon
        for step in self._config.forecast_horizon_hours:
            lags = build_lagged_features(gsp, np.timedelta64(step, "h")).loc[
                [gsp.index.max()]
            ]
            lags.index = [step]
            pv_autoregressive_lags.append(lags)

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
        cov = Covariates(
            covariates=X[self._config.required_model_covariates],
            installed_capacity_mwp_at_inference_time=data.gsp.sel(
                datetime_gmt=data.gsp.datetime_gmt.max().values
            )["installedcapacity_mwp"].values.item(),
            inference_datetime_utc=inference_time,
        )
        return cov

    def predict_from_covariates(self, covariates: Covariates) -> Dict[Hour, Prediction]:
        """Run model on generated features.

        Args:
            covariates (Covariates): Features generates from various datasources

        Returns:
            Dict[Hour, Prediction]: Predictions for each forecast horizon
        """
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
        """Sanitize model output into Prediction object"""

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