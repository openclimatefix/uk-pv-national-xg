from pathlib import Path
from typing import Dict

import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from gradboost_pv.inference.models import Prediction, NationalBoostInferenceModel
from gradboost_pv.utils.logger import getLogger
from gradboost_pv.utils.typing import Hour


def process_predictions_to_pandas(predictions: Dict[Hour, Prediction]) -> pd.DataFrame:
    return pd.concat(
        {
            hour: pd.DataFrame.from_dict(
                {
                    "datetime_of_model_inference_utc": pred.datetime_of_model_inference_utc,
                    "datetime_of_target_utc": pred.datetime_of_target_utc,
                    "forecast_kw": pred.forecast_kw,
                },
                orient="index",
            ).T
            for hour, pred in predictions.items()
        }
    ).set_index("datetime_of_model_inference_utc")


class MockDatabase:
    """Draft Database for prototyping, storing results in a pandas DataFrame."""

    def __init__(self, path_to_database: Path, overwrite_current: bool = True) -> None:
        self.path_to_database = path_to_database
        self.overwrite = overwrite_current

        self.data = None
        self.logger = getLogger("mock_database")

    def create_new_database(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "datetime_of_target_utc",
                "forecast_kw",
            ]
        )

    def load(self) -> None:
        if self.overwrite:
            self.data = self.create_new_database()
        else:
            try:
                self.data = pd.read_pickle(self.path_to_database)
            except FileNotFoundError:
                # could not find database, initialising a new one
                self.data = self.create_new_database()

    def write(self, predictions: Dict[Hour, Prediction]) -> None:
        assert self.data is not None

        self.data = pd.concat(
            [self.data, process_predictions_to_pandas(predictions)], axis=0
        )

    def disconnect(self):
        assert self.data is not None

        self.data.to_pickle(self.path_to_database)


class MockDatabaseConnection:
    """Class for running draft model inference, connecting to a local mock
    database.
    """

    def __init__(
        self, path_to_mock_database: Path, overwrite_database: bool = False
    ) -> None:
        self.path_to_database = path_to_mock_database
        self.overwrite_database = overwrite_database
        self.logger = getLogger("mock_database_connection")

        self.connected = False
        self.database = None

    def connect(self):
        self.database = MockDatabase(self.path_to_database, self.overwrite_database)
        self.database.load()
        self.connected = True
        self.logger.debug("Initialised connection to database")

    def disconnect(self):
        assert self.database is not None
        self.database.disconnect()
        self.database = None
        self.connected = False
        self.logger.debug("Closed connection to database")

    def __enter__(self) -> MockDatabase:
        if not self.connected:
            self.connect()

        assert self.database is not None
        return self.database

    def __exit__(self, type, value, traceback) -> None:
        self.disconnect()
        return


@functional_datapipe("nationalboost_model_inference")
class NationalBoostModelInference(IterDataPipe):
    def __init__(
        self,
        model: NationalBoostInferenceModel,
        data_feed: IterDataPipe,
        prediction_sink: MockDatabaseConnection,
    ) -> None:
        self.model = model
        self.data_feed = data_feed
        self.database_connection = prediction_sink

    def run(self):
        with self.database_connection as conn:
            for data in self.data_feed:
                prediction: Dict[Hour, Prediction] = self.model(data)
                conn.write(prediction)
