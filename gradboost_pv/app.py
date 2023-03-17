"""Script to simulate data read, model inference and prediction write"""
import logging
import os
import pathlib
from pathlib import Path
from typing import Optional
from datetime import timedelta

import click
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.convert import convert_df_to_national_forecast
from nowcasting_datamodel.save.update import (
    update_all_forecast_latest,
)
from xgboost import XGBRegressor

import gradboost_pv
from gradboost_pv.inference.data_feeds import ProductionDataFeed
from gradboost_pv.inference.models import Hour, NationalBoostInferenceModel, NationalPVModelConfig
from gradboost_pv.inference.run import MockDatabaseConnection, NationalBoostModelInference
from gradboost_pv.inference.utils import filter_forecasts_on_sun_elevation
from gradboost_pv.models.s3 import build_object_name, create_s3_client, load_model
from gradboost_pv.models.utils import load_nwp_coordinates

DEFAULT_PATH_TO_MOCK_DATABASE = (
    Path(gradboost_pv.__file__).parents[1] / "data" / "mock_inference_database.pickle"
)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
    format="[%(asctime)s] %(module)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

logging.getLogger("gradboost_pv").setLevel(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
)

logging.getLogger("__main__").setLevel(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
)

logging.getLogger("xgnational_model").setLevel(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
)

logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)
logging.getLogger("s3transfer").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("fsspec").setLevel(logging.INFO)
logging.getLogger("s3fs").setLevel(logging.INFO)


@click.command()
@click.option(
    "--path_to_model_config",
    type=click.Path(path_type=pathlib.Path),
    help="Path to NationalBoost model config yaml.",
    default="./configs/default_production_model.yaml",
)
@click.option(
    "--path_to_datafeed_config",
    type=click.Path(path_type=pathlib.Path),
    help="Path to Production Datafeed config yaml.",
    default="./configs/default_production_datafeed.yaml",
)
@click.option(
    "--write_to_database",
    is_flag=True,
    default=True,
    envvar="WRITE_TO_DATABASE",
    help="Set this flag to actually write the results to the database."
    "By default we only print to stdout using mock local database.",
)
@click.option(
    "--s3_access_key",
    type=str,
    default=None,
    help="Optional AWS s3 Access Key.",
)
@click.option(
    "--s3_secret_key",
    type=str,
    default=None,
    help="Optional AWS s3 Secret Key.",
)
@click.option(
    "--start_hour_to_save",
    type=int,
    default=8,
    help="From X hours out, we save the values in the forecast latest table",
)
def main(
    path_to_model_config: Path,
    path_to_datafeed_config: Path,
    write_to_database: bool = True,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
    start_hour_to_save: Optional[int] = 8,
):
    """Entry point for inference script"""

    logger.debug(f"Starting main app {gradboost_pv.__version__=}")

    if s3_access_key is None or s3_secret_key is None:
        logger.debug("Creating s3 client with default env.var keys.")
        client = create_s3_client()
    else:
        logger.debug(f"Creating s3 client with specified keys: {s3_access_key}/{s3_secret_key}.")
        client = create_s3_client(s3_access_key, s3_secret_key)

    def model_loader_by_hour(hour: Hour) -> XGBRegressor:
        """Get a model by forecast hour using client"""
        return load_model(client, build_object_name(hour))

    # load in our national pv model
    logger.debug("Intitialised model")
    x, y = load_nwp_coordinates()
    model_config = NationalPVModelConfig.load_from_yaml(path_to_model_config)
    model = NationalBoostInferenceModel(model_config, model_loader_by_hour, x, y)
    model.initialise()
    logger.debug("Intitialised model:done")

    logger.debug("Defined production feed.")
    data_feed = ProductionDataFeed(path_to_datafeed_config)
    logger.debug("Defined production feed:done")

    # create a mock database to write to
    logger.debug("Not writing to database, storing in local mock database")
    database_conn = MockDatabaseConnection(DEFAULT_PATH_TO_MOCK_DATABASE, overwrite_database=True)

    inference_pipeline = NationalBoostModelInference(model, data_feed, database_conn)
    inference_pipeline.run()
    logger.debug("Model inference complete")

    # get dataframe object
    logger.debug("Saving results to DB")
    database_conn = MockDatabaseConnection(DEFAULT_PATH_TO_MOCK_DATABASE, overwrite_database=False)
    database_conn.connect()
    results_df = database_conn.database.data

    if not write_to_database:
        print(results_df)
    else:
        connection = DatabaseConnection(url=os.getenv("DB_URL"))

        with connection.get_session() as session:
            # TODO fix, wrong units somewhere
            results_df["forecast_mw"] = results_df["forecast_kw"]
            results_df["target_datetime_utc"] = results_df["datetime_of_target_utc"]

            logger.debug(results_df[["forecast_mw", "target_datetime_utc"]])

            forecast_sql = convert_df_to_national_forecast(
                forecast_values_df=results_df,
                session=session,
                model_name="National_xg",
                version=gradboost_pv.__version__,
            )

            # zero out night times
            forecasts = filter_forecasts_on_sun_elevation(forecasts=[forecast_sql])
            forecast_sql = forecasts[0]

            # add to database
            logger.debug("Adding forecast to database")
            session.add(forecast_sql)
            session.commit()

            # only save 8 hour out, so we dont override PVnet
            target_time_filter = forecast_sql.forecast_creation_time + timedelta(
                hours=start_hour_to_save
            )
            forecast_sql.forecast_values = [
                f for f in forecast_sql.forecast_values if f.target_time >= target_time_filter
            ]
            logger.debug(f"Adding forecasts to latest, if target time is past {target_time_filter}. "
                         f"This will be {len(forecast_sql.forecast_values)} forecast values")
            update_all_forecast_latest(
                session=session, forecasts=forecasts, update_national=True, update_gsp=False
            )

    logger.info("Done")


if __name__ == "__main__":
    main()
