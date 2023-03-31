""" Function to save results to datbase """
import logging
from datetime import timedelta

import pandas as pd
from nowcasting_datamodel.models.convert import convert_df_to_national_forecast
from nowcasting_datamodel.save.update import update_all_forecast_latest
from sqlalchemy.orm import Session

import gradboost_pv
from gradboost_pv.inference.utils import filter_forecasts_on_sun_elevation

logger = logging.getLogger(__name__)


def save_to_database(results_df: pd.DataFrame, start_hour_to_save: int, session: Session):
    """
    Method to save results to a database
    """

    # TODO fix, wrong units somewhere
    results_df["forecast_mw"] = results_df["forecast_kw"].astype(float)
    results_df["target_datetime_utc"] = pd.to_datetime(results_df["datetime_of_target_utc"])

    # interpolate to 30 minutes
    results_df.set_index("datetime_of_target_utc", drop=True, inplace=True)
    results_df = pd.DataFrame(
        results_df["forecast_mw"].resample("30T").interpolate(method="linear")
    )
    results_df["target_datetime_utc"] = results_df.index

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
    session.add_all(forecast_sql.forecast_values)
    session.commit()
    session.flush()

    # only save 8 hour out, so we dont override PVnet
    target_time_filter = forecast_sql.forecast_creation_time + timedelta(hours=start_hour_to_save)
    forecast_sql.forecast_values = [
        f for f in forecast_sql.forecast_values if f.target_time >= target_time_filter
    ]
    logger.debug(
        f"Adding forecasts to latest, if target time is past {target_time_filter}. "
        f"This will be {len(forecast_sql.forecast_values)} forecast values"
    )
    update_all_forecast_latest(
        session=session, forecasts=[forecast_sql], update_national=True, update_gsp=False
    )
