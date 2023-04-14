""" Function to save results to datbase """
import logging

import pandas as pd
from nowcasting_datamodel.models import ForecastSQL
from nowcasting_datamodel.models.convert import convert_df_to_national_forecast
from nowcasting_datamodel.save.save import save_all_forecast_values_seven_days
from nowcasting_datamodel.save.update import update_all_forecast_latest
from sqlalchemy.orm import Session

import gradboost_pv
from gradboost_pv.inference.utils import filter_forecasts_on_sun_elevation

logger = logging.getLogger(__name__)


def save_to_database(results_df: pd.DataFrame, session: Session):
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
    logger.debug(f"Adding {len(forecast_sql.forecast_values)} forecast values to database")
    session.add_all(forecast_sql.forecast_values)
    session.commit()
    session.flush()

    # copy the forecast so that we can update the latest table
    forecast_sql = ForecastSQL(
        historic=forecast_sql.model,
        forecast_values=forecast_sql.forecast_values,
        forecast_creation_time=forecast_sql.forecast_creation_time,
        model=forecast_sql.model,
        location=forecast_sql.location,
        input_data_last_updated=forecast_sql.input_data_last_updated,
    )

    # only save 8 hour out, so we dont override PVnet
    logger.debug(
        f"Adding all forecasts to latest. "
        f"This will be {len(forecast_sql.forecast_values)} forecast values"
    )
    update_all_forecast_latest(
        session=session,
        forecasts=[forecast_sql],
        update_national=True,
        update_gsp=False,
        model_name="National_xg",
    )

    logger.debug("Saving to last seven days table")
    save_all_forecast_values_seven_days(session=session, forecasts=forecasts)
    session.commit()
