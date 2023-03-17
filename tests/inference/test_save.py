import pandas as pd
from datetime import datetime, timezone

from freezegun import freeze_time

from gradboost_pv.save import save_to_database

from nowcasting_datamodel.models import ForecastSQL, ForecastValueSQL, ForecastValueLatestSQL


@freeze_time("2023-01-01")
def test_save_to_database(db_session):
    results_df = pd.DataFrame(
        columns=[[ "datetime_of_target_utc", "forecast_kw"]],
        data=[
            [datetime(2023, 1, 1), 7.3],
            [datetime(2023, 1, 1, 1), 8.3],
            [datetime(2023, 1, 1, 2), 9.3],
        ],
    )

    save_to_database(session=db_session, results_df=results_df, start_hour_to_save=0)

    assert len(db_session.query(ForecastSQL).all()) == 2  # 1 normal, 1 historic
    assert len(db_session.query(ForecastValueSQL).all()) == 3
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 3


@freeze_time("2023-01-01")
def test_save_to_database_reduce(db_session):
    db_session.query(ForecastValueSQL).delete()
    db_session.query(ForecastValueLatestSQL).delete()
    db_session.query(ForecastSQL).delete()

    results_df = pd.DataFrame(
        columns=[[ "datetime_of_target_utc", "forecast_kw"]],
        data=[
            [datetime(2023, 1, 1), 7.3],
            [datetime(2023, 1, 1, 1), 8.3],
            [datetime(2023, 1, 1, 2), 9.3],
        ],
    )

    save_to_database(session=db_session, results_df=results_df, start_hour_to_save=2)

    assert len(db_session.query(ForecastSQL).all()) == 2  # 1 normal, 1 historic
    assert len(db_session.query(ForecastValueSQL).all()) == 3
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 1
