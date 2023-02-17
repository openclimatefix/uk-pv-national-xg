import numpy as np
import pandas as pd

from gradboost_pv.models.utils import (
    DEFAULT_UK_SOUTH_LOCATION,
    build_lagged_features,
    build_solar_pv_features,
)


def test_build_solar_pv_features():
    location = DEFAULT_UK_SOUTH_LOCATION
    datetimes = pd.DatetimeIndex(
        pd.date_range(
            np.datetime64("2020-02-02T00:00:00"),
            np.datetime64("2020-02-02T12:00:00"),
            freq="12H",
        )
    )

    expected_result = pd.DataFrame(
        data=np.array(
            [
                [0.0, 0.0, 144.05407043, -54.05407043, 352.67446952, -13.56942608],
                [
                    294.054255,
                    647.54572524,
                    69.79638589,
                    20.20361411,
                    175.40531574,
                    -13.63370714,
                ],
            ]
        ),
        columns=["ghi", "dni", "zenith", "elevation", "azimuth", "equation_of_time"],
        index=datetimes,
    )

    output = build_solar_pv_features(datetimes, location)

    pd.testing.assert_frame_equal(output, expected_result)


def test_build_lagged_features(mock_gsp_data: pd.DataFrame):

    forecast_horizon = np.timedelta64(0, "h")

    expected_result = pd.DataFrame(
        index=pd.DatetimeIndex([np.datetime64("2020-02-03T00:00:00")]),
        data=np.asarray([0.0, 46.0, 44.0]).reshape(1, -1),
        columns=["PV_LAG_DAY", "PV_LAG_1HR", "PV_LAG_2HR"],
    )
    expected_result.index.freq = "-30T"

    pd.testing.assert_frame_equal(
        expected_result, build_lagged_features(mock_gsp_data, forecast_horizon)
    )
