import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from gradboost_pv.preprocessing.region_filtered import (
    _process_nwp,
    generate_polygon_mask,
    process_eso_uk_multipolygon,
    query_eso_geojson,
)


@pytest.fixture
def mock_national_eso_polygon():
    geom_1 = MultiPolygon(
        [
            Polygon([[0, 0], [0, 100_000], [100_000, 100_000], [100_000, 0]]),
            Polygon(
                [[70_000, 70_000], [70_000, 100_000], [170_000, 120_000], [100_000, 0]]
            ),
        ]
    )
    geom_2 = MultiPolygon(
        [
            Polygon(
                [
                    [-50_000, -110_000],
                    [-30_000, -80_000],
                    [-10_000, -30_000],
                    [-40_000, -70_000],
                ]
            ),
            Polygon(
                [
                    [-60_000, -60_000],
                    [-60_000, -20_000],
                    [-20_000, -20_000],
                    [-20_000, -60_000],
                ]
            ),
        ]
    )

    fake_metadata = [["SITE_A", "_A"], ["SITE_B", "_B"]]

    multi_poly = gpd.GeoDataFrame(
        geometry=[geom_1, geom_2], columns=["GSPs", "GSPGroup"], data=fake_metadata
    )
    return multi_poly


@pytest.fixture
def mock_processed_polygon(mock_national_eso_polygon):
    return process_eso_uk_multipolygon(mock_national_eso_polygon)


@pytest.fixture
def mock_geospatial_mask(mock_processed_polygon):
    x_coords = np.asarray([-80_000, -30_000, 30_000, 120_000])
    y_coords = np.asarray([-80_000, -30_000, 30_000, 120_000])

    mask = generate_polygon_mask(x_coords, y_coords, mock_processed_polygon)
    mask = xr.DataArray(mask.T, dims=["x", "y"])

    return x_coords, y_coords, mask


def test_national_grid_eso_geojson_request():
    resp = query_eso_geojson()

    assert ["GSPs", "GSPGroup", "geometry"] == resp.columns.tolist()
    assert len(resp) == 333


def test_national_grid_result_processing(mock_processed_polygon: MultiPolygon):
    expected_result = MultiPolygon(
        (
            Polygon(
                (
                    (-50_000, -110_000),
                    (-40_000, -70_000),
                    (-32_500, -60_000),
                    (-60_000, -60_000),
                    (-60_000, -20_000),
                    (-20_000, -20_000),
                    (-20_000, -43333.333333333333336),
                    (-10_000, -30_000),
                    (-20_000, -55_000),
                    (-20_000, -60_000),
                    (-22_000, -60_000),
                    (-30_000, -80_000),
                    (-50_000, -110_000),
                )
            ),
            Polygon(
                (
                    (70_000, 100_000),
                    (170_000, 120_000),
                    (100_000, 0),
                    (0, 0),
                    (0, 100_000),
                    (70_000, 100_000),
                )
            ),
        )
    )

    assert expected_result == mock_processed_polygon


def test_mask_generation(mock_geospatial_mask):
    _, _, mask = mock_geospatial_mask

    expected_mask = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1.0, np.nan, np.nan],
            [np.nan, np.nan, 1.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    ).T

    assert np.allclose(expected_mask, mask.values, 1e-6, equal_nan=True)


def test_region_filtered_processing(mock_geospatial_mask, nwp_single_observation):
    x, y, mask = mock_geospatial_mask
    nwp_slice = nwp_single_observation.sel(x=x, y=y, method="nearest")

    within, outer = _process_nwp(nwp_slice, mask)
    within, outer = within.to_array().values, outer.to_array().values

    within_expected = np.array([283.435], dtype=np.float32)
    outer_expected = np.array([283.5064], dtype=np.float32)

    assert np.isclose(within_expected, within)
    assert np.isclose(outer_expected, outer)
