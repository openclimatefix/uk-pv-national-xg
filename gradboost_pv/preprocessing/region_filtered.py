from typing import Tuple, Union, Iterable
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union
import requests
import geopandas as gpd
from ocf_datapipes.utils.geospatial import osgb_to_lat_lon
import numpy as np
import multiprocessing as mp
import xarray as xr
import itertools

ESO_GEO_JSON_URL = (
    "https://data.nationalgrideso.com/backend/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/"
    "resource/08534dae-5408-4e31-8639-b579c8f1c50b/download/gsp_regions_20220314.geojson"
)

# processing takes quite a long time, so take a subset for now.
DEFAULT_VARIABLES_FOR_PROCESSING = [
    "dswrf",
    "hcct",
    "lcc",
    "t",
    "sde",
    "wdir10",
]


def get_eso_uk_multipolygon() -> MultiPolygon:
    with requests.get(ESO_GEO_JSON_URL) as response:
        shape_gpd = gpd.read_file(response.text)

        # calculate the centroid before using - to_crs
        shape_gpd["centroid_x"] = shape_gpd["geometry"].centroid.x
        shape_gpd["centroid_y"] = shape_gpd["geometry"].centroid.y
        shape_gpd["centroid_lat"], shape_gpd["centroid_lon"] = osgb_to_lat_lon(
            x=shape_gpd["centroid_x"], y=shape_gpd["centroid_y"]
        )

        shape_gpd.sort_values("GSPs", inplace=True)
        shape_gpd.reset_index(inplace=True, drop=True)
        shape_gpd["RegionID"] = range(1, len(shape_gpd) + 1)

        return MultiPolygon(
            Polygon(p.exterior) for p in unary_union(shape_gpd["geometry"].values)
        )


def generate_polygon_mask(
    coordinates_x: Iterable[int], coordinates_y: Iterable[int], polygon: MultiPolygon
) -> np.ndarray:
    coords = list(
        map(
            lambda x: Point(x[0], x[1]), itertools.product(coordinates_x, coordinates_y)
        )
    )

    # create a mask for belonging to UK region or not
    mask = check_points_in_multipolygon_multiprocessed(coords, polygon)
    mask = mask.reshape(len(coordinates_x), len(coordinates_y)).T
    mask = mask.astype(float)
    mask[mask == 0] = np.nan

    return mask


def check_point_in_multipolygon(
    point: Point, polygon: Union[MultiPolygon, Polygon]
) -> bool:
    return polygon.contains(point)


def check_points_in_multipolygon_multiprocessed(
    points: Iterable[Point],
    polygon: Union[MultiPolygon, Polygon],
    num_processes: int = 10,
) -> np.ndarray:
    items = [(point, polygon) for point in points]
    results = list()
    with mp.Pool(num_processes) as pool:
        for result in pool.starmap(check_point_in_multipolygon, items):
            results.append(result)
    return np.asarray(results)


class NWPUKRegionMaskedDatasetBuilder:
    def __init__(
        self, nwp: xr.Dataset, evaluation_timepoints: Iterable[np.datetime64]
    ) -> None:
        self.nwp = nwp
        self.eval_timepoints = evaluation_timepoints
        self.mask = self.load_mask()

    def load_mask(self) -> xr.DataArray:
        uk_polygon = get_eso_uk_multipolygon()
        mask = generate_polygon_mask(
            self.nwp.coords["x"], self.nwp.coords["y"], uk_polygon
        )

        # convert numpy array to xarray mask for a 1 variable, 1 step times series of (x,y) coords
        mask = xr.DataArray(
            np.tile(mask.T, (len(self.nwp.coords["init_time"]), 1, 1)),
            dims=["init_time", "x", "y"],
        )
        return mask

    def build_region_masked_covariates(
        self,
        variable: str,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        uk_region = xr.where(
            ~self.mask.isnull(), self.nwp.isel(step=step).sel(variable=variable), np.nan
        ).mean(dim=["x", "y"])
        outer_region = xr.where(
            self.mask.isnull(), self.nwp.isel(step=step).sel(variable=variable), np.nan
        ).mean(dim=["x", "y"])

        # interpolate both to the common GSP time points
        uk_region = (
            uk_region.interp(init_time=self.eval_timepoints, method="linear")
            .to_array()
            .as_numpy()
        )
        outer_region = (
            outer_region.interp(init_time=self.eval_timepoints, method="linear")
            .to_array()
            .as_numpy()
        )

        return (uk_region, outer_region)
