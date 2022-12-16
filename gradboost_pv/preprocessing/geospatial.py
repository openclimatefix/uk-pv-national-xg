from typing import Union, Iterable
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union
import requests
import geopandas as gpd
from ocf_datapipes.utils.geospatial import osgb_to_lat_lon
import numpy as np
import multiprocessing as mp

ESO_GEO_JSON_URL = (
    "https://data.nationalgrideso.com/backend/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/"
    "resource/08534dae-5408-4e31-8639-b579c8f1c50b/download/gsp_regions_20220314.geojson"
)


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
