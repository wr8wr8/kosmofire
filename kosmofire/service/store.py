from dataclasses import dataclass
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from ..config import PROJECT_ROOT

DEFAULT_GPKG = PROJECT_ROOT / "data" / "service" / "fire_service.gpkg"


@dataclass
class Query:
    geometry: BaseGeometry | None
    date_from: date | None
    date_to: date | None


class ServiceStore:
    def __init__(self, gpkg_path: Path = DEFAULT_GPKG):
        self.path = Path(gpkg_path)
        self.hotspots = gpd.read_file(self.path, layer="hotspots", engine="pyogrio")
        self.burns = gpd.read_file(self.path, layer="burns", engine="pyogrio")
        self.hotspots["detected_date"] = pd.to_datetime(self.hotspots["detected_at"], utc=True).dt.date
        self.burns["period_date"] = pd.to_datetime(self.burns["date_post"]).dt.date

    def select_hotspots(self, query: Query) -> gpd.GeoDataFrame:
        return self._filter(self.hotspots, query, "detected_date")

    def select_burns(self, query: Query) -> gpd.GeoDataFrame:
        selected = self._filter(self.burns, query, "period_date")
        if query.geometry is None or selected.empty:
            return selected
        clipped = selected.copy()
        clipped["geometry"] = selected.geometry.intersection(query.geometry)
        clipped = clipped[~clipped.geometry.is_empty]
        clipped["area_ha"] = [_area_ha(g, int(e)) for g, e in zip(clipped.geometry, clipped["epsg_utm"])]
        return clipped

    @staticmethod
    def _filter(frame: gpd.GeoDataFrame, query: Query, date_column: str) -> gpd.GeoDataFrame:
        selected = frame
        if query.geometry is not None:
            selected = selected[selected.intersects(query.geometry)]
        if query.date_from is not None:
            selected = selected[selected[date_column] >= query.date_from]
        if query.date_to is not None:
            selected = selected[selected[date_column] <= query.date_to]
        return selected


def _area_ha(geometry: BaseGeometry, epsg: int) -> float:
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return float(shapely_transform(to_utm.transform, geometry).area / 10_000.0)


def geometry_from_bbox(bbox: str) -> BaseGeometry:
    west, south, east, north = (float(v) for v in bbox.split(","))
    if not (west < east and south < north):
        raise ValueError("bbox must be west,south,east,north with west<east and south<north")
    return box(west, south, east, north)


def geometry_from_geojson(payload: dict) -> BaseGeometry:
    if payload.get("type") == "FeatureCollection":
        payload = payload["features"][0]
    if payload.get("type") == "Feature":
        payload = payload["geometry"]
    geometry = shape(payload)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("polygon is empty or invalid")
    return geometry
