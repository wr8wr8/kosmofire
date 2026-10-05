import io
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point

from kosmofire.geo_export import af_points, burn_polygons
from kosmofire.service.api import create_app


def _bs_row():
    return pd.Series(
        {
            "chip_id": "BS_tr_000001",
            "epsg": 32637,
            "x_min": 400000.0,
            "y_max": 5300000.0,
            "gsd": 20,
            "date_pre": "2024-06-01",
            "date_post": "2024-06-15",
        }
    )


def _af_row():
    return pd.Series(
        {
            "chip_id": "AF_tr_000001",
            "epsg": 32637,
            "x_min": 400000.0,
            "y_max": 5300000.0,
            "gsd": 375,
            "acq_datetime": "2024-06-10T10:00:00+00:00",
            "satellite": "SNPP",
        }
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("svc") / "svc.gpkg"
    severity = np.zeros((512, 512), dtype=np.uint8)
    severity[100:150, 100:150] = 1
    severity[200:260, 200:250] = 2
    severity[300:330, 300:340] = 3
    polygons = burn_polygons(_bs_row(), severity)
    burns = gpd.GeoDataFrame(polygons, geometry="geometry", crs="EPSG:4326")
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[10, 10] = 1
    mask[20, 30] = 1
    points = af_points(_af_row(), mask)
    hotspots = gpd.GeoDataFrame(points, geometry=[Point(p["lon"], p["lat"]) for p in points], crs="EPSG:4326").drop(columns=["lon", "lat"])
    hotspots.to_file(path, layer="hotspots", driver="GPKG")
    burns.to_file(path, layer="burns", driver="GPKG")
    return TestClient(create_app(path))


def test_health(client):
    assert client.get("/api/v1/health").json()["burn_polygons"] == 3


def test_report_area_matches_pixel_counts(client):
    data = client.get("/api/v1/report").json()
    by = {row["severity_class"]: row["area_ha"] for row in data["by_severity"]}
    assert by["low"] == pytest.approx(50 * 50 * 0.04, rel=1e-6)
    assert by["moderate"] == pytest.approx(60 * 50 * 0.04, rel=1e-6)
    assert by["high"] == pytest.approx(30 * 40 * 0.04, rel=1e-6)
    assert data["total_burned_area_ha"] == pytest.approx(sum(by.values()), abs=0.02)
    assert data["hotspot_count"] == 2


def test_date_filter_excludes_everything_outside_period(client):
    data = client.get("/api/v1/report", params={"date_from": "2025-01-01"}).json()
    assert data["total_burned_area_ha"] == 0
    assert data["hotspot_count"] == 0


def test_bbox_clipping_reduces_area_consistently(client):
    full = client.get("/api/v1/report").json()["total_burned_area_ha"]
    burns = client.get("/api/v1/burns").json()
    xs = [c[0] for f in burns["features"] for c in f["geometry"]["coordinates"][0]]
    ys = [c[1] for f in burns["features"] for c in f["geometry"]["coordinates"][0]]
    mid_x = (min(xs) + max(xs)) / 2
    bbox = f"{min(xs) - 0.01},{min(ys) - 0.01},{mid_x},{max(ys) + 0.01}"
    part = client.get("/api/v1/report", params={"bbox": bbox}).json()["total_burned_area_ha"]
    assert 0 < part < full


def test_polygon_post_matches_bbox_get(client):
    burns = client.get("/api/v1/burns").json()
    coords = [c for f in burns["features"] for c in f["geometry"]["coordinates"][0]]
    w, s = min(c[0] for c in coords) - 0.01, min(c[1] for c in coords) - 0.01
    e, n = max(c[0] for c in coords) + 0.01, max(c[1] for c in coords) + 0.01
    polygon = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    a = client.post("/api/v1/report", json={"polygon": polygon}).json()
    b = client.get("/api/v1/report", params={"bbox": f"{w},{s},{e},{n}"}).json()
    assert a["total_burned_area_ha"] == b["total_burned_area_ha"]


def test_burn_features_carry_required_attributes(client):
    feature = client.get("/api/v1/burns").json()["features"][0]
    assert {"contour_id", "severity", "severity_name", "area_ha"} <= set(feature["properties"])


def test_shapefile_and_geojson_export(client):
    shp = client.get("/api/v1/burns/export", params={"format": "shp"})
    assert shp.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(shp.content)).namelist()
    assert {"burns.shp", "burns.dbf", "burns.shx", "burns.prj"} <= set(names)
    gj = client.get("/api/v1/burns/export", params={"format": "geojson"})
    assert gj.json()["type"] == "FeatureCollection"


def test_csv_report(client):
    text = client.get("/api/v1/report", params={"format": "csv"}).text
    assert "severity_class" in text.splitlines()[0]


def test_invalid_bbox_is_422(client):
    assert client.get("/api/v1/report", params={"bbox": "10,10,5,5"}).status_code == 422
