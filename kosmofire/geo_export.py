import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio import features as rio_features
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from .config import BS_PIXEL_AREA_HA

MIN_POLYGON_PIXELS = 5
SEVERITY_NAMES = {1: "слабая", 2: "средняя", 3: "сильная"}
SEVERITY_KEYS = {1: "low", 2: "moderate", 3: "high"}


def _to_wgs84(epsg: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)


def af_points(chip_row: pd.Series, mask: np.ndarray, probability: np.ndarray | None = None) -> list[dict]:
    rows, cols = np.nonzero(mask == 1)
    if rows.size == 0:
        return []
    gsd = float(chip_row["gsd"])
    x = chip_row["x_min"] + (cols + 0.5) * gsd
    y = chip_row["y_max"] - (rows + 0.5) * gsd
    lon, lat = _to_wgs84(int(chip_row["epsg"])).transform(x, y)
    records = []
    for i in range(rows.size):
        records.append(
            {
                "chip_id": chip_row["chip_id"],
                "lon": float(lon[i]),
                "lat": float(lat[i]),
                "detected_at": str(chip_row["acq_datetime"]),
                "satellite": str(chip_row["satellite"]),
                "probability": float(probability[rows[i], cols[i]]) if probability is not None else None,
            }
        )
    return records


def burn_polygons(chip_row: pd.Series, severity: np.ndarray) -> list[dict]:
    epsg = int(chip_row["epsg"])
    gsd = float(chip_row["gsd"])
    transform = Affine(gsd, 0, chip_row["x_min"], 0, -gsd, chip_row["y_max"])
    to_wgs = _to_wgs84(epsg)
    records = []
    for klass in (1, 2, 3):
        class_mask = (severity == klass).astype(np.uint8)
        if not class_mask.any():
            continue
        for geom, value in rio_features.shapes(class_mask, mask=class_mask.astype(bool), transform=transform, connectivity=8):
            if value != 1:
                continue
            utm_geom = shape(geom)
            area_ha = utm_geom.area / 10_000.0
            if area_ha < MIN_POLYGON_PIXELS * BS_PIXEL_AREA_HA * (gsd / 20.0) ** 2:
                continue
            wgs_geom = shapely_transform(to_wgs.transform, utm_geom)
            records.append(
                {
                    "chip_id": chip_row["chip_id"],
                    "severity": klass,
                    "severity_key": SEVERITY_KEYS[klass],
                    "severity_name": SEVERITY_NAMES[klass],
                    "area_ha": float(area_ha),
                    "epsg_utm": epsg,
                    "date_pre": str(chip_row["date_pre"]),
                    "date_post": str(chip_row["date_post"]),
                    "geometry": wgs_geom,
                }
            )
    return records
