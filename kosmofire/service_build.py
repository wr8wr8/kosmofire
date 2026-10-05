import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .chipio import ChipStore
from .geo_export import af_points, burn_polygons
from .bs_train import decode_oof, load_oof


def build_service_gpkg(train_dir: Path, work_dir: Path, model_dir: Path, output: Path, af_tag: str = "_oh", bs_tag: str = "_v2") -> dict:
    store = ChipStore(train_dir)
    meta = store.meta().set_index("chip_id", drop=False)
    af_threshold = json.loads((model_dir / "af_meta.json").read_text())["threshold"]
    bs_meta = json.loads((model_dir / "bs_meta.json").read_text())
    burn_threshold = bs_meta["burn_threshold"]
    min_component = int(bs_meta.get("min_component", 0))

    af_ids = pd.read_csv(work_dir / "af_ids.csv", header=None)[0].tolist()
    af_prob = np.load(work_dir / f"af_oof{af_tag}.npy")
    point_records = []
    for i, chip_id in enumerate(af_ids):
        row = meta.loc[chip_id]
        mask = (af_prob[i] >= af_threshold).astype(np.uint8)
        point_records += af_points(row, mask, af_prob[i])

    bs_ids = store.chip_ids("bs")
    polygon_records = []
    for i, chip_id in enumerate(bs_ids):
        row = meta.loc[chip_id]
        severity = decode_oof(load_oof(work_dir, bs_tag, i), burn_threshold, 0.0, min_component)
        polygon_records += burn_polygons(row, severity)

    hotspots = gpd.GeoDataFrame(
        point_records,
        geometry=[Point(r["lon"], r["lat"]) for r in point_records],
        crs="EPSG:4326",
    ).drop(columns=["lon", "lat"])
    burns = gpd.GeoDataFrame(polygon_records, geometry="geometry", crs="EPSG:4326")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    hotspots.to_file(output, layer="hotspots", driver="GPKG", engine="pyogrio")
    burns.to_file(output, layer="burns", driver="GPKG", engine="pyogrio")
    return {"hotspots": len(hotspots), "burn_polygons": len(burns), "burned_area_ha": float(burns["area_ha"].sum())}
