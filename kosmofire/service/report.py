import io

import geopandas as gpd
import pandas as pd

SEVERITY_ORDER = (("low", "слабая"), ("moderate", "средняя"), ("high", "сильная"))


def build_report(burns: gpd.GeoDataFrame, hotspot_count: int) -> dict:
    by_class = {key: 0.0 for key, _ in SEVERITY_ORDER}
    polygons = {key: 0 for key, _ in SEVERITY_ORDER}
    if not burns.empty:
        grouped = burns.groupby("severity_key")["area_ha"].agg(["sum", "count"])
        for key in by_class:
            if key in grouped.index:
                by_class[key] = float(grouped.loc[key, "sum"])
                polygons[key] = int(grouped.loc[key, "count"])
    total = sum(by_class.values())
    return {
        "total_burned_area_ha": round(total, 2),
        "hotspot_count": int(hotspot_count),
        "by_severity": [
            {
                "severity_class": key,
                "severity_name": name,
                "area_ha": round(by_class[key], 2),
                "share": round(by_class[key] / total, 4) if total else 0.0,
                "polygon_count": polygons[key],
            }
            for key, name in SEVERITY_ORDER
        ],
    }


def report_csv(report: dict) -> str:
    frame = pd.DataFrame(report["by_severity"])
    frame.loc[len(frame)] = ["total", "всего", report["total_burned_area_ha"], 1.0 if report["total_burned_area_ha"] else 0.0, int(frame["polygon_count"].sum())]
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()
