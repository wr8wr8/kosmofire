import io
import json
import os
import tempfile
import zipfile
from datetime import date
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .report import build_report, report_csv
from .store import DEFAULT_GPKG, Query as SpatialQuery, ServiceStore, geometry_from_bbox, geometry_from_geojson

STATIC_DIR = Path(__file__).parent / "static"


def create_app(gpkg_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="KosmoFire", version="1.0.0")
    store = ServiceStore(gpkg_path or Path(os.environ.get("KOSMOFIRE_GPKG", DEFAULT_GPKG)))

    def parse_query(bbox: str | None, date_from: date | None, date_to: date | None, polygon: dict | None = None) -> SpatialQuery:
        try:
            geometry = geometry_from_geojson(polygon) if polygon else (geometry_from_bbox(bbox) if bbox else None)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from is after date_to")
        return SpatialQuery(geometry, date_from, date_to)

    def hotspots_payload(query: SpatialQuery) -> dict:
        frame = store.select_hotspots(query)
        return json.loads(frame.drop(columns=["detected_date"]).to_json(drop_id=True))

    def burns_payload(query: SpatialQuery) -> dict:
        frame = store.select_burns(query)
        frame = frame.drop(columns=["period_date"]).reset_index(drop=True)
        frame.insert(0, "contour_id", [f"C{i + 1:05d}" for i in range(len(frame))])
        frame["area_ha"] = frame["area_ha"].round(3)
        return json.loads(frame.to_json(drop_id=True))

    @app.get("/api/v1/health")
    def health() -> dict:
        return {"status": "ok", "hotspots": len(store.hotspots), "burn_polygons": len(store.burns)}

    @app.get("/api/v1/hotspots")
    def hotspots(bbox: str | None = None, date_from: date | None = None, date_to: date | None = None):
        return JSONResponse(hotspots_payload(parse_query(bbox, date_from, date_to)), media_type="application/geo+json")

    @app.post("/api/v1/hotspots")
    def hotspots_post(payload: dict = Body(...)):
        query = parse_query(None, payload.get("date_from"), payload.get("date_to"), payload.get("polygon"))
        return JSONResponse(hotspots_payload(query), media_type="application/geo+json")

    @app.get("/api/v1/burns")
    def burns(bbox: str | None = None, date_from: date | None = None, date_to: date | None = None):
        return JSONResponse(burns_payload(parse_query(bbox, date_from, date_to)), media_type="application/geo+json")

    @app.post("/api/v1/burns")
    def burns_post(payload: dict = Body(...)):
        query = parse_query(None, payload.get("date_from"), payload.get("date_to"), payload.get("polygon"))
        return JSONResponse(burns_payload(query), media_type="application/geo+json")

    @app.get("/api/v1/burns/export")
    def burns_export(bbox: str | None = None, date_from: date | None = None, date_to: date | None = None, format: str = Query("geojson", pattern="^(geojson|shp)$")):
        query = parse_query(bbox, date_from, date_to)
        frame = store.select_burns(query).drop(columns=["period_date"]).reset_index(drop=True)
        frame.insert(0, "contour_id", [f"C{i + 1:05d}" for i in range(len(frame))])
        frame["area_ha"] = frame["area_ha"].round(3)
        if format == "geojson":
            return Response(frame.to_json(drop_id=True), media_type="application/geo+json", headers={"Content-Disposition": "attachment; filename=burns.geojson"})
        with tempfile.TemporaryDirectory() as tmp:
            shp_frame = frame.rename(columns={"severity_key": "sev_key", "severity_name": "sev_name", "contour_id": "contour", "date_post": "date_post", "date_pre": "date_pre"})
            shp_frame.to_file(Path(tmp) / "burns.shp", encoding="utf-8")
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(Path(tmp).iterdir()):
                    archive.write(path, path.name)
        return Response(buffer.getvalue(), media_type="application/zip", headers={"Content-Disposition": "attachment; filename=burns_shapefile.zip"})

    def report_for(query: SpatialQuery) -> dict:
        burns_frame = store.select_burns(query)
        hotspot_frame = store.select_hotspots(query)
        report = build_report(burns_frame, len(hotspot_frame))
        report["query"] = {
            "date_from": query.date_from.isoformat() if query.date_from else None,
            "date_to": query.date_to.isoformat() if query.date_to else None,
            "bbox": list(query.geometry.bounds) if query.geometry is not None else None,
        }
        return report

    @app.get("/api/v1/report")
    def report(bbox: str | None = None, date_from: date | None = None, date_to: date | None = None, format: str = Query("json", pattern="^(json|csv)$")):
        data = report_for(parse_query(bbox, date_from, date_to))
        if format == "csv":
            return Response(report_csv(data), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=report.csv"})
        return data

    @app.post("/api/v1/report")
    def report_post(payload: dict = Body(...)):
        query = parse_query(None, payload.get("date_from"), payload.get("date_to"), payload.get("polygon"))
        return report_for(query)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
