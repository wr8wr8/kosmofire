import numpy as np
import pandas as pd


def af_groups(meta: pd.DataFrame, chip_ids: list[str], block_m: float = 200_000.0) -> np.ndarray:
    frame = meta.set_index("chip_id").loc[chip_ids]
    year = pd.to_datetime(frame["acq_datetime"], utc=True).dt.year.to_numpy()
    bx = np.floor(frame["x_min"].to_numpy() / block_m).astype(int)
    by = np.floor(frame["y_min"].to_numpy() / block_m).astype(int)
    epsg = frame["epsg"].to_numpy().astype(int)
    labels = [f"{y}_{e}_{x}_{yy}" for y, e, x, yy in zip(year, epsg, bx, by)]
    codes, _ = pd.factorize(pd.Series(labels))
    return codes


def bs_groups(meta: pd.DataFrame, chip_ids: list[str], block_m: float = 100_000.0) -> np.ndarray:
    frame = meta.set_index("chip_id").loc[chip_ids]
    bx = np.floor(frame["x_min"].to_numpy() / block_m).astype(int)
    by = np.floor(frame["y_min"].to_numpy() / block_m).astype(int)
    epsg = frame["epsg"].to_numpy().astype(int)
    labels = [_event_label(frame["fire_event_id"].iloc[i], e, x, y) for i, (e, x, y) in enumerate(zip(epsg, bx, by))]
    codes, _ = pd.factorize(pd.Series(labels))
    return codes


def _event_label(event_id, epsg: int, bx: int, by: int) -> str:
    if event_id is not None and not (isinstance(event_id, float) and np.isnan(event_id)):
        text = str(event_id).strip()
        if text and text.lower() != "nan":
            return f"event_{text}"
    return f"block_{epsg}_{bx}_{by}"
