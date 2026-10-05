import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .config import ChipFiles

ROLE_PATTERNS = {
    "viirs": re.compile(r"^(?P<id>AF_[a-z]+_\d+)_VIIRS_I1-I5$"),
    "aux": re.compile(r"^(?P<id>(?:AF|BS)_[a-z]+_\d+)_AUX$"),
    "mask": re.compile(r"^(?P<id>(?:AF|BS)_[a-z]+_\d+)_MASK$"),
    "s2_pre": re.compile(r"^(?P<id>BS_[a-z]+_\d+)_Sentinel-2_pre$"),
    "s2_post": re.compile(r"^(?P<id>BS_[a-z]+_\d+)_Sentinel-2_post$"),
    "s1_pre": re.compile(r"^(?P<id>BS_[a-z]+_\d+)_Sentinel-1_pre$"),
    "s1_post": re.compile(r"^(?P<id>BS_[a-z]+_\d+)_Sentinel-1_post$"),
}


class ChipStore:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self._files: dict[str, dict[str, Path]] = {}
        for path in sorted(self.data_dir.rglob("*.tif")):
            for role, pattern in ROLE_PATTERNS.items():
                match = pattern.match(path.stem)
                if match:
                    self._files.setdefault(match.group("id"), {})[role] = path
                    break

    def chip_ids(self, kind: str | None = None) -> list[str]:
        prefix = {"af": "AF_", "bs": "BS_"}.get(kind, "")
        return sorted(cid for cid in self._files if cid.startswith(prefix))

    def files(self, chip_id: str) -> ChipFiles:
        kind = "af" if chip_id.startswith("AF_") else "bs"
        return ChipFiles(chip_id, kind, self._files[chip_id])

    def has(self, chip_id: str) -> bool:
        return chip_id in self._files

    def meta(self) -> pd.DataFrame | None:
        frames = [pd.read_csv(p) for p in sorted(self.data_dir.rglob("meta.csv"))]
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


def read_raster_with_profile(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(), {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
        }


def read_mask(path: Path) -> np.ndarray:
    return read_raster(path)[0]
