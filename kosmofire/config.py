from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "work"

SEED = 20260918
NUM_THREADS = 4

AF_CHIP_SIZE = 256
BS_CHIP_SIZE = 512
BS_PIXEL_AREA_HA = 0.04

I4_SATURATION_K = 366.0
REFLECTANCE_SCALE = 10000.0
SAR_DB_SCALE = 100.0

SCL_CLOUD_CLASSES = (3, 8, 9, 10)
SCL_WATER_CLASS = 6

AF_BAND_NAMES = ("I1", "I2", "I3", "I4", "I5", "solar_zenith", "sensor_zenith", "valid")
AF_AUX_NAMES = ("landcover", "dem", "t2m", "rh2m", "wind_speed")
BS_S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12", "SCL")
BS_AUX_NAMES = ("dem", "slope", "landcover")

WORLDCOVER_CLASSES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)


@dataclass(frozen=True)
class ChipFiles:
    chip_id: str
    kind: str
    paths: dict[str, Path]
