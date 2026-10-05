import json
from pathlib import Path

import numpy as np

ENDMEMBER_NAMES = ("char", "vegetation", "soil")
NDVI_VEGETATION_MIN = 0.55
NDVI_SOIL_MAX = 0.2
MAX_PIXELS_PER_CLASS = 400_000


def unmix(reflectance: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    bands, height, width = reflectance.shape
    solver = np.linalg.pinv(endmembers.astype(np.float64))
    flat = reflectance.reshape(bands, -1).astype(np.float64)
    fractions = np.clip(solver @ flat, 0.0, 1.0)
    total = fractions.sum(axis=0, keepdims=True)
    normalised = np.where(total > 1e-9, fractions / np.maximum(total, 1e-9), 1.0 / fractions.shape[0])
    return normalised.reshape(fractions.shape[0], height, width).astype(np.float32)


def estimate_endmembers(samples: dict[str, np.ndarray]) -> np.ndarray:
    columns = []
    for name in ENDMEMBER_NAMES:
        pixels = samples[name]
        if pixels.shape[0] == 0:
            raise ValueError(f"no training pixels collected for endmember {name}")
        columns.append(np.median(pixels, axis=0))
    return np.stack(columns, axis=1)


def collect_endmember_samples(pre: np.ndarray, post: np.ndarray, severity: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    b4_pre, b8a_pre = pre[2], pre[6]
    b4_post, b8a_post = post[2], post[6]
    ndvi_pre = (b8a_pre - b4_pre) / (b8a_pre + b4_pre + 1e-6)
    ndvi_post = (b8a_post - b4_post) / (b8a_post + b4_post + 1e-6)
    char = valid & (severity == 3) & (ndvi_post < 0.3)
    vegetation = valid & (severity == 0) & (ndvi_pre > NDVI_VEGETATION_MIN)
    soil = valid & (severity == 0) & (ndvi_pre < NDVI_SOIL_MAX) & (ndvi_post < NDVI_SOIL_MAX) & (pre[8] > 0.05)
    return {
        "char": post[:, char].T,
        "vegetation": pre[:, vegetation].T,
        "soil": pre[:, soil].T,
    }


def save_endmembers(path: Path, endmembers: np.ndarray) -> None:
    path.write_text(json.dumps({"names": list(ENDMEMBER_NAMES), "spectra": endmembers.tolist()}))


def load_endmembers(path: Path) -> np.ndarray:
    return np.array(json.loads(path.read_text())["spectra"], dtype=np.float64)
