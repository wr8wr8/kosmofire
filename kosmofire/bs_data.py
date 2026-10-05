from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .chipio import read_mask, read_raster
from .config import REFLECTANCE_SCALE, SCL_CLOUD_CLASSES
from .features_bs import DEFAULT_GROUPS, bs_features, estimate_kndvi_sigma
from .parallel import parallel_map
from .unmixing import collect_endmember_samples, estimate_endmembers

SAMPLE_RATE = 0.06
BOUNDARY_RATE = 0.35
BOUNDARY_ITERATIONS = 6
ENDMEMBER_PIXELS_PER_CHIP = 4000


@dataclass(frozen=True)
class BSParams:
    groups: tuple[str, ...] = DEFAULT_GROUPS
    endmembers: np.ndarray | None = None
    kndvi_sigma: float | None = None


def load_bs_features(paths: dict[str, Path], params: BSParams) -> tuple[np.ndarray, np.ndarray]:
    arrays = [read_raster(paths[k]) for k in ("s2_pre", "s2_post", "s1_pre", "s1_post", "aux")]
    return bs_features(*arrays, groups=params.groups, endmembers=params.endmembers, kndvi_sigma=params.kndvi_sigma)


def _sample_chip(args):
    paths, seed, params = args
    features, _ = load_bs_features(paths, params)
    truth = read_mask(paths["mask"]).astype(np.uint8)
    rng = np.random.default_rng(seed)
    burned = truth >= 1
    boundary = ndi.binary_dilation(burned, iterations=BOUNDARY_ITERATIONS) & ~ndi.binary_erosion(burned, iterations=BOUNDARY_ITERATIONS)
    draw = rng.random(truth.shape)
    keep = (draw < SAMPLE_RATE) | (boundary & (rng.random(truth.shape) < BOUNDARY_RATE))
    inclusion = np.where(boundary, 1.0 - (1.0 - SAMPLE_RATE) * (1.0 - BOUNDARY_RATE), SAMPLE_RATE)
    return features[:, keep].T, truth[keep], (1.0 / inclusion[keep]).astype(np.float32), int(keep.sum())


def sample_bs_training_rows(paths_list, seed: int, params: BSParams):
    results = parallel_map(_sample_chip, [(p, seed + i, params) for i, p in enumerate(paths_list)])
    rows = np.concatenate([r[0] for r in results])
    labels = np.concatenate([r[1] for r in results])
    weights = np.concatenate([r[2] for r in results])
    owners = np.concatenate([np.full(r[3], i, dtype=np.int32) for i, r in enumerate(results)])
    return rows, labels, weights, owners


def _endmember_chip(args) -> dict[str, np.ndarray]:
    paths, seed = args
    pre = read_raster(paths["s2_pre"])
    post = read_raster(paths["s2_post"])
    truth = read_mask(paths["mask"]).astype(np.uint8)
    reflectance_pre = pre[:9].astype(np.float32) / REFLECTANCE_SCALE
    reflectance_post = post[:9].astype(np.float32) / REFLECTANCE_SCALE
    clear = ~np.isin(pre[9], SCL_CLOUD_CLASSES) & ~np.isin(post[9], SCL_CLOUD_CLASSES)
    samples = collect_endmember_samples(reflectance_pre, reflectance_post, truth, clear)
    rng = np.random.default_rng(seed)
    out = {}
    for name, pixels in samples.items():
        if pixels.shape[0] > ENDMEMBER_PIXELS_PER_CHIP:
            pixels = pixels[rng.choice(pixels.shape[0], ENDMEMBER_PIXELS_PER_CHIP, replace=False)]
        out[name] = pixels
    return out


def fit_endmembers(paths_list, seed: int) -> np.ndarray:
    results = parallel_map(_endmember_chip, [(p, seed + i) for i, p in enumerate(paths_list)])
    merged = {name: np.concatenate([r[name] for r in results]) for name in ("char", "vegetation", "soil")}
    return estimate_endmembers(merged)


def _sigma_chip(paths) -> tuple[np.ndarray, np.ndarray]:
    pre = read_raster(paths["s2_pre"]).astype(np.float32) / REFLECTANCE_SCALE
    post = read_raster(paths["s2_post"]).astype(np.float32) / REFLECTANCE_SCALE
    return np.concatenate([pre[6].ravel(), post[6].ravel()]), np.concatenate([pre[2].ravel(), post[2].ravel()])


def fit_kndvi_sigma(paths_list) -> float:
    return estimate_kndvi_sigma(parallel_map(_sigma_chip, paths_list))
