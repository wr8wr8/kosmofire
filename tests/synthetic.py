import numpy as np

from kosmofire.features_af import af_feature_names
from kosmofire.features_bs import bs_feature_names


def make_af_chip(size: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    viirs = np.zeros((8, size, size), dtype=np.float32)
    viirs[0] = 0.08 + rng.normal(0, 0.005, (size, size))
    viirs[1] = 0.30 + rng.normal(0, 0.01, (size, size))
    viirs[2] = 0.15 + rng.normal(0, 0.01, (size, size))
    viirs[3] = 295.0 + rng.normal(0, 0.8, (size, size))
    viirs[4] = 285.0 + rng.normal(0, 0.5, (size, size))
    viirs[5] = 40.0
    viirs[6] = 20.0
    viirs[7] = 1.0
    aux = np.zeros((5, size, size), dtype=np.float32)
    aux[0] = 30
    aux[1] = 100
    aux[2] = 290
    aux[3] = 50
    aux[4] = 3
    return viirs, aux


def make_bs_chip(size: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    s2_pre = np.zeros((10, size, size), dtype=np.uint16)
    for band, value in enumerate([300, 500, 400, 1200, 2800, 3200, 3400, 1800, 900]):
        s2_pre[band] = value + rng.integers(-20, 20, (size, size))
    s2_pre[9] = 4
    s2_post = s2_pre.copy()
    s2_post[6, 20:40, 20:40] = 1200
    s2_post[8, 20:40, 20:40] = 2400
    s1_pre = np.zeros((2, size, size), dtype=np.int16)
    s1_pre[0] = -1000 + rng.integers(-50, 50, (size, size))
    s1_pre[1] = -1800 + rng.integers(-50, 50, (size, size))
    s1_post = s1_pre.copy()
    s1_post[1, 20:40, 20:40] -= 400
    aux = np.zeros((3, size, size), dtype=np.int16)
    aux[0] = 100
    aux[1] = 4
    aux[2] = 10
    return s2_pre, s2_post, s1_pre, s1_post, aux


def af_column(features: np.ndarray, name: str) -> np.ndarray:
    return features[af_feature_names().index(name)]


def bs_column(features: np.ndarray, name: str) -> np.ndarray:
    return features[bs_feature_names().index(name)]
