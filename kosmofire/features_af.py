import numpy as np
from scipy import ndimage as ndi

from .config import I4_SATURATION_K
from .dozier import dozier_retrieval
from .landcover import landcover_index, landcover_one_hot, one_hot_names

EPS = 1e-6
CONTEXT_WINDOWS = (7, 15)


def _robust_context(band: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    background = ndi.median_filter(band, size=window, mode="reflect")
    deviation = np.abs(band - background)
    spread = ndi.median_filter(deviation, size=window, mode="reflect")
    z = (band - background) / (spread + 1.0)
    return background, spread, z


DEFAULT_GROUPS = ("dozier", "global")
DOZIER_NAMES = ["p_subpixel", "tf_estimated", "dozier_converged"]
GLOBAL_NAMES = ["i4_global_bg", "i4_global_mad", "i4_global_z", "dmt_global_z"]


def af_feature_names(groups: tuple[str, ...] = DEFAULT_GROUPS) -> list[str]:
    names = ["I1", "I2", "I3", "I4", "I5", "delta_mir_tir", "i4_saturated", "glint", "ndvi_like"]
    for window in CONTEXT_WINDOWS:
        names += [
            f"i4_bg_{window}",
            f"i4_mad_{window}",
            f"i4_z_{window}",
            f"dmt_bg_{window}",
            f"dmt_mad_{window}",
            f"dmt_z_{window}",
        ]
    names += ["i4_minus_bg_7", "i4_local_max_3", "hot_neighbours_3"]
    names += ["solar_zenith", "sensor_zenith", "dem", "t2m", "rh2m", "wind_speed"]
    names += one_hot_names() if "onehot" in groups else ["landcover"]
    if "global" in groups:
        names += GLOBAL_NAMES
    if "dozier" in groups:
        names += DOZIER_NAMES
    return names


def af_features(viirs: np.ndarray, aux: np.ndarray, groups: tuple[str, ...] = DEFAULT_GROUPS) -> np.ndarray:
    i1, i2, i3, i4, i5 = (viirs[i].astype(np.float32) for i in range(5))
    solar_zenith = viirs[5].astype(np.float32)
    sensor_zenith = viirs[6].astype(np.float32)

    delta = i4 - i5
    saturated = (i4 >= I4_SATURATION_K).astype(np.float32)
    glint = i3 / (i2 + EPS)
    ndvi_like = (i2 - i1) / (i2 + i1 + EPS)

    layers = [i1, i2, i3, i4, i5, delta, saturated, glint, ndvi_like]

    for window in CONTEXT_WINDOWS:
        i4_bg, i4_mad, i4_z = _robust_context(i4, window)
        d_bg, d_mad, d_z = _robust_context(delta, window)
        layers += [i4_bg, i4_mad, i4_z, d_bg, d_mad, d_z]

    layers.append(i4 - ndi.median_filter(i4, size=7, mode="reflect"))
    local_max = ndi.maximum_filter(i4, size=3, mode="reflect")
    layers.append(local_max)
    hot = (i4 - ndi.uniform_filter(i4, size=15, mode="reflect")) > 8.0
    layers.append(ndi.uniform_filter(hot.astype(np.float32), size=3, mode="constant") * 9.0)

    layers += [solar_zenith, sensor_zenith]
    layers += [aux[1].astype(np.float32), aux[2].astype(np.float32), aux[3].astype(np.float32), aux[4].astype(np.float32)]

    if "onehot" in groups:
        layers += list(landcover_one_hot(aux[0]))
    else:
        layers.append(landcover_index(aux[0]))

    stacked = np.nan_to_num(np.stack(layers).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    extra = []
    if "global" in groups:
        i4_clean = np.nan_to_num(i4, nan=float(np.nanmedian(i4)) if np.isfinite(i4).any() else 0.0)
        global_bg = float(np.median(i4_clean))
        global_mad = float(np.median(np.abs(i4_clean - global_bg)))
        delta_clean = np.nan_to_num(delta, nan=0.0)
        delta_bg = float(np.median(delta_clean))
        delta_mad = float(np.median(np.abs(delta_clean - delta_bg)))
        extra += [
            np.full(i4.shape, global_bg, dtype=np.float32),
            np.full(i4.shape, global_mad, dtype=np.float32),
            ((i4_clean - global_bg) / (global_mad + 1.0)).astype(np.float32),
            ((delta_clean - delta_bg) / (delta_mad + 1.0)).astype(np.float32),
        ]
    if "dozier" in groups:
        bg4 = ndi.median_filter(np.nan_to_num(i4, nan=295.0), size=15, mode="reflect")
        bg5 = ndi.median_filter(np.nan_to_num(i5, nan=285.0), size=15, mode="reflect")
        p, tf, converged = dozier_retrieval(i4, i5, bg4, bg5)
        extra += [p.astype(np.float32), tf.astype(np.float32), converged.astype(np.float32)]
    if extra:
        stacked = np.concatenate([stacked, np.stack(extra)], axis=0)
    return stacked


def af_valid_mask(viirs: np.ndarray) -> np.ndarray:
    valid = viirs[7] > 0.5
    finite = np.isfinite(viirs[3]) & np.isfinite(viirs[4])
    return valid & finite
