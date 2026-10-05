import numpy as np
from scipy import ndimage as ndi

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if min_size <= 1 or not binary.any():
        return binary
    labels, count = ndi.label(binary, structure=EIGHT_CONNECTED)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def clean_severity(severity: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return severity
    kept = remove_small_components(severity >= 1, min_size)
    return np.where(kept, severity, 0).astype(np.uint8)
