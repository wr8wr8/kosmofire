import numpy as np

from .ordinal import decode_severity
from .postprocess import clean_severity


def decide_severity(p_ge1: np.ndarray, p_ge2: np.ndarray, p_ge3: np.ndarray, keep: np.ndarray | None, burn_threshold: float, min_component: int) -> np.ndarray:
    severity = decode_severity(p_ge1, p_ge2, p_ge3, burn_threshold)
    if keep is not None:
        severity = np.where(keep, severity, 0).astype(np.uint8)
    return clean_severity(severity, min_component)
