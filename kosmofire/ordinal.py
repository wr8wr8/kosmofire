import numpy as np


def pava_nonincreasing(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    c = np.asarray(q3, dtype=np.float64)

    merged_ab = a < b
    m_ab = (a + b) / 2.0
    all_mean = (a + b + c) / 3.0

    ab_then_c = merged_ab & (m_ab < c)
    ab_only = merged_ab & ~ab_then_c

    m_bc = (b + c) / 2.0
    bc_merge = ~merged_ab & (b < c)
    bc_all = bc_merge & (a < m_bc)
    bc_only = bc_merge & ~bc_all

    r1 = a.copy()
    r2 = b.copy()
    r3 = c.copy()

    r1 = np.where(ab_only, m_ab, r1)
    r2 = np.where(ab_only, m_ab, r2)

    r2 = np.where(bc_only, m_bc, r2)
    r3 = np.where(bc_only, m_bc, r3)

    everything = ab_then_c | bc_all
    r1 = np.where(everything, all_mean, r1)
    r2 = np.where(everything, all_mean, r2)
    r3 = np.where(everything, all_mean, r3)
    return r1, r2, r3


def enforce_monotonic(p_ge1: np.ndarray, p_ge2: np.ndarray, p_ge3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pava_nonincreasing(p_ge1, p_ge2, p_ge3)


def class_probabilities(p_ge1: np.ndarray, p_ge2: np.ndarray, p_ge3: np.ndarray) -> np.ndarray:
    q1, q2, q3 = pava_nonincreasing(p_ge1, p_ge2, p_ge3)
    return np.stack([1.0 - q1, q1 - q2, q2 - q3, q3], axis=-1)


def decode_severity(p_ge1: np.ndarray, p_ge2: np.ndarray, p_ge3: np.ndarray, burn_threshold: float = 0.5) -> np.ndarray:
    q1, q2, q3 = pava_nonincreasing(p_ge1, p_ge2, p_ge3)
    burned_probabilities = np.stack([q1 - q2, q2 - q3, q3], axis=-1)
    severity = np.argmax(burned_probabilities, axis=-1) + 1
    return np.where(np.asarray(p_ge1) >= burn_threshold, severity, 0).astype(np.uint8)
