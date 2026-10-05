import numpy as np

C1 = 1.191042e8
C2 = 1.4387752e4
LAMBDA_MIR = 3.74
LAMBDA_TIR = 11.45
TF_MIN = 320.0
TF_MAX = 1500.0
BISECTION_STEPS = 40
MIN_CONTRAST_K = 1.0


def planck(wavelength_um: float, temperature_k: np.ndarray) -> np.ndarray:
    t = np.maximum(np.asarray(temperature_k, dtype=np.float64), 1.0)
    x = C2 / (wavelength_um * t)
    return C1 / (wavelength_um**5 * np.expm1(np.minimum(x, 700.0)))


def inverse_planck(wavelength_um: float, radiance: np.ndarray) -> np.ndarray:
    r = np.maximum(np.asarray(radiance, dtype=np.float64), 1e-12)
    return C2 / (wavelength_um * np.log1p(C1 / (wavelength_um**5 * r)))


def _mixture_residual(tf: np.ndarray, l4: np.ndarray, l5: np.ndarray, lb4: np.ndarray, lb5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b4 = planck(LAMBDA_MIR, tf)
    b5 = planck(LAMBDA_TIR, tf)
    p = (l4 - lb4) / (b4 - lb4)
    return (p * (b5 - lb5) + lb5 - l5), p


def dozier_retrieval(i4_k: np.ndarray, i5_k: np.ndarray, bg4_k: np.ndarray, bg5_k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i4 = np.asarray(i4_k, dtype=np.float64)
    i5 = np.asarray(i5_k, dtype=np.float64)
    l4 = planck(LAMBDA_MIR, i4)
    l5 = planck(LAMBDA_TIR, i5)
    lb4 = planck(LAMBDA_MIR, np.asarray(bg4_k, dtype=np.float64))
    lb5 = planck(LAMBDA_TIR, np.asarray(bg5_k, dtype=np.float64))

    finite = np.isfinite(i4) & np.isfinite(i5) & np.isfinite(bg4_k) & np.isfinite(bg5_k)
    contrast = finite & (i4 - bg4_k > MIN_CONTRAST_K)

    lo = np.full(i4.shape, TF_MIN)
    hi = np.full(i4.shape, TF_MAX)
    with np.errstate(all="ignore"):
        f_lo, _ = _mixture_residual(lo, l4, l5, lb4, lb5)
        f_hi, _ = _mixture_residual(hi, l4, l5, lb4, lb5)
        b4_lo = planck(LAMBDA_MIR, lo)
        p_lo_valid = (l4 - lb4) / (b4_lo - lb4)
        bracket = contrast & np.isfinite(f_lo) & np.isfinite(f_hi) & (np.sign(f_lo) != np.sign(f_hi))

        lo = np.where(p_lo_valid > 1.0, np.where(bracket, _tf_for_full_pixel(l4), lo), lo)
        f_lo, _ = _mixture_residual(lo, l4, l5, lb4, lb5)
        bracket &= np.isfinite(f_lo) & (np.sign(f_lo) != np.sign(f_hi))

        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            f_mid, _ = _mixture_residual(mid, l4, l5, lb4, lb5)
            same_as_lo = np.sign(f_mid) == np.sign(f_lo)
            lo = np.where(same_as_lo, mid, lo)
            f_lo = np.where(same_as_lo, f_mid, f_lo)
            hi = np.where(same_as_lo, hi, mid)

        tf = 0.5 * (lo + hi)
        residual, p = _mixture_residual(tf, l4, l5, lb4, lb5)

    valid = bracket & np.isfinite(p) & (p > 0.0) & (p <= 1.0 + 1e-6) & (np.abs(residual) <= 1e-3 * np.maximum(l5, 1e-9))
    p_out = np.where(valid, np.clip(p, 0.0, 1.0), np.nan)
    tf_out = np.where(valid, tf, np.nan)
    return p_out, tf_out, valid


def _tf_for_full_pixel(l4: np.ndarray) -> np.ndarray:
    return np.clip(inverse_planck(LAMBDA_MIR, l4), TF_MIN, TF_MAX)
