import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(prob: np.ndarray, truth: np.ndarray, max_samples: int = 2_000_000, seed: int = 0) -> IsotonicRegression:
    flat_p = prob.ravel()
    flat_t = truth.ravel()
    if flat_p.size > max_samples:
        rng = np.random.default_rng(seed)
        positives = np.flatnonzero(flat_t == 1)
        negatives = np.flatnonzero(flat_t == 0)
        keep_neg = rng.choice(negatives, size=min(len(negatives), max_samples - len(positives)), replace=False)
        index = np.concatenate([positives, keep_neg])
        weights = np.concatenate([np.ones(len(positives)), np.full(len(keep_neg), len(negatives) / max(len(keep_neg), 1))])
        flat_p, flat_t = flat_p[index], flat_t[index]
    else:
        weights = None
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(flat_p, flat_t, sample_weight=weights)
    return model


def apply_isotonic(model: IsotonicRegression, prob: np.ndarray) -> np.ndarray:
    return model.predict(prob.ravel()).reshape(prob.shape).astype(np.float32)


def blend(prob_a: np.ndarray, prob_b: np.ndarray, weight_a: float) -> np.ndarray:
    return weight_a * prob_a + (1.0 - weight_a) * prob_b
