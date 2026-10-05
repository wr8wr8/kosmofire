from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    low: float
    high: float
    excludes_zero: bool
    samples: np.ndarray


def grouped_bootstrap_difference(
    per_group_a: dict,
    per_group_b: dict,
    score_fn: Callable[[list], float],
    iterations: int = 500,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    keys = sorted(per_group_a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations)
    for i in range(iterations):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        chosen = [keys[j] for j in picked]
        diffs[i] = score_fn([per_group_a[k] for k in chosen]) - score_fn([per_group_b[k] for k in chosen])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(diffs, [alpha, 1.0 - alpha])
    return BootstrapResult(float(diffs.mean()), float(low), float(high), bool(low > 0 or high < 0), diffs)
