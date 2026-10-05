from pathlib import Path

import numpy as np

from .features_bs import sar_only_indices
from .lgbm_utils import fit_binary, load_booster

MONOTONE_INCREASING = ("dnbr", "rdnbr", "dsavi", "dbai", "dnbr_corr")
MONOTONE_DECREASING = ("dmirbi",)
CLOUD_FULL_WEIGHT = 0.5


def monotone_vector(names: list[str]) -> list[int]:
    return [1 if n in MONOTONE_INCREASING else -1 if n in MONOTONE_DECREASING else 0 for n in names]


def fit_cascade(rows: np.ndarray, labels: np.ndarray, rounds: int, names: list[str], monotone: bool, weights: np.ndarray | None = None) -> list:
    overrides = {"monotone_constraints": monotone_vector(names)} if monotone else {}
    return [fit_binary(rows, (labels >= k).astype(np.float32), rounds, names, weight=weights, **overrides) for k in (1, 2, 3)]


def fit_sar_cascade(rows: np.ndarray, labels: np.ndarray, rounds: int, names: list[str]) -> list:
    keep = sar_only_indices(names)
    sub_names = [names[i] for i in keep]
    sub_rows = np.ascontiguousarray(rows[:, keep])
    return [fit_binary(sub_rows, (labels >= k).astype(np.float32), rounds, sub_names) for k in (1, 2, 3)]


def predict_cascade(models: list, features: np.ndarray, columns: list[int] | None = None, threads: int = 4) -> np.ndarray:
    selected = features if columns is None else features[columns]
    flat = selected.reshape(selected.shape[0], -1).T
    shape = features.shape[1:]
    return np.stack([m.predict(flat, num_threads=threads).reshape(shape) for m in models])


def blend_probabilities(full: np.ndarray, sar: np.ndarray | None, cloud_local: np.ndarray | None, scale: float) -> np.ndarray:
    if sar is None or cloud_local is None or scale <= 0:
        return full
    weight = np.clip(cloud_local / scale, 0.0, 1.0)
    return (1.0 - weight) * full + weight * sar


def save_models(model_dir: Path, tag: str, full: list, sar: list | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for k, model in zip((1, 2, 3), full):
        model.save_model(str(model_dir / f"bs_ge{k}{tag}.txt"))
    if sar is not None:
        for k, model in zip((1, 2, 3), sar):
            model.save_model(str(model_dir / f"bs_sar_ge{k}{tag}.txt"))


def load_models(model_dir: Path, tag: str, with_sar: bool) -> tuple[list, list | None]:
    full = [load_booster(model_dir / f"bs_ge{k}{tag}.txt") for k in (1, 2, 3)]
    sar = [load_booster(model_dir / f"bs_sar_ge{k}{tag}.txt") for k in (1, 2, 3)] if with_sar else None
    return full, sar
