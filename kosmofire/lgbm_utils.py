from pathlib import Path

import lightgbm as lgb
import numpy as np

from .config import NUM_THREADS, SEED


def base_params(**overrides) -> dict:
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": NUM_THREADS,
    }
    params.update(overrides)
    return params


def fit_binary(x: np.ndarray, y: np.ndarray, rounds: int, names: list[str], weight: np.ndarray | None = None, **overrides) -> lgb.Booster:
    categorical = [n for n in names if n == "landcover"]
    dataset = lgb.Dataset(x, label=y, weight=weight, feature_name=names, categorical_feature=categorical or "auto", free_raw_data=True)
    return lgb.train(base_params(**overrides), dataset, num_boost_round=rounds)


def load_booster(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
