import json
import logging
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold

from .af_ensemble import best_threshold_f1
from .af_train import _gather, _predict_chips
from .chipio import ChipStore
from .config import SEED
from .features_af import af_feature_names
from .lgbm_utils import fit_binary
from .validation import af_groups

logger = logging.getLogger("af_optuna")


def tune_af(train_dir: Path, work_dir: Path, model_dir: Path, trials: int = 8, folds: int = 3, rounds: int = 300, tag: str = "", groups: tuple[str, ...] = ("onehot",)) -> dict:
    store = ChipStore(train_dir)
    chip_ids = store.chip_ids("af")
    meta = store.meta()
    names = af_feature_names(groups)
    x = np.load(work_dir / f"af_x{tag}.npy", mmap_mode="r")
    valid = np.load(work_dir / "af_valid.npy")
    truth = np.load(work_dir / "af_truth.npy")
    fold_groups = af_groups(meta[meta["kind"] == "af"], chip_ids)
    rng = np.random.default_rng(SEED)
    splits = list(GroupKFold(n_splits=folds).split(np.arange(len(chip_ids)), groups=fold_groups))
    sampled = [_gather(x, valid, truth, tr, names, rng) for tr, _ in splits]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 30.0, log=True),
        }
        oof = np.zeros(truth.shape, dtype=np.float32)
        for (tr, va), (rows, labels) in zip(splits, sampled):
            model = fit_binary(rows, labels, rounds, names, **params)
            oof[va] = _predict_chips(model, x, valid, va)
        mask = np.zeros(len(chip_ids), dtype=bool)
        for _, va in splits:
            mask[va] = True
        return best_threshold_f1(oof[mask], truth[mask])[1]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=trials)
    result = {"best_value": study.best_value, "best_params": study.best_params, "trials": [{"value": t.value, "params": t.params} for t in study.trials], "folds": folds, "rounds": rounds}
    (model_dir / f"af_optuna{tag}.json").write_text(json.dumps(result, indent=2))
    return result
