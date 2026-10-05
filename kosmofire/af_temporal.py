import json
from pathlib import Path

import numpy as np
import pandas as pd

from .af_ensemble import best_threshold_f1
from .af_train import _gather, _predict_chips
from .audit import af_confusion
from .chipio import ChipStore
from .config import SEED
from .features_af import af_feature_names
from .lgbm_utils import fit_binary


def temporal_holdout(train_dir: Path, work_dir: Path, holdout_year: int = 2024, tag: str = "_oh", groups=("onehot",), rounds: int = 500) -> dict:
    store = ChipStore(train_dir)
    ids = store.chip_ids("af")
    meta = store.meta().set_index("chip_id").loc[ids]
    years = pd.to_datetime(meta["acq_datetime"], utc=True).dt.year.to_numpy()
    names = af_feature_names(groups)
    x = np.load(work_dir / f"af_x{tag}.npy", mmap_mode="r")
    valid = np.load(work_dir / "af_valid.npy")
    truth = np.load(work_dir / "af_truth.npy")
    train_idx = np.flatnonzero(years < holdout_year)
    val_idx = np.flatnonzero(years == holdout_year)
    rng = np.random.default_rng(SEED)
    rows, labels = _gather(x, valid, truth, train_idx, names, rng)
    model = fit_binary(rows, labels, rounds, names)
    prob = _predict_chips(model, x, valid, val_idx)
    threshold, f1 = best_threshold_f1(prob, truth[val_idx])
    result = {"holdout_year": holdout_year, "train_chips": int(len(train_idx)), "validation_chips": int(len(val_idx)), "threshold": threshold, "f1": f1, "confusion": af_confusion(prob, truth[val_idx], threshold)}
    (work_dir / "af_temporal_holdout.json").write_text(json.dumps(result, indent=2))
    return result
