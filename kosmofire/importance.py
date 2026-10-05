import json
from pathlib import Path

import numpy as np

from .bs_data import load_bs_features
from .chipio import ChipStore, read_mask
from .config import SEED
from .features_af import af_feature_names
from .features_bs import bs_feature_names
from .lgbm_utils import load_booster
from .unmixing import load_endmembers


def shap_ranking(model, rows: np.ndarray, names: list[str], top: int = 20) -> list[dict]:
    contrib = model.predict(rows, pred_contrib=True, num_threads=2)[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)
    gain = model.feature_importance(importance_type="gain")
    order = np.argsort(-mean_abs)[:top]
    total = float(mean_abs.sum()) or 1.0
    return [{"feature": names[i], "mean_abs_shap": float(mean_abs[i]), "share": float(mean_abs[i] / total), "gain": float(gain[i])} for i in order]


def zero_importance_features(model, rows: np.ndarray, names: list[str], share_threshold: float = 0.001) -> list[str]:
    contrib = model.predict(rows, pred_contrib=True, num_threads=2)[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    return [names[i] for i in range(len(names)) if mean_abs[i] / total < share_threshold]


def run_importance(train_dir: Path, work_dir: Path, model_dir: Path, output: Path, sample: int = 40000) -> dict:
    rng = np.random.default_rng(SEED)
    af_meta = json.loads((model_dir / "af_meta.json").read_text())
    af_names = af_feature_names(tuple(af_meta["groups"]))
    x = np.load(work_dir / "af_x_oh.npy", mmap_mode="r")
    valid = np.load(work_dir / "af_valid.npy")
    truth = np.load(work_dir / "af_truth.npy")
    rows = []
    for i in rng.choice(x.shape[0], 120, replace=False):
        chip = np.asarray(x[i])
        pos = np.flatnonzero((truth[i] == 1).ravel())
        neg = np.flatnonzero(((truth[i] == 0) & valid[i]).ravel())
        pick = np.concatenate([pos, rng.choice(neg, min(len(neg), 300), replace=False)])
        rows.append(chip.reshape(chip.shape[0], -1)[:, pick].T)
    af_rows = np.concatenate(rows)[:sample]
    af_model = load_booster(model_dir / "af_lgbm.txt")

    bs_meta = json.loads((model_dir / "bs_meta.json").read_text())
    groups = tuple(bs_meta["groups"])
    bs_names = bs_feature_names(groups)
    endmembers = load_endmembers(model_dir / "bs_endmembers.json")
    store = ChipStore(train_dir)
    ids = store.chip_ids("bs")
    bs_rows = []
    for i in rng.choice(len(ids), 20, replace=False):
        paths = store.files(ids[i]).paths
        features, _ = load_bs_features(paths, groups, endmembers)
        truth_bs = read_mask(paths["mask"])
        flat = features.reshape(features.shape[0], -1)
        burn = np.flatnonzero((truth_bs >= 1).ravel())
        rest = np.flatnonzero((truth_bs == 0).ravel())
        pick = np.concatenate([rng.choice(burn, min(len(burn), 1500), replace=False), rng.choice(rest, 1500, replace=False)])
        bs_rows.append(flat[:, pick].T)
    bs_rows = np.concatenate(bs_rows)
    result = {"af": {"shap_top": shap_ranking(af_model, af_rows, af_names), "low_importance": zero_importance_features(af_model, af_rows, af_names)}, "bs": {}}
    for k in (1, 2, 3):
        model = load_booster(model_dir / f"bs_ge{k}.txt")
        result["bs"][f"ge{k}"] = {"shap_top": shap_ranking(model, bs_rows, bs_names), "low_importance": zero_importance_features(model, bs_rows, bs_names)}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
