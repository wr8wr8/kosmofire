import json
from pathlib import Path

import numpy as np

from .af_ensemble import best_threshold_f1
from .bootstrap import grouped_bootstrap_difference
from .bs_train import decode_oof, load_oof
from .chipio import ChipStore, read_mask
from .validation import af_groups


def _f1(items: list) -> float:
    tp = sum(i[0] for i in items)
    fp = sum(i[1] for i in items)
    fn = sum(i[2] for i in items)
    return 1.0 if tp + fp + fn == 0 else 2 * tp / max(2 * tp + fp + fn, 1)


def af_group_counts(prob: np.ndarray, truth: np.ndarray, groups: np.ndarray, threshold: float) -> dict:
    counts = {}
    pred = prob >= threshold
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        p, t = pred[idx], truth[idx] == 1
        counts[int(g)] = (int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()))
    return counts


def compare_af(train_dir: Path, work_dir: Path, variants: dict[str, str], reference: str, iterations: int = 400) -> dict:
    store = ChipStore(train_dir)
    ids = store.chip_ids("af")
    meta = store.meta()
    groups = af_groups(meta[meta["kind"] == "af"], ids)
    truth = np.load(work_dir / "af_truth.npy")
    ref_prob = np.load(work_dir / variants[reference])
    ref_threshold, ref_f1 = best_threshold_f1(ref_prob, truth)
    ref_counts = af_group_counts(ref_prob, truth, groups, ref_threshold)
    out = {reference: {"f1": ref_f1, "threshold": ref_threshold}}
    for name, filename in variants.items():
        if name == reference:
            continue
        prob = np.load(work_dir / filename)
        threshold, f1 = best_threshold_f1(prob, truth)
        counts = af_group_counts(prob, truth, groups, threshold)
        ci = grouped_bootstrap_difference(counts, ref_counts, _f1, iterations=iterations, seed=1)
        out[name] = {"f1": f1, "threshold": threshold, "diff_vs_reference": ci.mean_difference, "ci95": [ci.low, ci.high], "significant": ci.excludes_zero}
    return out


def bs_chip_counts(work_dir: Path, tag: str, truths: list[np.ndarray], threshold: float, min_component: int) -> dict:
    counts = {}
    for i, truth in enumerate(truths):
        pred = decode_oof(load_oof(work_dir, tag, i), threshold, 0.0, min_component)
        row = []
        for mask_t, mask_p in ((truth >= 1, pred >= 1), (truth == 1, pred == 1), (truth == 2, pred == 2), (truth == 3, pred == 3)):
            row += [int((mask_t & mask_p).sum()), int((~mask_t & mask_p).sum()), int((mask_t & ~mask_p).sum())]
        counts[i] = tuple(row)
    return counts


def bs_partial_score(items: list) -> float:
    total = np.array(items, dtype=np.float64).sum(axis=0).reshape(4, 3)
    ious = [t[0] / max(t.sum(), 1) for t in total]
    return 0.35 * ious[0] + 0.30 * float(np.mean(ious[1:]))


def compare_bs(train_dir: Path, work_dir: Path, models_dir: Path, variants: dict[str, str], reference: str, iterations: int = 400) -> dict:
    store = ChipStore(train_dir)
    ids = store.chip_ids("bs")
    truths = [read_mask(store.files(c).paths["mask"]).astype(np.uint8) for c in ids]
    tables = {}
    for name, tag in variants.items():
        meta = json.loads((models_dir / f"bs_meta{tag}.json").read_text())
        tables[name] = bs_chip_counts(work_dir, tag, truths, meta["burn_threshold"], int(meta.get("min_component", 0)))
    out = {}
    for name, table in tables.items():
        score = bs_partial_score(list(table.values()))
        entry = {"partial_score": score}
        if name != reference:
            ci = grouped_bootstrap_difference(table, tables[reference], bs_partial_score, iterations=iterations, seed=1)
            entry.update({"diff_vs_reference": ci.mean_difference, "ci95": [ci.low, ci.high], "significant": ci.excludes_zero})
        out[name] = entry
    return out
