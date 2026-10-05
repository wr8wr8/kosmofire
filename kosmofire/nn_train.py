import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from .chipio import ChipStore, read_mask, read_raster
from .config import SEED
from .metrics import ScoreAccumulator
from .nn_bs import SiameseUNet, ordinal_loss, prepare_inputs, seed_everything
from .validation import bs_groups

logger = logging.getLogger("train_nn")

CROP = 128
CLASS_WEIGHTS = (0.6, 1.4, 1.2, 1.0)


def load_chip(paths: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre, post, static = prepare_inputs(
        read_raster(paths["s2_pre"]), read_raster(paths["s2_post"]), read_raster(paths["s1_pre"]), read_raster(paths["s1_post"]), read_raster(paths["aux"])
    )
    truth = read_mask(paths["mask"]).astype(np.int64)
    return pre.astype(np.float16), post.astype(np.float16), static.astype(np.float16), truth.astype(np.uint8)


def _batch(chips, indices, rng, size, device):
    pres, posts, statics, targets = [], [], [], []
    for i in indices:
        pre, post, static, truth = chips[i]
        y = rng.integers(0, pre.shape[1] - size + 1)
        x = rng.integers(0, pre.shape[2] - size + 1)
        window = (slice(None), slice(y, y + size), slice(x, x + size))
        p, q, s = pre[window], post[window], static[window]
        t = truth[y : y + size, x : x + size]
        if rng.random() < 0.5:
            p, q, s, t = p[:, :, ::-1], q[:, :, ::-1], s[:, :, ::-1], t[:, ::-1]
        if rng.random() < 0.5:
            p, q, s, t = p[:, ::-1], q[:, ::-1], s[:, ::-1], t[::-1]
        pres.append(np.ascontiguousarray(p))
        posts.append(np.ascontiguousarray(q))
        statics.append(np.ascontiguousarray(s))
        targets.append(np.ascontiguousarray(t))
    to = lambda a, dt: torch.from_numpy(np.stack(a)).to(device=device, dtype=dt)
    return to(pres, torch.float32), to(posts, torch.float32), to(statics, torch.float32), to(targets, torch.long)


def train_model(chips, train_idx, epochs: int, steps_per_epoch: int, batch: int, base: int, depth: int, device: str, seed: int) -> SiameseUNet:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = SiameseUNet(base=base, depth=depth).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    total = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=3e-3, total_steps=total)
    weights = torch.tensor(CLASS_WEIGHTS, device=device)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for _ in range(steps_per_epoch):
            picked = rng.choice(train_idx, size=batch)
            pre, post, static, target = _batch(chips, picked, rng, CROP, device)
            loss = ordinal_loss(model(pre, post, static), target, 0.5, weights)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            running += float(loss)
        logger.info("epoch %d loss %.4f", epoch, running / steps_per_epoch)
    return model.to("cpu")


def predict_chip(model: SiameseUNet, chip) -> np.ndarray:
    model.eval()
    pre, post, static, _ = chip
    with torch.no_grad():
        logits = model(
            torch.from_numpy(pre.astype(np.float32))[None],
            torch.from_numpy(post.astype(np.float32))[None],
            torch.from_numpy(static.astype(np.float32))[None],
        )
    return torch.softmax(logits, dim=1)[0].numpy()


def run_nn_cv(train_dir: Path, work_dir: Path, model_dir: Path, epochs: int = 12, steps: int = 60, batch: int = 8, base: int = 16, depth: int = 4, folds: int = 5, device: str | None = None) -> dict:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    store = ChipStore(train_dir)
    chip_ids = store.chip_ids("bs")
    meta = store.meta()
    groups = bs_groups(meta[meta["kind"] == "bs"], chip_ids)
    chips = [load_chip(store.files(cid).paths) for cid in chip_ids]

    (work_dir / "nn_oof").mkdir(parents=True, exist_ok=True)
    started = time.time()
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(np.arange(len(chips)), groups=groups)):
        model = train_model(chips, tr, epochs, steps, batch, base, depth, device, SEED + fold)
        for i in va:
            np.save(work_dir / "nn_oof" / f"{i:04d}.npy", predict_chip(model, chips[i]).astype(np.float16))
        logger.info("fold %d done at %.0fs", fold, time.time() - started)

    acc = ScoreAccumulator()
    for i, chip in enumerate(chips):
        probs = np.load(work_dir / "nn_oof" / f"{i:04d}.npy").astype(np.float32)
        acc.add_bs(chip[3], probs.argmax(axis=0).astype(np.uint8))
    result = acc.result()

    final = train_model(chips, np.arange(len(chips)), epochs, steps, batch, base, depth, device, SEED)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": final.state_dict(), "base": base, "depth": depth}, model_dir / "bs_siamese_unet.pt")
    report = {"oof_iou_burn": result.iou_burn, "oof_miou_sev": result.miou_sev, "oof_iou_by_class": list(result.iou_by_class), "epochs": epochs, "steps": steps, "base": base, "depth": depth, "device": device}
    (model_dir / "bs_nn_meta.json").write_text(json.dumps(report, indent=2))
    return report
