import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from .chipio import ChipStore
from .config import PROJECT_ROOT
from .submission import validate_submission


def link_subset(train_dir: Path, target: Path, n_af: int, n_bs: int) -> Path:
    store = ChipStore(train_dir)
    rows = []
    for kind, count in (("af", n_af), ("bs", n_bs)):
        for chip_id in store.chip_ids(kind)[:count]:
            for role, path in store.files(chip_id).paths.items():
                if role == "mask":
                    continue
                destination = target / path.parent.relative_to(train_dir) / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(path.resolve(), destination)
            rows += [(chip_id, 1)] if kind == "af" else [(chip_id, k) for k in (1, 2, 3)]
    pd.DataFrame(rows, columns=["chip_id", "class_id"]).assign(rle="").to_csv(target / "sample_submission.csv", index=False)
    return target


def time_inference(data_dir: Path, model_dir: Path, output: Path) -> tuple[float, list[str]]:
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "inference.py"), "--data-dir", str(data_dir), "--output", str(output), "--model-dir", str(model_dir)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        return elapsed, [result.stderr[-500:]]
    return elapsed, validate_submission(output, data_dir / "sample_submission.csv")


def run_benchmark(train_dir: Path, model_dir: Path, n_af: int = 90, n_bs: int = 45) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = link_subset(train_dir, Path(tmp) / "data", n_af, n_bs)
        first, problems = time_inference(data_dir, model_dir, Path(tmp) / "run1.csv")
        second, problems2 = time_inference(data_dir, model_dir, Path(tmp) / "run2.csv")
        identical = (Path(tmp) / "run1.csv").read_bytes() == (Path(tmp) / "run2.csv").read_bytes()
    return {"seconds_run1": first, "seconds_run2": second, "problems": problems + problems2, "identical_outputs": identical, "chips_af": n_af, "chips_bs": n_bs}
