import subprocess
import sys
from pathlib import Path

import pandas as pd

from kosmofire.submission import validate_submission
from tests.dataset_factory import build_test_dataset, build_test_models

ROOT = Path(__file__).resolve().parents[1]


def _run(data, models, output):
    return subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), "--data-dir", str(data), "--output", str(output), "--model-dir", str(models), "--workers", "2", "--skip-bundle-check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_chip_fully_rejected_by_gate_and_prefilter_still_yields_empty_rle_rows(tmp_path):
    data = build_test_dataset(tmp_path / "data")
    models = build_test_models(tmp_path / "models", gate_threshold=1.5, af_prefilter={"i4_minus_bg_7": 1e6, "delta_mir_tir": 1e6})
    out = tmp_path / "submission.csv"
    result = _run(data, models, out)
    assert result.returncode == 0, result.stderr
    assert validate_submission(out, data / "sample_submission.csv") == []
    frame = pd.read_csv(out, keep_default_na=False, dtype=str)
    assert len(frame) == 3 + 2 * 3
    assert (frame["rle"] == "").all()
    assert out.read_text().splitlines()[1] == "AF_te_000001,1,"


def test_worker_count_does_not_change_output(tmp_path):
    data = build_test_dataset(tmp_path / "data")
    models = build_test_models(tmp_path / "models", gate_threshold=0.3)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert _run(data, models, a).returncode == 0
    single = subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), "--data-dir", str(data), "--output", str(b), "--model-dir", str(models), "--workers", "1", "--skip-bundle-check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert single.returncode == 0, single.stderr
    assert a.read_bytes() == b.read_bytes()
