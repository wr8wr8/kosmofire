import subprocess
import sys
import time
from pathlib import Path

import pytest

from kosmofire.submission import validate_submission
from tests.dataset_factory import build_test_dataset, build_test_models

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("e2e")
    data = build_test_dataset(base / "data")
    models = build_test_models(base / "models")
    return base, data, models


def _run(data, models, output):
    return subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), "--data-dir", str(data), "--output", str(output), "--model-dir", str(models), "--skip-bundle-check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_black_box_inference_produces_valid_submission(workspace):
    base, data, models = workspace
    out = base / "submission.csv"
    result = _run(data, models, out)
    assert result.returncode == 0, result.stderr
    assert validate_submission(out, data / "sample_submission.csv") == []


def test_cloudy_chip_does_not_crash_pipeline(workspace):
    base, data, models = workspace
    out = base / "submission_cloud.csv"
    result = _run(data, models, out)
    assert result.returncode == 0
    assert "BS_te_000002" in out.read_text()


def test_two_runs_are_byte_identical(workspace):
    base, data, models = workspace
    a, b = base / "run_a.csv", base / "run_b.csv"
    assert _run(data, models, a).returncode == 0
    assert _run(data, models, b).returncode == 0
    assert a.read_bytes() == b.read_bytes()


def test_inference_smoke_time_on_five_synthetic_chips(workspace):
    base, data, models = workspace
    started = time.perf_counter()
    assert _run(data, models, base / "timed.csv").returncode == 0
    assert time.perf_counter() - started < 30.0
