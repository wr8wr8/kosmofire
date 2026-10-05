import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kosmofire.config import DEFAULT_MODEL_DIR, PROJECT_ROOT
from kosmofire.submission import validate_submission

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"
TIERS = ((30.0, 8), (300.0, 6), (720.0, 4), (1800.0, 2), (3600.0, 0))
GUARANTEED_BUDGET_S = float(os.environ.get("KOSMOFIRE_SPEED_BUDGET_S", "300"))


def _tier(seconds: float) -> int:
    for limit, points in TIERS:
        if seconds < limit:
            return points
    return 0


@pytest.mark.skipif(not (TEST_DIR / "sample_submission.csv").exists(), reason="downloaded test set is not present")
def test_full_real_test_set_inference_time_and_score_tier(tmp_path, record_property):
    output = tmp_path / "submission.csv"
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "inference.py"), "--data-dir", str(TEST_DIR), "--output", str(output), "--model-dir", str(DEFAULT_MODEL_DIR)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    elapsed = time.perf_counter() - started
    record_property("seconds", round(elapsed, 1))
    record_property("speed_points", _tier(elapsed))
    print(f"\nfull real test set: {elapsed:.1f} s, workers={os.cpu_count()} cpus, speed tier {_tier(elapsed)}/8 points")
    assert result.returncode == 0, result.stderr
    assert validate_submission(output, TEST_DIR / "sample_submission.csv") == []
    assert elapsed < GUARANTEED_BUDGET_S, f"{elapsed:.1f} s exceeds the {GUARANTEED_BUDGET_S:.0f} s guaranteed budget"
