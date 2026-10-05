import json
from pathlib import Path

from kosmofire.audit import run_audit
from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR

if __name__ == "__main__":
    print(json.dumps(run_audit(DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR, DEFAULT_MODEL_DIR), indent=2, ensure_ascii=False))
