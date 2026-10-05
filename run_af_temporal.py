import json

from kosmofire.af_temporal import temporal_holdout
from kosmofire.config import DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR

if __name__ == "__main__":
    print(json.dumps(temporal_holdout(DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR), indent=2))
