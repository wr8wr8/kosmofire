import json
from pathlib import Path

from kosmofire.bundle import build_bundle
from kosmofire.config import DEFAULT_MODEL_DIR

if __name__ == "__main__":
    print(json.dumps(build_bundle(DEFAULT_MODEL_DIR), indent=2))
