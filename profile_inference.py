import json
from pathlib import Path

from kosmofire.config import DEFAULT_MODEL_DIR, PROJECT_ROOT
from kosmofire.profiling import profile_stages

if __name__ == "__main__":
    print(json.dumps(profile_stages(PROJECT_ROOT / "data" / "raw" / "test", DEFAULT_MODEL_DIR), indent=2))
