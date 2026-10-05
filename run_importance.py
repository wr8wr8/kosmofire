import json

from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR, PROJECT_ROOT
from kosmofire.importance import run_importance

if __name__ == "__main__":
    result = run_importance(DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR, DEFAULT_MODEL_DIR, PROJECT_ROOT / "reports" / "feature_importance.json")
    for name, block in [("AF", result["af"])] + [(f"BS {k}", v) for k, v in result["bs"].items()]:
        print(name, [(r["feature"], round(r["share"], 3)) for r in block["shap_top"][:8]])
        print("  low-importance count:", len(block["low_importance"]))
