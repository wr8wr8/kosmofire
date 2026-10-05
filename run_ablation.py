import json
import sys
from pathlib import Path

from kosmofire.ablation import compare_af, compare_bs
from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR


def main() -> None:
    af_variants = {
        "one_hot_baseline": "af_oof_oh.npy",
        "categorical_landcover": "af_oof_cat.npy",
        "dozier_and_global_context": "af_oof_v2.npy",
        "catboost": "af_oof_cat_oh.npy",
    }
    af_variants = {k: v for k, v in af_variants.items() if (DEFAULT_WORK_DIR / v).exists()}
    result = {"af": compare_af(DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR, af_variants, "one_hot_baseline")}
    bs_variants = {"final_v2": "_v2", "v2_one_hot": "_v2oh", "no_p1_features": "_nop1", "no_monotone": "_nomono"}
    bs_variants = {k: v for k, v in bs_variants.items() if (DEFAULT_MODEL_DIR / f"bs_meta{v}.json").exists()}
    if "final_v2" in bs_variants:
        result["bs"] = compare_bs(DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR, DEFAULT_MODEL_DIR, bs_variants, "final_v2")
    out = DEFAULT_WORK_DIR / "ablation_report.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
