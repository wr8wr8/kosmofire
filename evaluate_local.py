import argparse
import json
from pathlib import Path

from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR
from kosmofire.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--af-tag", default="_oh")
    parser.add_argument("--bs-tag", default="_v3")
    args = parser.parse_args()
    report = evaluate(args.train_dir, args.work_dir, args.model_dir, args.af_tag, args.bs_tag)
    out = Path(__file__).parent / "reports" / f"evaluation{args.bs_tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
