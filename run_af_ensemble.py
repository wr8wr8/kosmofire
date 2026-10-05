import argparse
import json
from pathlib import Path

from kosmofire.af_ensemble import run_af_ensemble
from kosmofire.config import DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--tag", default="_oh")
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(run_af_ensemble(args.train_dir, args.work_dir, args.tag, iterations=args.iterations), indent=2))


if __name__ == "__main__":
    main()
