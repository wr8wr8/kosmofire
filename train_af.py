import argparse
import logging
from pathlib import Path

from kosmofire.af_train import train_af
from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--groups", nargs="*", default=["dozier", "global"])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = train_af(args.train_dir, args.work_dir, args.model_dir, args.rounds, args.folds, tuple(args.groups), args.tag)
    print({k: v for k, v in report.items() if k != "features"})


if __name__ == "__main__":
    main()
