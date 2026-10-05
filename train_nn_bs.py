import argparse
import logging
from pathlib import Path

from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR
from kosmofire.nn_train import run_nn_cv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(run_nn_cv(args.train_dir, args.work_dir, args.model_dir, args.epochs, args.steps, args.batch, args.base, args.depth, device=args.device))


if __name__ == "__main__":
    main()
