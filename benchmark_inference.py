import argparse
import json
from pathlib import Path

from kosmofire.benchmark import run_benchmark
from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--af", type=int, default=90)
    parser.add_argument("--bs", type=int, default=45)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.train_dir, args.model_dir, args.af, args.bs), indent=2))


if __name__ == "__main__":
    main()
