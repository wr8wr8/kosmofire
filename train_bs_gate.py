import argparse
import json
import logging
from pathlib import Path

from kosmofire.bs_gate import train_bs_gate
from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--oof-tag", default="_v2")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(train_bs_gate(args.train_dir, args.work_dir, args.model_dir, args.oof_tag), indent=2))


if __name__ == "__main__":
    main()
