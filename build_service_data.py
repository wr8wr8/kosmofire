import argparse
from pathlib import Path

from kosmofire.config import DEFAULT_MODEL_DIR, DEFAULT_TRAIN_DIR, DEFAULT_WORK_DIR, PROJECT_ROOT
from kosmofire.service_build import build_service_gpkg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "service" / "fire_service.gpkg")
    args = parser.parse_args()
    print(build_service_gpkg(args.train_dir, args.work_dir, args.model_dir, args.output))


if __name__ == "__main__":
    main()
