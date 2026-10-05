import argparse
import json
import sys
import time
from pathlib import Path

from kosmofire.bundle import verify_bundle
from kosmofire.chipio import ChipStore
from kosmofire.config import DEFAULT_MODEL_DIR
from kosmofire.inference_pipeline import predict_af, predict_bs
from kosmofire.submission import build_rows, read_template, validate_submission, write_submission


def find_template(data_dir: Path) -> Path:
    matches = sorted(data_dir.rglob("sample_submission.csv"))
    if not matches:
        raise FileNotFoundError(f"sample_submission.csv not found under {data_dir}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--on-error", choices=("fail", "empty"), default="fail")
    parser.add_argument("--skip-bundle-check", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    if not args.skip_bundle_check:
        problems = verify_bundle(args.model_dir)
        if problems:
            print("model bundle check failed:\n" + "\n".join(problems), file=sys.stderr)
            return 3

    template_path = find_template(args.data_dir)
    template = read_template(template_path)
    store = ChipStore(args.data_dir)

    af_ids = sorted({c for c in template["chip_id"] if c.startswith("AF_")})
    bs_ids = sorted({c for c in template["chip_id"] if c.startswith("BS_")})
    missing = [c for c in af_ids + bs_ids if not store.has(c)]
    if missing:
        print(f"no input files for chips: {missing[:10]}", file=sys.stderr)
        return 4

    af_masks, af_failures = predict_af(store, af_ids, args.model_dir, args.workers, args.on_error) if af_ids else ({}, {})
    bs_masks, bs_failures = predict_bs(store, bs_ids, args.model_dir, args.workers, args.on_error) if bs_ids else ({}, {})
    failures = {**af_failures, **bs_failures}

    manifest = {
        "chips_total": len(af_ids) + len(bs_ids),
        "chips_processed": len(af_ids) + len(bs_ids) - len(failures),
        "chips_failed": len(failures),
        "failures": failures,
        "on_error": args.on_error,
        "seconds": round(time.perf_counter() - started, 2),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    if failures and args.on_error == "fail":
        print(f"{len(failures)} chips failed, see {manifest_path}", file=sys.stderr)
        return 2

    write_submission(build_rows(template, af_masks, bs_masks), args.output)
    problems = validate_submission(args.output, template_path)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    if failures:
        print(f"warning: {len(failures)} chips written as empty masks, see {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
