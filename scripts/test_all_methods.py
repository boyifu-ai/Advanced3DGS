from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    DEFAULT_DATASET,
    DEFAULT_MIN_FREE_DISK_GB,
    DEFAULT_OUTPUT_ROOT,
)


def run(command: List[str]) -> int:
    print()
    print("=" * 72, flush=True)
    print("RUN:", " ".join(command), flush=True)
    print("=" * 72, flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all confirmed methods, require a completely passing "
            "preflight, then run iteration=1 save-completeness tests."
        )
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--min-free-disk-gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB
    )
    parser.add_argument("--min-free-vram-gb", type=float, default=4.0)
    parser.add_argument("--run-real", action="store_true")
    args = parser.parse_args()

    if not args.run_real:
        print("Preview only. No installation, patching, extension build, or training started.")
        print(
            "Use --run-real with an explicit CUDA_VISIBLE_DEVICES value to prepare "
            "and test every selected method."
        )
        return 0
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        print("Refusing to continue without CUDA_VISIBLE_DEVICES on the shared server.")
        return 2

    method_args: List[str] = []
    for method in args.method:
        method_args.extend(["--method", method])
    prepare_status = run(
        [
            sys.executable,
            "-u",
            "scripts/prepare_method_acceptance.py",
            "--run-real",
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--min-free-disk-gb",
            str(args.min_free_disk_gb),
            "--dataset",
            str(args.dataset),
            "--output-root",
            str(args.output_root),
            *method_args,
        ]
    )
    if prepare_status != 0:
        print()
        print("Preparation or complete preflight failed. No training was started.", flush=True)
        print(
            "Review outputs/validation/_method_acceptance/preflight_report.json and the "
            "preceding preparation reports.",
            flush=True,
        )
        return prepare_status

    return run(
        [
            sys.executable,
            "-u",
            "scripts/run_method_save_check.py",
            "--run-real",
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--min-free-disk-gb",
            str(args.min_free_disk_gb),
            "--min-free-vram-gb",
            str(args.min_free_vram_gb),
            "--dataset",
            str(args.dataset),
            "--output-root",
            str(args.output_root),
            *method_args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
