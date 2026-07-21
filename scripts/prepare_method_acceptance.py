from __future__ import annotations

import argparse
import json
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
from unified3dgs.dataset_overlay import prepare_dataset_overlay

PREPARATION_REVISION = "catalog-preflight-r20"


def run(command: List[str]) -> int:
    print()
    print("=" * 72)
    print("RUN:", " ".join(command))
    print("=" * 72)
    sys.stdout.flush()
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build every detected method extension, then run the complete "
            "zero-training preflight even when some builds fail."
        )
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--min-free-disk-gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB
    )
    parser.add_argument("--pip-timeout-seconds", type=int, default=120)
    parser.add_argument("--pip-retries", type=int, default=10)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/preparation_report.json"),
    )
    args = parser.parse_args()
    print(f"Method preparation revision: {PREPARATION_REVISION}", flush=True)

    dataset = args.dataset.expanduser()
    if not dataset.is_absolute():
        dataset = (PROJECT_ROOT / dataset).resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    try:
        acceptance_dataset = prepare_dataset_overlay(dataset, output_root)
    except Exception as exc:
        print(f"Dataset overlay compatibility preparation failed: {exc!r}", flush=True)
        return 2
    print(f"Dataset overlay compatibility view: {acceptance_dataset}", flush=True)

    configuration_status = run(
        [sys.executable, "scripts/check_method_profiles.py"]
    )
    if configuration_status != 0:
        print("Configuration audit failed. No installation or build was started.")
        return configuration_status

    method_args: List[str] = []
    for method in args.method:
        method_args.extend(["--method", method])

    compatibility_command = [
        sys.executable,
        "scripts/prepare_method_compatibility.py",
        *method_args,
    ]
    if args.run_real:
        compatibility_command.append("--run-real")
    compatibility_status = run(compatibility_command)

    build_command = [
        sys.executable,
        "scripts/install_catalog_method_extensions.py",
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--min-free-disk-gb",
        str(args.min_free_disk_gb),
        *method_args,
    ]
    if args.run_real:
        build_command.append("--run-real")
    build_status = run(build_command)

    dependency_command = [
        sys.executable,
        "-u",
        "scripts/install_catalog_method_python_dependencies.py",
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--min-free-disk-gb",
        str(args.min_free_disk_gb),
        "--pip-timeout-seconds",
        str(args.pip_timeout_seconds),
        "--pip-retries",
        str(args.pip_retries),
        *method_args,
    ]
    if args.run_real:
        dependency_command.append("--run-real")
    dependency_status = run(dependency_command)

    preflight_status = run(
        [
            sys.executable,
            "scripts/check_method_preflight.py",
            "--dataset",
            str(acceptance_dataset),
            "--output-root",
            str(output_root),
            *method_args,
        ]
    )
    summary_status = run(
        [sys.executable, "scripts/summarize_method_failures.py"]
    )

    report = args.report.expanduser()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "extension_build_exit_code": build_status,
                "compatibility_exit_code": compatibility_status,
                "configuration_audit_exit_code": configuration_status,
                "python_dependency_exit_code": dependency_status,
                "preflight_exit_code": preflight_status,
                "failure_summary_exit_code": summary_status,
                "run_real": args.run_real,
                "methods": args.method or "all confirmed methods",
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Preparation report: {report}")
    print("No training was started.")
    if compatibility_status or build_status or dependency_status or preflight_status:
        print("Preparation found blockers. Review both build and preflight reports.")
        return 2
    print("All selected methods are ready for controlled iteration=1 training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
