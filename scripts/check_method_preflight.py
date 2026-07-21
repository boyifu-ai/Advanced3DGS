from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_ROOT,
    preflight_all,
    select_methods,
)
from scripts.patch_third_party_readers import REPOS as READER_REPOS, patch_file


def repair_selected_readers(selected, project_root: Path = PROJECT_ROOT):
    records = []
    errors = []
    for method, _profile in selected:
        key = str(method["key"])
        relative = READER_REPOS.get(key)
        if relative is None:
            records.append({"method": key, "status": "not_applicable"})
            continue
        reader = project_root / relative / "scene" / "dataset_readers.py"
        try:
            status = patch_file(reader)
        except Exception as exc:
            message = f"reader repair failed for {key}: {exc}"
            records.append(
                {
                    "method": key,
                    "reader": str(reader),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            errors.append(message)
        else:
            records.append(
                {
                    "method": key,
                    "reader": str(reader),
                    "status": status,
                }
            )
    return records, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an exhaustive zero-training preflight for methods."
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--repair-readers",
        action="store_true",
        help=(
            "Explicitly repair known third-party dataset-reader compatibility "
            "issues before running the full preflight."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/preflight_report.json"),
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser()
    if not dataset.is_absolute():
        dataset = (PROJECT_ROOT / dataset).resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    report = args.report.expanduser()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()

    selected = select_methods(args.method)
    reader_repairs = []
    repair_errors = []
    if args.repair_readers:
        print("Repairing selected third-party dataset readers before preflight...")
        reader_repairs, repair_errors = repair_selected_readers(selected)
        for record in reader_repairs:
            print(f"- {record['method']}: {record['status']}")
        print()
    global_errors, results = preflight_all(selected, dataset, output_root)
    global_errors = repair_errors + global_errors
    passed = [result for result in results if result.passed]
    failed = [result for result in results if not result.passed]
    payload = {
        "dataset": str(dataset),
        "output_root": str(output_root),
        "reader_repairs": reader_repairs,
        "global_errors": global_errors,
        "summary": {
            "passed": [result.key for result in passed],
            "failed": [result.key for result in failed],
        },
        "methods": [result.as_dict() for result in results],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Method acceptance preflight")
    print(f"Methods checked: {len(results)}")
    print(f"Report: {report}")
    if global_errors:
        print("\nGLOBAL ERRORS:")
        for error in global_errors:
            print(f"- {error}")
    for result in results:
        state = "PASS" if result.passed else "FAIL"
        print(f"\n[{state}] {result.title} [{result.key}]")
        print(f"  repo:  {result.repo}")
        print(f"  entry: {result.entry or 'unresolved'}")
        for warning in result.warnings:
            print(f"  WARNING: {warning}")
        for error in result.errors:
            print(f"  ERROR: {error}")

    print()
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    if global_errors or failed:
        if passed and not global_errors:
            method_args = " ".join(f"--method {result.key}" for result in passed)
            print("\nReady subset can be tested without the failed methods:")
            print(
                "CUDA_VISIBLE_DEVICES=<gpu> python scripts/run_method_save_check.py "
                f"--run-real {method_args}"
            )
        print("No training should start until every selected method passes this preflight.")
        return 2
    print("All selected methods passed. It is safe to start the iteration=1 acceptance runner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
