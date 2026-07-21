from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import DEFAULT_DATASET, load_confirmed_catalog
from unified3dgs.dataset_overlay import prepare_dataset_overlay


def run_streaming(command: Sequence[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify every method through the same configurable training "
            "interface exposed to menu users, using iterations=1."
        )
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/validation/_training_acceptance"),
    )
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args()

    catalog = load_confirmed_catalog()
    catalog_by_key = {str(method["key"]): method for method in catalog}
    selected = args.method or sorted(catalog_by_key)
    unknown = sorted(set(selected) - set(catalog_by_key))
    if unknown:
        parser.error("unknown confirmed method(s): " + ", ".join(unknown))

    dataset = args.dataset.expanduser()
    if not dataset.is_absolute():
        dataset = (PROJECT_ROOT / dataset).resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    print("Unified method training-interface acceptance")
    print("Acceptance budget: iterations=1")
    print(f"Original read-only dataset: {dataset}")
    print(f"Selected methods: {len(selected)}")
    if not args.run_real:
        print("Preview only. Add --run-real to execute all selected interfaces.")
        return 0
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        print("Refusing to train without an explicit CUDA_VISIBLE_DEVICES selection.")
        return 2

    acceptance_dataset = prepare_dataset_overlay(dataset, output_root)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    attempt_root = output_root / "attempts" / run_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    print(f"Project-local acceptance view: {acceptance_dataset}")
    print(f"Attempt root: {attempt_root}")

    results: List[Dict[str, object]] = []
    for index, key in enumerate(selected, 1):
        pair_root = attempt_root / key
        output = pair_root / "method_outputs"
        command = [
            sys.executable,
            "train_all.py",
            "--method",
            key,
            "--config",
            "configs/methods/catalog_method.yaml",
            "--data",
            str(acceptance_dataset),
            "--output",
            str(output),
            "--set",
            "iterations=1",
            "--set",
            "images=images_8",
            "--set",
            f"resolution={8 if key == 'hac_plus' else -1}",
            "--set",
            "eval=true",
            "--set",
            f"timeout_seconds={max(0, args.timeout_seconds)}",
            "--set",
            f"heartbeat_seconds={max(0, args.heartbeat_seconds)}",
        ]
        print()
        print("=" * 72)
        print(f"[{index}/{len(selected)}] {key}: configurable training interface")
        print("Command:", " ".join(command))
        print("=" * 72)
        status = run_streaming(command, pair_root / "train.log")
        report_path = output / "unified3dgs_training_report.json"
        verified = False
        if report_path.is_file():
            try:
                verified = bool(json.loads(report_path.read_text(encoding="utf-8"))["passed"])
            except Exception:
                verified = False
        results.append(
            {
                "method": key,
                "exit_code": status,
                "verified": verified,
                "output": str(output),
                "log": str(pair_root / "train.log"),
            }
        )

    passed = [result for result in results if result["exit_code"] == 0 and result["verified"]]
    report = {
        "all_passed": len(passed) == len(results),
        "expected_method_count": len(selected),
        "passed_method_count": len(passed),
        "iterations": 1,
        "attempt_root": str(attempt_root),
        "methods": results,
    }
    report_path = attempt_root / "acceptance_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    latest = output_root / "latest_attempt.json"
    latest.write_text(
        json.dumps({"report": str(report_path)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print()
    print("=" * 72)
    print(
        f"Training-interface acceptance: {len(passed)}/{len(results)} methods passed"
    )
    print(f"Report: {report_path}")
    if len(passed) != len(results):
        print("Failed methods:")
        for result in results:
            if result not in passed:
                print(f"- {result['method']}: {result['log']}")
        return 2
    print("All selected configurable long-training interfaces passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
