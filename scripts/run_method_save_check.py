from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    DEFAULT_DATASET,
    DEFAULT_MIN_FREE_DISK_GB,
    DEFAULT_OUTPUT_ROOT,
    build_method_env,
    build_acceptance_command,
    preflight_all,
    result_output_roots,
    saved_result_files,
    select_methods,
    unexpected_iteration_artifacts,
)
from unified3dgs.dataset_overlay import prepare_dataset_overlay

EXPECTED_ITERATION = 1
RUNNER_REVISION = "catalog-acceptance-runner-r18"


def verified_completion(
    status: int,
    stopped_after_save: bool,
    saved: List[Path],
    unexpected: List[Path],
    profile: Dict[str, object],
) -> Tuple[bool, str]:
    exact_save_verified = bool(saved) and not unexpected
    if stopped_after_save and exact_save_verified:
        return True, "verified_save_stop"
    if status == 0 and exact_save_verified:
        return True, "process_exit"
    if profile.get("stop_after_verified_save") is True and exact_save_verified:
        return True, "verified_save_after_process_exit"
    return False, "process_exit"


def log_tail(path: Path, lines: int = 20) -> List[str]:
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError as exc:
        return [f"could not read log tail: {exc!r}"]


def _signal_process_group(process: subprocess.Popen, sig: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
    except ProcessLookupError:
        pass


def stable_verified_save_probe(
    output: Path,
    profile: Dict[str, object],
    newer_than: float,
    settle_seconds: float = 1.0,
) -> Callable[[], bool]:
    extra_globs = profile.get("result_globs", [])
    if not isinstance(extra_globs, list):
        extra_globs = []
    previous: Optional[Tuple[Tuple[str, int, int], ...]] = None
    unchanged_since = 0.0

    def probe() -> bool:
        nonlocal previous, unchanged_since
        try:
            files: List[Path] = []
            for root in result_output_roots(output, profile):
                files.extend(
                    saved_result_files(
                        root,
                        expected_iteration=EXPECTED_ITERATION,
                        newer_than=newer_than,
                        extra_globs=extra_globs,
                    )
                )
            snapshot = tuple(
                sorted(
                    (str(path), path.stat().st_size, path.stat().st_mtime_ns)
                    for path in files
                    if path.is_file()
                )
            )
        except OSError:
            return False
        now = time.monotonic()
        if not snapshot:
            previous = None
            unchanged_since = 0.0
            return False
        if snapshot != previous:
            previous = snapshot
            unchanged_since = now
            return False
        return now - unchanged_since >= settle_seconds

    return probe


def run_streaming(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    log: Path,
    stop_when: Optional[Callable[[], bool]] = None,
) -> Tuple[int, bool]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name == "posix",
        )
        assert process.stdout is not None
        stopped_after_save = False
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
            if stop_when is not None and stop_when():
                message = (
                    "\nUnified 3DGS: verified iteration=1 result is stable; "
                    "ending this method-specific completeness run.\n"
                )
                print(message, end="", flush=True)
                handle.write(message)
                handle.flush()
                stopped_after_save = True
                _signal_process_group(process, signal.SIGINT)
                break
        if stopped_after_save:
            try:
                remainder, _ = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                remainder, _ = process.communicate()
            if remainder:
                print(remainder, end="")
                handle.write(remainder)
                handle.flush()
        return process.wait(), stopped_after_save


def runtime_resource_errors(output_root: Path, min_disk_gb: float, min_vram_gb: float) -> List[str]:
    errors: List[str] = []
    free_disk = shutil.disk_usage(output_root).free
    if free_disk < min_disk_gb * 1024**3:
        errors.append(
            f"free disk is below {min_disk_gb:g} GiB: {free_disk / 1024**3:.2f} GiB"
        )
    try:
        import torch

        free_vram, _ = torch.cuda.mem_get_info()
        if free_vram < min_vram_gb * 1024**3:
            errors.append(
                f"free GPU memory is below {min_vram_gb:g} GiB: "
                f"{free_vram / 1024**3:.2f} GiB"
            )
    except Exception as exc:
        errors.append(f"cannot inspect selected GPU memory immediately before training: {exc!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run iteration=1 save-completeness tests for all selected methods."
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Hard wall-clock limit for each method. Default: 1800 seconds.",
    )
    parser.add_argument(
        "--run-real",
        action="store_true",
        help="Required guard. Without it, only the full preflight is run.",
    )
    parser.add_argument(
        "--min-free-disk-gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB
    )
    parser.add_argument("--min-free-vram-gb", type=float, default=4.0)
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional attempt identifier. Defaults to a timestamp plus process id.",
    )
    args = parser.parse_args()
    print(f"Method acceptance runner revision: {RUNNER_REVISION}", flush=True)

    if args.run_real and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        print("Refusing to train without CUDA_VISIBLE_DEVICES on the shared server.")
        print("Example: CUDA_VISIBLE_DEVICES=5 python scripts/run_method_save_check.py --run-real")
        return 2

    dataset = args.dataset.expanduser()
    if not dataset.is_absolute():
        dataset = (PROJECT_ROOT / dataset).resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    try:
        acceptance_dataset = prepare_dataset_overlay(dataset, output_root)
    except Exception as exc:
        print(f"Dataset overlay compatibility preparation failed: {exc!r}")
        return 2
    run_id = args.run_id.strip() or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    attempt_root = output_root / "attempts" / run_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    print(f"Original read-only dataset: {dataset}")
    print(f"Project-local dataset overlay: {acceptance_dataset}")
    print(f"Attempt root: {attempt_root}")

    selected = select_methods(args.method)
    global_errors, preflight = preflight_all(selected, acceptance_dataset, output_root)
    preflight_report = attempt_root / "preflight_report.json"
    preflight_report.parent.mkdir(parents=True, exist_ok=True)
    preflight_report.write_text(
        json.dumps(
            {
                "original_dataset": str(dataset),
                "acceptance_dataset": str(acceptance_dataset),
                "output_root": str(output_root),
                "global_errors": global_errors,
                "methods": [result.as_dict() for result in preflight],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    failed_preflight = [result for result in preflight if not result.passed]
    if global_errors or failed_preflight:
        print("Preflight failed. No training was started.")
        for error in global_errors:
            print(f"- GLOBAL: {error}")
        for result in failed_preflight:
            for error in result.errors:
                print(f"- {result.key}: {error}")
        return 2
    if not args.run_real:
        print("All selected methods passed preflight. No training was started.")
        print("Add --run-real to execute the iteration=1 save-completeness tests.")
        return 0

    summary: List[Dict[str, object]] = []
    profiles = {str(profile["key"]): profile for _, profile in selected}
    for result in preflight:
        pair_root = attempt_root / result.key
        output = pair_root / "method_outputs"
        if pair_root.exists() and any(pair_root.iterdir()):
            summary.append(
                {
                    "method": result.key,
                    "status": "blocked_existing_output",
                    "output": str(pair_root),
                    "errors": [
                        "method acceptance directory is not empty; old and new runs are never mixed"
                    ],
                }
            )
            print(f"\nBLOCKED {result.key}: method acceptance directory is not empty: {pair_root}")
            print("Clean or archive that method directory before retrying; old and new runs are never mixed.")
            continue
        pair_root.mkdir(parents=True, exist_ok=True)
        resource_errors = runtime_resource_errors(
            output_root, args.min_free_disk_gb, args.min_free_vram_gb
        )
        if resource_errors:
            summary.append(
                {
                    "method": result.key,
                    "status": "blocked_resources",
                    "errors": resource_errors,
                }
            )
            print(f"\nBLOCKED {result.key}: runtime resource guard failed.")
            for error in resource_errors:
                print(f"- {error}")
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = build_acceptance_command(
            result, acceptance_dataset, output, profiles[result.key]
        )
        env = build_method_env(result.key, result.repo)
        env["UNIFIED3DGS_OUTPUT_PATH"] = str(output.resolve())
        profile = profiles[result.key]
        manifest = {
            "method": result.key,
            "entry": str(result.entry),
            "cwd": str(result.entry.parent),
            "original_dataset": str(dataset),
            "acceptance_dataset": str(acceptance_dataset),
            "output": str(output),
            "expected_iteration": EXPECTED_ITERATION,
            "runner_revision": RUNNER_REVISION,
            "command": command,
            "environment": {
                name: env.get(name, "")
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_HOME",
                    "TORCH_CUDA_ARCH_LIST",
                    "TORCH_HOME",
                    "WANDB_MODE",
                    "PYTORCH_ALLOC_CONF",
                    "UNIFIED3DGS_READONLY_DATASET_ROOT",
                    "UNIFIED3DGS_NUMPY_LEGACY_ALIASES",
                    "UNIFIED3DGS_PY38_FUNCTOOLS_CACHE",
                    "PYTHONPATH",
                )
            },
            "stop_after_verified_save": profile.get("stop_after_verified_save", False),
            "stop_after_verified_save_reason": profile.get(
                "stop_after_verified_save_reason", ""
            ),
        }
        (pair_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        started = time.time()
        (pair_root / "stage_start.timestamp").write_text(
            f"{started:.9f}\n", encoding="utf-8"
        )
        print(f"\nRUN {result.title} [{result.key}]")
        print("Command:", " ".join(command))
        limited_command = [
            "timeout",
            "--signal=INT",
            "--kill-after=30s",
            str(args.timeout_seconds),
        ] + command
        train_log = pair_root / "train.log"
        runtime_errors: List[str] = []
        stop_when = None
        if profile.get("stop_after_verified_save") is True:
            stop_when = stable_verified_save_probe(output, profile, started)
        try:
            status, stopped_after_save = run_streaming(
                limited_command,
                result.entry.parent,
                env,
                train_log,
                stop_when=stop_when,
            )
        except Exception as exc:
            status = -1
            stopped_after_save = False
            runtime_errors.append(f"failed to launch or monitor training process: {exc!r}")
            print(f"FAIL {result.key}: {runtime_errors[-1]}")
        extra_globs = profile.get("result_globs", [])
        result_roots = result_output_roots(output, profile)
        try:
            saved = []
            unexpected = []
            for result_root in result_roots:
                saved.extend(
                    saved_result_files(
                        result_root,
                        expected_iteration=EXPECTED_ITERATION,
                        newer_than=started,
                        extra_globs=extra_globs if isinstance(extra_globs, list) else [],
                    )
                )
                unexpected.extend(
                    unexpected_iteration_artifacts(
                        result_root,
                        expected_iteration=EXPECTED_ITERATION,
                        newer_than=started,
                    )
                )
        except Exception as exc:
            saved = []
            unexpected = []
            runtime_errors.append(f"failed to inspect saved result artifacts: {exc!r}")
        passed, completion_mode = verified_completion(
            status, stopped_after_save, saved, unexpected, profile
        )
        record = {
            "method": result.key,
            "status": "passed" if passed else "failed",
            "exit_code": status,
            "stopped_after_verified_save": stopped_after_save,
            "completion_mode": completion_mode,
            "elapsed_seconds": round(time.time() - started, 3),
            "timeout_seconds": args.timeout_seconds,
            "output": str(output),
            "result_roots": [str(path) for path in result_roots],
            "command": command,
            "log": str(train_log),
            "log_tail": log_tail(train_log),
            "errors": runtime_errors,
            "saved_files": [str(path) for path in saved],
            "unexpected_iteration_artifacts": [str(path) for path in unexpected],
        }
        summary.append(record)
        if passed:
            (pair_root / ".train.done").write_text(
                time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8"
            )
            if stopped_after_save:
                suffix = " Framework stopped the method after the exact result became stable."
            elif completion_mode == "verified_save_after_process_exit":
                suffix = (
                    " Exact iteration=1 results were verified; a later upstream "
                    "post-save stage did not invalidate training completeness."
                )
            else:
                suffix = ""
            print(
                f"PASS {result.key}: saved {len(saved)} verified result file(s).{suffix}"
            )
        else:
            print(
                f"FAIL {result.key}: exit={status}, exact iteration=1 saved files={len(saved)}, "
                f"higher-iteration artifacts={len(unexpected)}."
            )

    report = attempt_root / "acceptance_results.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [item for item in summary if item["status"] != "passed"]
    print(f"\nAcceptance report: {report}")
    print(f"Passed: {len(summary) - len(failed)}")
    print(f"Failed/blocked: {len(failed)}")
    (output_root / "latest_attempt.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "attempt_root": str(attempt_root),
                "report": str(report),
                "passed": len(summary) - len(failed),
                "failed_or_blocked": len(failed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if failed:
        print()
        print("=" * 72)
        print("COMPLETE ITERATION=1 FAILURE SUMMARY")
        print("=" * 72)
        for item in failed:
            print(f"\n[{item['status']}] method={item['method']}")
            for error in item.get("errors", []):
                print(f"ERROR: {error}")
            if item.get("log"):
                print(f"LOG: {item['log']}")
            for line in item.get("log_tail", []):
                print(f"  {line}")
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
