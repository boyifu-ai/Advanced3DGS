from __future__ import annotations

import argparse
from collections import deque
import json
import os
import queue
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    build_method_env,
    build_training_command,
    preflight_method,
    reader_safety_errors,
    result_output_roots,
    saved_result_files,
    select_methods,
    official_dataset_args,
    unexpected_iteration_artifacts,
)
from unified3dgs.method_backend import (
    activate_official_backend,
    classify_failure,
)
from unified3dgs.utils.network import available_tcp_port, forwarded_has_flag


FAILED_OUTPUT_REPORT = "unified3dgs_training_report.json"
DISPOSABLE_FAILED_OUTPUT_FILES = {
    FAILED_OUTPUT_REPORT,
    "cameras.json",
    "cfg_args",
    "exposure.json",
    "input.ply",
}


def remove_disposable_failed_output(output: Path) -> bool:
    """Remove a failed run only when it contains no reusable result artifacts."""
    if not output.is_dir() or output.is_symlink():
        return False
    report_path = output / FAILED_OUTPUT_REPORT
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(report, dict) or report.get("passed") is not False:
        return False

    saved_files = report.get("saved_files", [])
    if not isinstance(saved_files, list) or saved_files:
        return False
    resolved_roots = report.get("resolved_result_roots", [])
    if not isinstance(resolved_roots, list):
        return False
    for value in resolved_roots:
        try:
            root = Path(str(value)).expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        if root != output and root.is_dir() and any(root.iterdir()):
            return False

    for path in output.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        if len(relative.parts) == 1 and (
            relative.name in DISPOSABLE_FAILED_OUTPUT_FILES
            or relative.name.startswith("events.out.tfevents")
        ):
            continue
        return False

    shutil.rmtree(output)
    return True


def resource_errors(output: Path, min_disk_gb: float, min_vram_gb: float) -> List[str]:
    errors: List[str] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(output.parent).free
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
        errors.append(f"cannot inspect selected GPU memory: {exc!r}")
    return errors


def auto_training_port_args(
    result: object,
    user_forwarded: Sequence[object],
    output: Path,
) -> List[str]:
    declared = set(result.details.get("static_declared_cli_options", []))
    if "--port" not in declared or forwarded_has_flag(user_forwarded, "--port"):
        return []
    port = available_tcp_port(f"{result.key}:{output}")
    return ["--port", str(port)]


def reader_patch_errors(repo: Path) -> List[str]:
    return reader_safety_errors(repo / "scene" / "dataset_readers.py")


def stable_save_probe(
    output: Path,
    profile: Dict[str, object],
    iteration: int,
    newer_than: float,
    settle_seconds: float = 1.0,
) -> Callable[[], bool]:
    previous: Optional[Tuple[Tuple[str, int, int], ...]] = None
    unchanged_since = 0.0
    extra_globs = profile.get("result_globs", [])
    if not isinstance(extra_globs, list):
        extra_globs = []

    def probe() -> bool:
        nonlocal previous, unchanged_since
        files: List[Path] = []
        try:
            for root in result_output_roots(output, profile):
                files.extend(
                    saved_result_files(
                        root,
                        expected_iteration=iteration,
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


def signal_process_group(process: subprocess.Popen, sig: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
    except ProcessLookupError:
        pass


def run_streaming(
    command: Sequence[str],
    cwd: Path,
    env: Dict[str, str],
    stop_when: Optional[Callable[[], bool]],
    timeout_seconds: int = 0,
    heartbeat_seconds: int = 30,
) -> Tuple[int, bool, bool, str]:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=os.name == "posix",
    )
    assert process.stdout is not None
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def read_stdout() -> None:
        try:
            assert process.stdout is not None
            try:
                for output_line in process.stdout:
                    lines.put(output_line)
            except (OSError, ValueError):
                # communicate() closes this pipe after a verified-save stop.
                pass
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    started = time.monotonic()
    last_output = started
    last_heartbeat = started
    stopped_after_save = False
    timed_out = False
    reader_done = False
    output_tail = deque(maxlen=200)
    while process.poll() is None or not reader_done or not lines.empty():
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            line = ""
        now = time.monotonic()
        if line is None:
            reader_done = True
        elif line:
            print(line, end="", flush=True)
            output_tail.append(line)
            last_output = now
        if process.poll() is None and stop_when is not None and stop_when():
            print(
                "\nUnified 3DGS: requested final result is stable; "
                "ending upstream post-training work.",
                flush=True,
            )
            stopped_after_save = True
            signal_process_group(process, signal.SIGTERM)
            break
        if (
            process.poll() is None
            and heartbeat_seconds > 0
            and now - last_heartbeat >= heartbeat_seconds
        ):
            print(
                "\nUnified 3DGS heartbeat: upstream training is still running "
                f"(pid={process.pid}, elapsed={now - started:.0f}s, "
                f"silent_for={now - last_output:.0f}s).",
                flush=True,
            )
            last_heartbeat = now
        if (
            process.poll() is None
            and timeout_seconds > 0
            and now - started >= timeout_seconds
        ):
            print(
                f"\nUnified 3DGS: upstream training exceeded {timeout_seconds}s; "
                "terminating.",
                flush=True,
            )
            timed_out = True
            signal_process_group(process, signal.SIGTERM)
            break
    if stopped_after_save:
        try:
            remainder, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            remainder, _ = process.communicate()
        if remainder:
            print(remainder, end="", flush=True)
    elif timed_out:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            process.wait()
    return process.wait(), stopped_after_save, timed_out, "".join(output_tail)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a confirmed method through its configurable unified "
            "long-training interface."
        )
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--images", default="")
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--test-iterations", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--min-free-disk-gb", type=float, default=5.0)
    parser.add_argument("--min-free-vram-gb", type=float, default=4.0)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        print("Refusing to train without an explicit CUDA_VISIBLE_DEVICES selection.")
        return 2

    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        print(f"Refusing output outside project workspace: {output}")
        return 2
    if output.is_dir() and any(output.iterdir()):
        if remove_disposable_failed_output(output):
            print(
                "Removed a verified failed-run residue with no reusable model "
                f"artifacts: {output}"
            )
        else:
            print(f"Refusing to mix a new training run with existing output: {output}")
            print(
                "The directory has no disposable failed-run report or contains "
                "reusable artifacts. Resume it or explicitly request a clean rerun."
            )
            return 2

    selected = select_methods([args.method])
    method, profile = selected[0]
    backend = activate_official_backend(
        args.method,
        profile,
        PROJECT_ROOT,
        Path(__file__).resolve(),
        sys.argv[1:],
    )
    if backend.errors:
        output.mkdir(parents=True, exist_ok=True)
        classification = classify_failure("", backend, official_protocol=True)
        report = {
            "method": args.method,
            "iterations": args.iterations,
            "dataset": str(dataset),
            "output": str(output),
            "passed": False,
            "official_protocol": True,
            "official_backend": backend.runtime,
            "official_runtime_verified": backend.official and backend.passed,
            "failure_classification": classification,
        }
        (output / "unified3dgs_training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("Official backend preflight failed. No training was started.")
        for error in backend.errors:
            print(f"- {error}")
        return 3
    result = preflight_method(method, profile, dataset)
    if not result.passed:
        print("Long-training preflight failed. No training was started.")
        for error in result.errors:
            print(f"- {result.key}: {error}")
        return 2
    patch_errors = reader_patch_errors(result.repo)
    if patch_errors:
        print("Long-training reader-safety preflight failed. No training was started.")
        for error in patch_errors:
            print(f"- {result.key}: {error}")
        return 2

    user_forwarded = list(args.extra_args)
    if user_forwarded and user_forwarded[0] == "--":
        user_forwarded = user_forwarded[1:]
    auto_forwarded = auto_training_port_args(result, user_forwarded, output)
    forwarded = (
        official_dataset_args(profile, dataset, args.dataset_label or None)
        + auto_forwarded
        + user_forwarded
    )
    command = build_training_command(
        result=result,
        dataset=dataset,
        output=output,
        iterations=args.iterations,
        profile=profile,
        images=args.images or None,
        resolution=args.resolution,
        eval_enabled=args.eval,
        test_iterations=args.test_iterations,
        extra_args=forwarded,
    )
    declared = set(result.details.get("static_declared_cli_options", []))
    supplied_flags = {
        str(value).split("=", 1)[0]
        for value in command
        if str(value).startswith("-")
    }
    required = set(result.details.get("required_cli_options", []))
    command_errors = [
        "long-training command is missing required option(s): "
        + ", ".join(sorted(required - supplied_flags))
        if required - supplied_flags
        else ""
    ]
    forwarded_flags = {
        str(value).split("=", 1)[0]
        for value in user_forwarded
        if str(value).startswith("--")
    }
    unknown_forwarded = sorted(forwarded_flags - declared)
    if unknown_forwarded:
        command_errors.append(
            "user extra_args contain option(s) absent from the training CLI: "
            + ", ".join(unknown_forwarded)
        )
    command_errors = [error for error in command_errors if error]
    if command_errors:
        print("Long-training command audit failed. No training was started.")
        for error in command_errors:
            print(f"- {error}")
        return 2

    errors = resource_errors(output, args.min_free_disk_gb, args.min_free_vram_gb)
    if errors:
        print("Long-training resource check failed. No training was started.")
        for error in errors:
            print(f"- {error}")
        return 2

    env = build_method_env(result.key, result.repo, runtime_namespace="method_runtime")
    env["UNIFIED3DGS_OUTPUT_PATH"] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stop_when = None
    if profile.get("stop_after_verified_save") is True:
        stop_when = stable_save_probe(output, profile, args.iterations, started)

    print(f"Method: {result.key}")
    print(f"Iterations: {args.iterations}")
    print(f"Dataset: {dataset}")
    print(f"Output: {output}")
    print("Command:", " ".join(str(value) for value in command))
    status = 1
    stopped_after_save = False
    timed_out = False
    captured_output = ""
    status, stopped_after_save, timed_out, captured_output = run_streaming(
        command,
        result.entry.parent,
        env,
        stop_when,
        timeout_seconds=max(0, args.timeout_seconds),
        heartbeat_seconds=max(0, args.heartbeat_seconds),
    )

    saved: List[Path] = []
    unexpected: List[Path] = []
    resolved_roots: List[Path] = []
    extra_globs = profile.get("result_globs", [])
    if not isinstance(extra_globs, list):
        extra_globs = []
    for root in result_output_roots(output, profile):
        if root.is_dir():
            resolved_roots.append(root.resolve())
        saved.extend(
            saved_result_files(
                root,
                expected_iteration=args.iterations,
                newer_than=started,
                extra_globs=extra_globs,
            )
        )
        unexpected.extend(
            unexpected_iteration_artifacts(
                root, expected_iteration=args.iterations, newer_than=started
            )
        )
    passed = bool(saved) and not unexpected and not timed_out and (status == 0 or stopped_after_save)
    classification = (
        {"category": "passed", "objective_limit": False, "reason": ""}
        if passed
        else classify_failure(
            captured_output,
            backend,
            official_protocol=True,
        )
    )
    report = {
        "method": result.key,
        "iterations": args.iterations,
        "dataset": str(dataset),
        "output": str(output),
        "command": command,
        "exit_code": status,
        "stopped_after_verified_save": stopped_after_save,
        "timed_out": timed_out,
        "resolved_result_roots": [
            str(path) for path in sorted(set(resolved_roots))
        ],
        "saved_files": [str(path) for path in sorted(set(saved))],
        "unexpected_iteration_artifacts": [
            str(path) for path in sorted(set(unexpected))
        ],
        "passed": passed,
        "official_protocol": True,
        "official_backend": backend.runtime,
        "official_runtime_verified": backend.official and backend.passed,
        "failure_classification": classification,
    }
    (output / "unified3dgs_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        print("Training output verification failed.")
        print(json.dumps(report, indent=2, sort_keys=True))
        return status or 2
    print(f"Verified final iteration {args.iterations}: {len(saved)} saved file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
