from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_metrics import (
    METRICS,
    collect_pair_metrics,
    collect_rows,
    write_group_summaries,
    write_scene_summaries,
)
from scripts.run_method_stage import stage_preflight
from unified3dgs.method_catalog import load_confirmed_catalog, load_profiles, preflight_method
from unified3dgs.dataset_overlay import prepare_dataset_overlay
from unified3dgs.dataset_config import representative_datasets
from unified3dgs.method_backend import check_official_backend


DEFAULT_DATASETS: Tuple[Tuple[str, Path], ...] = representative_datasets()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def training_source_mentions_flag(train_payload: Dict[str, object], flag: str) -> bool:
    entry_value = train_payload.get("entry")
    if not entry_value:
        return False
    entry = Path(str(entry_value))
    if not entry.is_absolute():
        entry = PROJECT_ROOT / entry
    candidates = [entry]
    for root in (entry.parent / "arguments", entry.parent.parent / "arguments"):
        if root.is_dir():
            candidates.extend(sorted(root.rglob("*.py")))
    pattern = re.compile(rf"(?<![\w-]){re.escape(flag)}(?![\w-])")
    for path in candidates:
        try:
            if path.is_file() and pattern.search(
                path.read_text(encoding="utf-8", errors="replace")
            ):
                return True
        except OSError:
            continue
    return False


def parse_dataset_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip().strip("/")
    if not label or "/" not in label:
        raise argparse.ArgumentTypeError(
            "dataset label must look like dataset_family/scene"
        )
    return label, Path(raw_path.strip())


def run_logged(
    command: Sequence[str],
    log: Path,
    timeout_seconds: int,
    heartbeat_seconds: int,
    env: Optional[Dict[str, str]] = None,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=dict(env or os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name == "posix",
        )
        assert process.stdout is not None
        lines: "queue.Queue[Optional[str]]" = queue.Queue()

        def reader() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        started = time.monotonic()
        last_output = started
        last_heartbeat = started
        timed_out = False
        reader_done = False
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
                handle.write(line)
                handle.flush()
                last_output = now
            if (
                process.poll() is None
                and heartbeat_seconds > 0
                and now - last_heartbeat >= heartbeat_seconds
            ):
                message = (
                    "\nUnified 3DGS heartbeat: command still running "
                    f"(pid={process.pid}, elapsed={now - started:.0f}s, "
                    f"silent_for={now - last_output:.0f}s, log={log}).\n"
                )
                print(message, end="", flush=True)
                handle.write(message)
                handle.flush()
                last_heartbeat = now
            if (
                process.poll() is None
                and timeout_seconds > 0
                and now - started >= timeout_seconds
            ):
                message = (
                    f"\nUnified 3DGS: command exceeded {timeout_seconds}s; "
                    "terminating.\n"
                )
                print(message, end="", flush=True)
                handle.write(message)
                handle.flush()
                timed_out = True
                if os.name == "posix":
                    os.killpg(process.pid, 15)
                else:
                    process.terminate()
                break
        if timed_out:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, 9)
                else:
                    process.kill()
                process.wait()
            return 124
        return int(process.wait())


def verify_json_report(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("passed"))


def load_json_report(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def selected_methods(keys: Sequence[str]) -> List[Dict[str, object]]:
    catalog = {
        str(method["key"]): method
        for method in load_confirmed_catalog()
    }
    selected_keys = list(keys) if keys else sorted(catalog)
    unknown = sorted(set(selected_keys) - set(catalog))
    if unknown:
        raise ValueError("unknown confirmed method(s): " + ", ".join(unknown))
    return [catalog[key] for key in selected_keys]


def preflight_all(
    methods: Sequence[Dict[str, object]],
    profiles: Dict[str, Dict[str, object]],
    datasets: Sequence[Tuple[str, Path]],
) -> List[Dict[str, object]]:
    blockers: List[Dict[str, object]] = []
    primary_label, primary_dataset = datasets[0]
    for index, method in enumerate(methods, 1):
        key = str(method["key"])
        print(
            f"[preflight {index}/{len(methods)}] {key}: "
            "train CLI, render entry, official/unified evaluator",
            flush=True,
        )
        profile = profiles.get(key, {"key": key})
        backend = check_official_backend(key, profile, PROJECT_ROOT)
        train_payload: Dict[str, object]
        if backend.official:
            if backend.errors:
                train_payload = {
                    "passed": False,
                    "errors": backend.errors,
                    "command_flags": {},
                    "details": {},
                }
            else:
                backend_env = dict(backend.environment)
                pythonpath = backend_env.get("PYTHONPATH", "")
                backend_env["PYTHONPATH"] = (
                    f"{PROJECT_ROOT}{os.pathsep}{pythonpath}"
                    if pythonpath
                    else str(PROJECT_ROOT)
                )
                completed = subprocess.run(
                    [
                        str(backend.python),
                        str(
                            PROJECT_ROOT
                            / "scripts"
                            / "preflight_method_backend.py"
                        ),
                        "--method",
                        key,
                        "--dataset",
                        str(primary_dataset),
                    ],
                    cwd=PROJECT_ROOT,
                    env=backend_env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=180,
                )
                try:
                    train_payload = json.loads(
                        completed.stdout.strip().splitlines()[-1]
                    )
                except Exception:
                    train_payload = {
                        "passed": False,
                        "errors": [
                            "official-backend preflight did not return JSON: "
                            + (
                                completed.stderr.strip()
                                or completed.stdout.strip()
                                or f"exit={completed.returncode}"
                            )
                        ],
                        "command_flags": {},
                        "details": {},
                    }
        else:
            train_payload = preflight_method(
                method,
                profile,
                primary_dataset,
            ).as_dict()

        if train_payload.get("passed") is not True:
            blockers.append(
                {
                    "method": key,
                    "dataset": primary_label,
                    "stage": "train",
                    "errors": list(train_payload.get("errors", [])),
                }
            )
        elif dict(train_payload.get("command_flags", {})).get("eval") is None:
            blockers.append(
                {
                    "method": key,
                    "dataset": primary_label,
                    "stage": "train",
                    "errors": [
                        "training CLI has no verified evaluation-split flag; "
                        "held-out metrics cannot be guaranteed"
                    ],
                }
            )
        else:
            dataset_args = profile.get("official_dataset_args", {})
            if isinstance(dataset_args, dict):
                declared = set(
                    dict(train_payload.get("details", {})).get(
                        "static_declared_cli_options", []
                    )
                )
                for family, values in dataset_args.items():
                    if not isinstance(values, list):
                        continue
                    supplied = {
                        str(value).split("=", 1)[0]
                        for value in values
                        if str(value).startswith("--")
                    }
                    unknown = sorted(
                        flag
                        for flag in supplied
                        if flag not in declared
                        and not training_source_mentions_flag(train_payload, flag)
                    )
                    if unknown:
                        blockers.append(
                            {
                                "method": key,
                                "dataset": str(family),
                                "stage": "train",
                                "errors": [
                                    "official dataset args contain option(s) "
                                    "absent from the training CLI: "
                                    + ", ".join(unknown)
                                ],
                            }
                        )
            scene_args = profile.get("official_scene_args", {})
            if isinstance(scene_args, dict):
                declared = set(
                    dict(train_payload.get("details", {})).get(
                        "static_declared_cli_options", []
                    )
                )
                for scene_label, values in scene_args.items():
                    if not isinstance(values, list):
                        continue
                    supplied = {
                        str(value).split("=", 1)[0]
                        for value in values
                        if str(value).startswith("--")
                    }
                    unknown = sorted(
                        flag
                        for flag in supplied
                        if flag not in declared
                        and not training_source_mentions_flag(train_payload, flag)
                    )
                    if unknown:
                        blockers.append(
                            {
                                "method": key,
                                "dataset": str(scene_label),
                                "stage": "train",
                                "errors": [
                                    "official scene args contain option(s) "
                                    "absent from the training CLI: "
                                    + ", ".join(unknown)
                                ],
                            }
                        )
        for stage in ("render", "eval"):
            stage_result = stage_preflight(key, stage)
            if not stage_result.passed:
                blockers.append(
                    {
                        "method": key,
                        "stage": stage,
                        "errors": stage_result.errors,
                    }
                )
    return blockers


def command_for(
    script: str,
    method: str,
    config: str,
    dataset: Path,
    output: Path,
    values: Dict[str, object],
    extra_args: Sequence[object] = (),
) -> List[str]:
    command = [
        sys.executable,
        script,
        "--method",
        method,
        "--config",
        config,
        "--data",
        str(dataset),
        "--output",
        str(output),
    ]
    for key, value in values.items():
        command.extend(["--set", f"{key}={value}"])
    if extra_args:
        command.append("--")
        command.extend(str(value) for value in extra_args)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify methods with short train-render-eval runs that must "
            "produce finite PSNR/SSIM/LPIPS on all selected dataset families."
        )
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--dataset", action="append", type=parse_dataset_arg, default=[])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/validation/_method_acceptance"),
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--dependency-timeout-seconds",
        type=int,
        default=0,
        help="Timeout for one-time LPIPS/VGG preparation. Zero disables timeout.",
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--use-reduced-overlay",
        action="store_true",
        help=(
            "Diagnostic mode only: use a reduced writable dataset overlay. "
            "Results from this mode are not official-protocol acceptance."
        ),
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    methods = selected_methods(args.method)
    profiles = load_profiles()
    datasets = args.dataset or list(DEFAULT_DATASETS)
    output_root = resolve_path(args.output_root)

    print("Unified method metrics acceptance")
    print(f"Acceptance budget: iterations={args.iterations}")
    print("Required stages: train -> render -> eval -> finite PSNR/SSIM/LPIPS")
    print(f"Selected methods: {len(methods)}")
    for label, path in datasets:
        print(f"Dataset: {label} = {path}")
    if not args.run_real:
        print("Preview only. Add --run-real to execute metrics acceptance.")
        return 0
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        print("Refusing to train without an explicit CUDA_VISIBLE_DEVICES selection.")
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    overlays: List[Tuple[str, Path, Path]] = []
    dataset_errors: List[Dict[str, object]] = []
    for index, (label, path) in enumerate(datasets, 1):
        mode = "reduced diagnostic overlay" if args.use_reduced_overlay else "official source"
        print(f"[dataset {index}/{len(datasets)}] preparing {mode} for {label}", flush=True)
        try:
            source = resolve_path(path)
            overlay = (
                prepare_dataset_overlay(source, output_root)
                if args.use_reduced_overlay
                else source
            )
            overlays.append(
                (label, source, overlay)
            )
            print(f"[dataset {index}/{len(datasets)}] ready: {overlay}", flush=True)
        except Exception as exc:
            dataset_errors.append(
                {
                    "method": "*",
                    "dataset": label,
                    "stage": "dataset",
                    "errors": [repr(exc)],
                }
            )

    blockers = dataset_errors
    if not blockers:
        blockers.extend(
            preflight_all(
                methods,
                profiles,
                [(label, overlay) for label, _, overlay in overlays],
            )
        )
    if blockers:
        report = {
            "training_started": False,
            "blocker_count": len(blockers),
            "blockers": blockers,
        }
        report_path = output_root / "preflight_blockers.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("Metrics acceptance preflight failed. No training was started.")
        for blocker in blockers:
            dataset_label = f" {blocker.get('dataset')}" if blocker.get("dataset") else ""
            print(f"- {blocker['method']}{dataset_label} {blocker['stage']}:")
            for error in blocker["errors"]:
                print(f"  - {error}")
        print(f"Report: {report_path}")
        return 2

    metrics_env = os.environ.copy()
    metrics_cache = (
        PROJECT_ROOT
        / "third_party_build"
        / "runtime"
        / "method_runtime"
        / "shared_torch_cache"
    )
    metrics_cache.mkdir(parents=True, exist_ok=True)
    metrics_env["TORCH_HOME"] = str(metrics_cache)
    metrics_probe = [
        sys.executable,
        "-c",
        (
            "import torch, lpips, PIL, numpy; "
            "print('torch', torch.__version__, 'cuda', torch.version.cuda); "
            "print('Preparing LPIPS VGG weights in TORCH_HOME...'); "
            "model=lpips.LPIPS(net='vgg'); "
            "print('LPIPS VGG ready')"
        ),
    ]
    print()
    print("Preparing shared metrics dependencies before any training starts.")
    metrics_status = run_logged(
        metrics_probe,
        output_root / "metrics_dependency_preflight.log",
        max(0, args.dependency_timeout_seconds),
        max(0, args.heartbeat_seconds),
        env=metrics_env,
    )
    if metrics_status != 0:
        report = {
            "training_started": False,
            "blocker_count": 1,
            "blockers": [
                {
                    "method": "*",
                    "stage": "metrics_dependencies",
                    "errors": [
                        "LPIPS/VGG metrics dependency preparation failed; "
                        "see metrics_dependency_preflight.log"
                    ],
                }
            ],
        }
        report_path = output_root / "preflight_blockers.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("Metrics dependency preflight failed. No training was started.")
        print(f"Report: {report_path}")
        return 2

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    attempt_root = output_root / "attempts" / run_id
    attempt_root.mkdir(parents=True, exist_ok=False)

    results: List[Dict[str, object]] = []
    config = "configs/methods/catalog_method.yaml"
    for method in methods:
        key = str(method["key"])
        profile = profiles.get(key, {})
        for label, source_dataset, default_overlay in overlays:
            pair_root = attempt_root / key / label
            output = pair_root / "method_outputs"
            resolution_value = profile.get("acceptance_resolution")
            dataset_family = label.split("/", 1)[0]
            dataset = default_overlay
            point_caps = profile.get("acceptance_dataset_point_caps", {})
            if (
                args.use_reduced_overlay
                and isinstance(point_caps, dict)
                and dataset_family in point_caps
            ):
                dataset = prepare_dataset_overlay(
                    source_dataset,
                    output_root,
                    max_points=int(point_caps[dataset_family]),
                )
                print(
                    f"[{key} {label}] using method-specific point cap "
                    f"{int(point_caps[dataset_family])}: {dataset}",
                    flush=True,
                )
            common_values = {
                "timeout_seconds": max(0, args.timeout_seconds),
                "heartbeat_seconds": max(0, args.heartbeat_seconds),
            }
            acceptance_images = profile.get("acceptance_images")
            if acceptance_images:
                common_values["images"] = str(acceptance_images)
            elif args.use_reduced_overlay:
                common_values["images"] = "images_8"
            if resolution_value is not None:
                common_values["resolution"] = int(resolution_value)
            row: Dict[str, object] = {
                "method": key,
                "dataset": label,
                "output": str(output),
                "stages": {},
                "metrics": {},
                "passed": False,
                "official_protocol": not args.use_reduced_overlay,
            }
            print()
            print("=" * 72)
            print(f"{key} on {label}: train -> render -> eval metrics")
            print("=" * 72)

            train_values = {
                **common_values,
                "iterations": args.iterations,
                "dataset_label": label,
                "eval": "true",
            }
            stage_commands = [
                (
                    "train",
                    command_for(
                        "train_all.py",
                        key,
                        config,
                        dataset,
                        output,
                        train_values,
                    ),
                    pair_root / "train.log",
                ),
                (
                    "render",
                    command_for(
                        "render_all.py",
                        key,
                        config,
                        dataset,
                        output,
                        {**common_values, "render_iteration": args.iterations, "eval": "true"},
                    ),
                    pair_root / "render.log",
                ),
                (
                    "eval",
                    command_for(
                        "eval_all.py",
                        key,
                        config,
                        dataset,
                        output,
                        {**common_values, "render_iteration": args.iterations},
                    ),
                    pair_root / "eval.log",
                ),
            ]

            failed = False
            for stage, command, log in stage_commands:
                print(f"\n[{key} {label}] {stage}")
                print("Command:", " ".join(str(value) for value in command))
                status = run_logged(
                    command,
                    log,
                    (
                        args.timeout_seconds + 60
                        if args.timeout_seconds > 0
                        else 0
                    ),
                    max(0, args.heartbeat_seconds),
                )
                row["stages"][stage] = {"exit_code": status, "log": str(log)}
                if stage == "train":
                    training_report = load_json_report(
                        output / "unified3dgs_training_report.json"
                    )
                    if training_report:
                        row["official_runtime_verified"] = training_report.get(
                            "official_runtime_verified", False
                        )
                        classification = training_report.get(
                            "failure_classification"
                        )
                        if isinstance(classification, dict):
                            row["failure_classification"] = classification
                            row["hardware_limited"] = (
                                classification.get("category")
                                == "hardware_limit_confirmed"
                                and classification.get("objective_limit") is True
                            )
                if status != 0:
                    failed = True
                    break
                if stage == "train" and not verify_json_report(
                    output / "unified3dgs_training_report.json"
                ):
                    row["stages"][stage]["verification_error"] = (
                        "missing or failed unified3dgs_training_report.json"
                    )
                    failed = True
                    break
                if stage == "train":
                    training_report = load_json_report(
                        output / "unified3dgs_training_report.json"
                    )
                    row["official_runtime_verified"] = training_report.get(
                        "official_runtime_verified", False
                    )
                if stage in {"render", "eval"}:
                    verify_stage = "render" if stage == "render" else "eval"
                    verify_command = [
                        sys.executable,
                        "scripts/verify_scene_outputs.py",
                        "--stage",
                        verify_stage,
                        "--output",
                        str(output),
                        "--iteration",
                        str(args.iterations),
                    ]
                    verify_status = run_logged(
                        verify_command,
                        pair_root / f"verify_{verify_stage}.log",
                        max(0, args.timeout_seconds),
                        max(0, args.heartbeat_seconds),
                    )
                    row["stages"][f"verify_{verify_stage}"] = {
                        "exit_code": verify_status,
                        "log": str(pair_root / f"verify_{verify_stage}.log"),
                    }
                    if verify_status != 0:
                        failed = True
                        break

            if not failed:
                metrics, source = collect_pair_metrics(pair_root, args.iterations)
                row["metrics"] = metrics
                row["metric_source"] = source
                missing = [name for name in METRICS if name not in metrics]
                if missing:
                    row["metric_error"] = "missing metrics: " + ", ".join(missing)
                    failed = True
                else:
                    if row.get("official_runtime_verified") is not True:
                        row["protocol_error"] = (
                            "metrics were produced, but official runtime "
                            "compatibility has not been verified"
                        )
                        failed = True
                    else:
                        row["passed"] = True
                        print(
                            "Verified metrics: "
                            + ", ".join(
                                f"{name}={metrics[name]:.6f}" for name in METRICS
                            )
                        )

            results.append(row)
            if failed and args.stop_on_failure:
                break
        if args.stop_on_failure and results and not results[-1]["passed"]:
            break

    passed = [row for row in results if row.get("passed")]
    hardware_limited = [
        row for row in results if row.get("hardware_limited") is True
    ]
    program_failed = [
        row
        for row in results
        if not row.get("passed") and row.get("hardware_limited") is not True
    ]
    expected_result_count = len(methods) * len(overlays)
    report = {
        "all_passed": len(passed) == expected_result_count,
        "expected_result_count": expected_result_count,
        "hardware_limited_count": len(hardware_limited),
        "passed_result_count": len(passed),
        "program_failed_count": len(program_failed),
        "iterations": args.iterations,
        "attempt_root": str(attempt_root),
        "results": results,
        "protocol_mode": (
            "reduced_diagnostic"
            if args.use_reduced_overlay
            else "official_short"
        ),
    }
    report_path = attempt_root / "metrics_acceptance_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scene_rows = collect_rows(attempt_root, args.iterations)
    write_scene_summaries(scene_rows)
    write_group_summaries(scene_rows, attempt_root)
    latest = output_root / "latest_attempt.json"
    latest.write_text(
        json.dumps({"report": str(report_path)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print()
    print("=" * 72)
    print(
        f"Metrics acceptance: {len(passed)}/{report['expected_result_count']} "
        "method/dataset pairs passed"
    )
    if hardware_limited:
        print(
            f"Hardware-limited pairs: {len(hardware_limited)} "
            "(official runtime/protocol reached an objective GPU-memory limit)"
        )
    if program_failed:
        print(f"Program/environment failed pairs: {len(program_failed)}")
    print(f"Report: {report_path}")
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
