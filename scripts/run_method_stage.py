from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    EVAL_FLAGS,
    IMAGES_FLAGS,
    OUTPUT_FLAGS,
    SOURCE_FLAGS,
    build_method_env,
    first_declared_flag,
    first_present_flag,
    has_standard_3dgs_arguments,
    resolve_project_path,
    select_methods,
    static_cli_text,
    static_declared_flags,
)
from unified3dgs.metrics_io import resolved_model_output
from unified3dgs.method_backend import activate_official_backend


DEFAULT_RENDER_ENTRIES = (
    "render.py",
    "scripts/render.py",
    "ms/render.py",
    "ms_d/render.py",
)
RENDER_ITERATION_FLAGS = ("--iteration", "--load_iteration", "--loaded_iter")
SKIP_TRAIN_FLAGS = ("--skip_train",)


@dataclass
class StagePreflight:
    method: str
    stage: str
    repo: Path
    entry: Optional[Path]
    flags: Dict[str, Optional[str]]
    declared: List[str]
    errors: List[str]

    @property
    def passed(self) -> bool:
        return not self.errors


def choose_existing(repo: Path, candidates: Sequence[object]) -> Optional[Path]:
    for candidate in candidates:
        path = repo / str(candidate)
        if path.is_file():
            return path
    return None


def discover_render_entry(repo: Path) -> Optional[Path]:
    candidates = []
    for path in repo.rglob("render.py"):
        try:
            relative = path.relative_to(repo)
        except ValueError:
            continue
        lowered = {part.lower() for part in relative.parts}
        if lowered & {"submodules", "third_party", "sibr_viewers", ".git"}:
            continue
        if len(relative.parts) <= 4:
            candidates.append(path)
    if not candidates:
        return None
    return min(candidates, key=lambda path: (len(path.relative_to(repo).parts), str(path)))


def framework_contract_errors(
    repo: Path, profile: Dict[str, object]
) -> List[str]:
    contracts = profile.get("framework_render_contracts", [])
    if not isinstance(contracts, list):
        return ["profile framework_render_contracts must be a list"]
    errors: List[str] = []
    for contract in contracts:
        if not isinstance(contract, dict) or not contract.get("path"):
            errors.append(f"invalid framework render contract: {contract!r}")
            continue
        path = (repo / str(contract["path"])).resolve()
        try:
            path.relative_to(repo.resolve())
        except ValueError:
            errors.append(f"framework render contract escapes repository: {path}")
            continue
        if not path.is_file():
            errors.append(f"framework render contract source is missing: {path}")
            continue
        markers = contract.get("contains", [])
        if not isinstance(markers, list) or not markers:
            errors.append(f"framework render contract has no markers: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [str(marker) for marker in markers if str(marker) not in text]
        if missing:
            errors.append(
                f"framework render contract changed in {path}: missing "
                + ", ".join(repr(marker) for marker in missing)
            )
    return errors


def stage_preflight(method: str, stage: str) -> StagePreflight:
    selected = select_methods([method])
    method_row, profile = selected[0]
    repo = resolve_project_path(method_row["local_path"])
    key = str(method_row["key"])
    errors: List[str] = []

    if stage == "eval":
        official_style = str(profile.get("official_metrics_style") or "").strip()
        entry = (
            PROJECT_ROOT / "scripts" / "evaluate_render_pairs_official.py"
            if official_style
            else PROJECT_ROOT / "scripts" / "evaluate_render_pairs.py"
        )
        if not entry.is_file():
            errors.append(f"missing metrics evaluator: {entry}")
        if official_style:
            if official_style not in {"standard_3dgs", "beta_splatting"}:
                errors.append(
                    f"unsupported official_metrics_style: {official_style}"
                )
            required = [repo / "utils" / "image_utils.py", repo / "lpipsPyTorch"]
            if official_style == "standard_3dgs":
                required.append(repo / "utils" / "loss_utils.py")
            for path in required:
                if not path.exists():
                    errors.append(
                        "official metrics dependency is missing: "
                        f"{path.relative_to(repo) if repo in path.parents else path}"
                    )
        return StagePreflight(
            key,
            stage,
            repo,
            entry if entry.is_file() else None,
            {
                "source": None,
                "output": "--output",
                "images": None,
                "resolution": None,
                "eval": None,
                "iteration": "--iteration",
                "skip_train": None,
                "official_style": official_style or None,
            },
            ["--output", "--iteration", "--results-output"],
            errors,
        )

    default_candidates = list(DEFAULT_RENDER_ENTRIES)
    framework_entry = profile.get("framework_render_entry")
    if framework_entry:
        entry = (PROJECT_ROOT / str(framework_entry)).resolve()
        candidates = [str(framework_entry)]
        errors.extend(framework_contract_errors(repo, profile))
    else:
        entry = None
        candidates = profile.get(
            "render_entry_candidates", default_candidates
        )
    if not isinstance(candidates, list):
        candidates = default_candidates
        errors.append("profile render_entry_candidates must be a list")

    entry = entry or choose_existing(repo, candidates)
    if entry is None:
        entry = discover_render_entry(repo)
    flags: Dict[str, Optional[str]] = {
        "source": None,
        "output": None,
        "images": None,
        "resolution": None,
        "eval": None,
        "iteration": None,
        "skip_train": None,
    }
    declared: List[str] = []
    if entry is None:
        errors.append(
            "no supported "
            f"{stage} entry found; checked: " + ", ".join(str(item) for item in candidates)
        )
        return StagePreflight(key, stage, repo, None, flags, declared, errors)

    try:
        compile(entry.read_text(encoding="utf-8"), str(entry), "exec")
    except Exception as exc:
        errors.append(f"{stage} entry does not compile: {exc}")

    text = static_cli_text(repo, entry)
    declared = static_declared_flags(repo, entry)
    flags["output"] = first_present_flag(text, OUTPUT_FLAGS)
    flags["source"] = first_present_flag(text, SOURCE_FLAGS)
    flags["images"] = first_present_flag(text, IMAGES_FLAGS)
    flags["eval"] = first_present_flag(text, EVAL_FLAGS)
    flags["iteration"] = first_present_flag(text, RENDER_ITERATION_FLAGS)
    flags["skip_train"] = first_present_flag(text, SKIP_TRAIN_FLAGS)
    flags["resolution"] = "--resolution" if "--resolution" in declared else None
    declared_fallbacks = {
        "source": SOURCE_FLAGS,
        "output": OUTPUT_FLAGS,
        "images": IMAGES_FLAGS,
        "eval": EVAL_FLAGS,
        "iteration": RENDER_ITERATION_FLAGS,
        "skip_train": SKIP_TRAIN_FLAGS,
    }
    for name, candidates in declared_fallbacks.items():
        if flags[name] is None:
            flags[name] = first_declared_flag(declared, candidates)
    if has_standard_3dgs_arguments(repo, entry):
        flags["source"] = flags["source"] or "-s"
        flags["output"] = flags["output"] or "-m"
    overrides = profile.get("render_flags", {})
    if isinstance(overrides, dict):
        for name, value in overrides.items():
            if name in flags and value:
                flags[name] = str(value)

    if flags["output"] is None:
        errors.append(f"could not resolve required {stage} output/model argument")
    if stage == "render" and flags["source"] is None:
        errors.append("could not resolve required render source/dataset argument")
    paired_output_markers = (
        r"""["']gt["']""",
        r"\bgts?_path\b",
        r"\bground_truth\b",
        r"save_image\s*\(\s*gt",
    )
    if not framework_entry and not any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in paired_output_markers
    ):
        errors.append(
            "render entry does not statically show paired GT export; add a framework "
            "render wrapper or explicit profile support before metrics acceptance"
        )
    return StagePreflight(key, stage, repo, entry, flags, declared, errors)


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
    timeout_seconds: int,
    heartbeat_seconds: int,
) -> Tuple[int, bool]:
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
            last_output = now
        if (
            process.poll() is None
            and heartbeat_seconds > 0
            and now - last_heartbeat >= heartbeat_seconds
        ):
            print(
                "\nUnified 3DGS heartbeat: "
                f"{Path(command[1]).name if len(command) > 1 else 'stage'} still running "
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
                f"\nUnified 3DGS: stage exceeded {timeout_seconds}s; terminating.",
                flush=True,
            )
            timed_out = True
            signal_process_group(process, signal.SIGTERM)
            break
    if timed_out:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            process.wait()
    return process.wait(), timed_out


def build_stage_command(
    preflight: StagePreflight,
    dataset: Path,
    output: Path,
    iteration: int,
    images: str,
    resolution: Optional[str],
    eval_enabled: bool,
    extra_args: Sequence[str],
    results_output: Optional[Path] = None,
    profile: Optional[Dict[str, object]] = None,
) -> List[str]:
    if preflight.entry is None or not preflight.passed:
        raise ValueError(f"Cannot build {preflight.stage} command for failed preflight")
    command = [sys.executable, str(preflight.entry)]
    if preflight.stage == "render":
        command.extend([str(preflight.flags["source"]), str(dataset)])
        command.extend([str(preflight.flags["output"]), str(output)])
        if preflight.flags.get("iteration"):
            command.extend([str(preflight.flags["iteration"]), str(iteration)])
        if preflight.flags.get("images") and images:
            command.extend([str(preflight.flags["images"]), images])
        if preflight.flags.get("resolution") and resolution is not None:
            command.extend([str(preflight.flags["resolution"]), str(resolution)])
        if preflight.flags.get("eval") and eval_enabled:
            command.append(str(preflight.flags["eval"]))
        if preflight.flags.get("skip_train"):
            command.append(str(preflight.flags["skip_train"]))
    else:
        official_style = preflight.flags.get("official_style")
        if official_style:
            command.extend(
                [
                    "--method",
                    preflight.method,
                    "--repo",
                    str(preflight.repo),
                    "--style",
                    str(official_style),
                    str(preflight.flags["output"]),
                    str(output),
                    str(preflight.flags["iteration"]),
                    str(iteration),
                ]
            )
        else:
            command.extend(
                [
                    str(preflight.flags["output"]),
                    str(output),
                    str(preflight.flags["iteration"]),
                    str(iteration),
                ]
            )
        if results_output is not None:
            command.extend(["--results-output", str(results_output)])
    if profile and preflight.stage == "render":
        render_extra_args = profile.get("render_extra_args", [])
        if isinstance(render_extra_args, list):
            replacements = {
                "{dataset}": str(dataset),
                "{output}": str(output),
                "{iteration}": str(iteration),
            }
            for value in render_extra_args:
                rendered = str(value)
                for token, replacement in replacements.items():
                    rendered = rendered.replace(token, replacement)
                command.append(rendered)
    command.extend(extra_args)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a render or eval stage for a confirmed method."
    )
    parser.add_argument("--stage", required=True, choices=("render", "eval"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--images", default="")
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    _, profile = select_methods([args.method])[0]
    backend = activate_official_backend(
        args.method,
        profile,
        PROJECT_ROOT,
        Path(__file__).resolve(),
        sys.argv[1:],
    )
    if backend.errors:
        print("Official backend preflight failed. No stage was started.")
        for error in backend.errors:
            print(f"- {error}")
        return 3

    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        print(f"Refusing output outside project workspace: {output}")
        return 2

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    preflight = stage_preflight(args.method, args.stage)
    if not preflight.passed:
        print(f"{args.stage} preflight failed. No stage was started.")
        for error in preflight.errors:
            print(f"- {preflight.method}: {error}")
        return 2

    model_output = resolved_model_output(output)
    command = build_stage_command(
        preflight,
        dataset,
        model_output,
        args.iteration,
        args.images,
        args.resolution,
        args.eval,
        extra_args,
        results_output=output / "results.json",
        profile=profile,
    )
    env = build_method_env(preflight.method, preflight.repo, runtime_namespace="method_runtime")
    env["UNIFIED3DGS_OUTPUT_PATH"] = str(output)
    print(f"Method: {preflight.method}")
    print(f"Stage: {args.stage}")
    print(f"Dataset: {dataset}")
    print(f"Output: {output}")
    if model_output != output:
        print(f"Resolved model output: {model_output}")
    print("Command:", " ".join(str(value) for value in command))
    started = time.time()
    status, timed_out = run_streaming(
        command,
        preflight.entry.parent if preflight.entry else preflight.repo,
        env,
        max(0, args.timeout_seconds),
        max(0, args.heartbeat_seconds),
    )
    report = {
        "method": preflight.method,
        "stage": args.stage,
        "dataset": str(dataset),
        "output": str(output),
        "resolved_model_output": str(model_output),
        "iteration": args.iteration,
        "command": command,
        "exit_code": status,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.time() - started, 3),
        "passed": status == 0 and not timed_out,
    }
    (output / f"unified3dgs_{args.stage}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report["passed"] else status or 2


if __name__ == "__main__":
    raise SystemExit(main())
