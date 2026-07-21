from __future__ import annotations

import os
import getpass
import json
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from unified3dgs.config.loader import load_config, parse_scalar
from unified3dgs.dataset_config import (
    LOCAL_DATASET_PATHS,
    clear_local_dataset_path,
    dataset_root,
    load_dataset_definitions,
    load_local_dataset_paths,
    readonly_dataset_root_env,
    save_local_dataset_path,
)
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_REPOSITORY_URL = "https://github.com/3DAgentWorld/Advanced3DGS/"
DEFAULT_VALIDATION_ROOT = Path("outputs/validation")
DEFAULT_ITERATION = 30000
DEFAULT_RESOLUTION = -1
METHOD_CATALOG_PATH = PROJECT_ROOT / "configs" / "method_catalog.json"


@dataclass(frozen=True)
class MethodOption:
    roman: str
    key: str
    title: str
    config: str
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetFamily:
    letter: str
    key: str
    title: str
    scenes: Tuple[Tuple[str, str], ...]


def to_roman(number: int) -> str:
    values = (
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    result = ""
    remaining = number
    while remaining:
        for value, numeral in values:
            if value <= remaining:
                result += numeral
                remaining -= value
                break
    return result


EXTENDED_WORKFLOW_METHODS: Tuple[MethodOption, ...] = (
    MethodOption(
        "I",
        "vanilla_3dgs",
        "Vanilla 3DGS",
        "configs/methods/vanilla_3dgs.yaml",
        ("Uses Graphdeco train/render/metrics scripts.",),
    ),
    MethodOption(
        "II",
        "2dgs",
        "2D Gaussian Splatting",
        "configs/methods/2dgs.yaml",
        (
            "Formal validation keeps test renders and skips large train-view exports.",
            "Mesh export is disabled by default to reduce disk pressure.",
        ),
    ),
    MethodOption(
        "III",
        "3dgs_mcmc",
        "3DGS-MCMC",
        "configs/methods/3dgs_mcmc.yaml",
        (
            "Requires scene configs containing cap_max.",
            "treehill/flowers use generated project-extension configs when upstream configs are absent.",
        ),
    ),
    MethodOption(
        "IV",
        "3dhgs",
        "3D-HGS / Half Gaussian Splitting",
        "configs/methods/3dhgs.yaml",
        (
            "Renderer is patched to save paired GT images for unified metrics.",
            "Evaluation uses the compatible Vanilla 3DGS metrics script.",
        ),
    ),
    MethodOption(
        "V",
        "sss",
        "3D Student Splatting and Scooping",
        "configs/methods/sss.yaml",
        (
            "Requires SSS scene configs containing cap_max.",
            "treehill/flowers use generated project-extension configs when upstream configs are absent.",
        ),
    ),
)


def load_catalog_method_options() -> Tuple[MethodOption, ...]:
    data = json.loads(METHOD_CATALOG_PATH.read_text(encoding="utf-8"))
    confirmed = [
        method
        for method in data
        if method.get("source_status") == "confirmed" and method.get("repository")
    ]
    start = len(EXTENDED_WORKFLOW_METHODS) + 1
    return tuple(
        MethodOption(
            to_roman(index),
            str(method["key"]),
            str(method["title"]),
            "configs/methods/catalog_method.yaml",
            (
                "Unified train, render, and evaluate interface.",
                "Requires paired test renders and finite PSNR, SSIM, and LPIPS.",
            ),
        )
        for index, method in enumerate(confirmed, start)
    )


CATALOG_METHODS = load_catalog_method_options()
CATALOG_METHOD_KEYS = frozenset(method.key for method in CATALOG_METHODS)
METHODS: Tuple[MethodOption, ...] = EXTENDED_WORKFLOW_METHODS + CATALOG_METHODS
METHODS_BY_KEY = {method.key: method for method in METHODS}


def load_dataset_families() -> Tuple[DatasetFamily, ...]:
    families: List[DatasetFamily] = []
    for index, definition in enumerate(load_dataset_definitions()):
        root = dataset_root(definition)
        families.append(
            DatasetFamily(
                chr(ord("A") + index),
                definition.key,
                definition.title,
                tuple(
                    (str(root / scene), f"{definition.key}/{scene}")
                    for scene in definition.scenes
                ),
            )
        )
    return tuple(families)


RUNNER_DEFAULTS: Dict[str, object] = {
    "iterations": DEFAULT_ITERATION,
    "resolution": DEFAULT_RESOLUTION,
    "validation_root": "outputs/validation",
    "benchmark_protocol": True,
    "check_runtime_deps": True,
    "auto_patch_readers": True,
    "aggregate_after_eval": True,
    "force": False,
    "min_free_gb": 5,
    "cuda_visible_devices": "",
    "cuda_home": "/usr/local/cuda-11.8",
    "torch_cuda_arch_list": "8.6",
    "max_jobs": 8,
}

RUNNER_ONLY_KEYS = {
    "validation_root",
    "benchmark_protocol",
    "check_runtime_deps",
    "auto_patch_readers",
    "aggregate_after_eval",
    "force",
    "min_free_gb",
    "cuda_visible_devices",
}


def main() -> int:
    while True:
        choice = show_main_menu()
        if choice == "self_check":
            run_self_check()
            pause()
            continue
        if choice == "metrics":
            run_metrics_aggregation_menu()
            continue
        if choice == "resume":
            run_resume_menu()
            continue
        if choice == "resources":
            run_resource_status_menu()
            continue
        if choice == "catalog":
            run_method_catalog_menu()
            continue
        if choice == "datasets":
            run_dataset_path_menu()
            continue
        if choice == "exit":
            print("Bye. No job was started.")
            return 0
        method = choice
        datasets = show_dataset_menu()
        if datasets is None:
            continue
        params: Optional[Dict[str, object]] = None
        while True:
            params = configure_parameters(method, datasets, params)
            if params is None:
                break
            if not select_gpu(params):
                continue
            run_jobs(method, datasets, params)
            pause()
            break


def show_main_menu() -> MethodOption:
    while True:
        print_header("Unified 3DGS Framework")
        print_welcome()
        print()
        print("Choose a method:")
        print()
        for method in METHODS:
            print(f"  {method.roman:<7} {method.title:<36} [{method.key}]")
        action_start = len(METHODS) + 1
        actions = (
            ("catalog", "Method setup / capability status"),
            ("datasets", "Dataset paths"),
            ("resume", "Resume full validation experiments"),
            ("metrics", "Metrics aggregation"),
            ("self_check", "Environment / method self-check"),
            ("resources", "System resource status"),
            ("exit", "Exit"),
        )
        for offset, (_, title) in enumerate(actions):
            print(f"  {to_roman(action_start + offset):<7} {title}")
        print()
        raw = input("Input Roman numeral: ").strip().upper()
        for method in METHODS:
            if raw in {method.roman, str(METHODS.index(method) + 1), method.key.upper()}:
                return method
        for offset, (key, _) in enumerate(actions):
            number = action_start + offset
            aliases = {to_roman(number), str(number)}
            if key == "exit":
                aliases.update({"Q", "QUIT", "EXIT"})
            if raw in aliases:
                return key  # type: ignore[return-value]
        print_error(f"Unknown option. Try I through {to_roman(action_start + len(actions) - 1)}.")


def load_method_catalog() -> List[Dict[str, str]]:
    data = json.loads(METHOD_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Method catalog must be a non-empty list: {METHOD_CATALOG_PATH}")
    return data


def run_dataset_path_menu() -> None:
    while True:
        print_header("Dataset Paths")
        definitions = load_dataset_definitions()
        local_paths = load_local_dataset_paths()
        print("Configure dataset roots used by the menu, validation scripts,")
        print("preflight checks, and dataset write guard.")
        print()
        print(f"Local config file: {LOCAL_DATASET_PATHS}")
        print("Priority: local menu config > environment variable > configs/datasets.json")
        print()
        for index, definition in enumerate(definitions, 1):
            root = dataset_root(definition)
            source = "local config" if definition.key in local_paths else (
                f"env {definition.env_var}" if os.environ.get(definition.env_var) else "default config"
            )
            exists = "exists" if root.is_dir() else "missing"
            print(f"  {to_roman(index):<5} {definition.title}")
            print(f"        key:    {definition.key}")
            print(f"        root:   {root}")
            print(f"        source: {source}; {exists}")
            print(f"        scenes: {', '.join(definition.scenes)}")
        print()
        print("  S     Set or change a dataset root")
        print("  C     Clear a local dataset root override")
        print("  V     Validate configured dataset folders")
        print("  0     Return to main menu")
        print()
        raw = input("Dataset path action: ").strip().upper()
        if raw in {"0", "BACK"}:
            return
        if raw == "S":
            selected = select_dataset_definition(definitions)
            if selected is None:
                continue
            value = input(f"New root for {selected.key}: ").strip()
            if not value:
                print_error("Path was empty; no change made.")
                continue
            save_local_dataset_path(selected.key, Path(value))
            print(f"Saved {selected.key} root to {LOCAL_DATASET_PATHS}")
            pause()
            continue
        if raw == "C":
            selected = select_dataset_definition(definitions)
            if selected is None:
                continue
            clear_local_dataset_path(selected.key)
            print(f"Cleared local override for {selected.key}.")
            pause()
            continue
        if raw == "V":
            validate_dataset_roots(definitions)
            pause()
            continue
        print_error("Choose S, C, V, or 0.")


def select_dataset_definition(definitions):
    raw = input("Dataset key / Roman numeral / 0: ").strip().upper()
    if raw in {"0", "BACK"}:
        return None
    for index, definition in enumerate(definitions, 1):
        if raw in {to_roman(index), str(index), definition.key.upper()}:
            return definition
    print_error("Unknown dataset.")
    return None


def validate_dataset_roots(definitions) -> None:
    print_header("Dataset Path Validation")
    ok = True
    for definition in definitions:
        root = dataset_root(definition)
        print(f"{definition.key}: {root}")
        if not root.is_dir():
            ok = False
            print("  ERROR: root does not exist")
            continue
        for scene in definition.scenes:
            scene_root = root / scene
            status = "ok" if scene_root.is_dir() else "missing"
            print(f"  {scene:<12} {status}  {scene_root}")
            if not scene_root.is_dir():
                ok = False
    if ok:
        print("All configured dataset scene folders exist.")
    else:
        print("One or more configured dataset scene folders are missing.")


def run_method_catalog_menu() -> None:
    methods = load_method_catalog()
    while True:
        print_header("Method Setup And Capabilities")
        confirmed_count = sum(
            1
            for method in methods
            if method.get("source_status") == "confirmed" and method.get("repository")
        )
        print(f"{confirmed_count} confirmed repositories are registered in the top-level menu.")
        print("Run setup/preflight/metrics acceptance before publishing new results.")
        print("This page manages setup, preflight, and PSNR/SSIM/LPIPS acceptance")
        print("evidence across Mip-NeRF 360, Tanks & Temples, and Deep Blending.")
        print()
        for index, method in enumerate(methods, 1):
            source = "repo confirmed" if method["source_status"] == "confirmed" else "source pending"
            print(
                f"  {to_roman(index):<5} {method['title']:<34} "
                f"[{method['key']}] {source}"
            )
        print()
        print("  A     Print safe server clone commands")
        print("  C     Clone all confirmed repositories")
        print("  B     Build detected method CUDA/C++ extensions in isolation")
        print("  P     Preflight all confirmed methods (no training)")
        print("  S     Verify train-render-eval metrics with iterations=1")
        print("  V     Audit historical metrics acceptance")
        print("  E     Audit static render/evaluate blockers")
        print("  0     Return to main menu")
        print()
        raw = input("Method / action: ").strip()
        upper = raw.upper()
        if upper in {"0", "BACK"}:
            return
        if upper == "A":
            run_checked(
                [sys.executable, "scripts/manage_method_repositories.py", "commands", "--all"],
                env=build_env(RUNNER_DEFAULTS),
            )
            pause()
            continue
        if upper == "C":
            confirmation = input(
                "Clone all confirmed repositories under third_party/? [Y/N]: "
            ).strip().lower()
            if confirmation in {"y", "yes"}:
                run_checked(
                    [sys.executable, "scripts/manage_method_repositories.py", "clone", "--all"],
                    env=build_env(RUNNER_DEFAULTS),
                )
                pause()
            continue
        if upper == "P":
            status = subprocess.run(
                [sys.executable, "scripts/check_method_preflight.py"],
                cwd=PROJECT_ROOT,
                env=build_env(RUNNER_DEFAULTS),
            ).returncode
            print(f"Method preflight exit code: {status}")
            pause()
            continue
        if upper == "B":
            confirmation = input(
                "Build extensions and preflight all confirmed methods? [Y/N]: "
            ).strip().lower()
            if confirmation not in {"y", "yes"}:
                continue
            status = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_method_acceptance.py",
                    "--run-real",
                ],
                cwd=PROJECT_ROOT,
                env=build_env(RUNNER_DEFAULTS),
            ).returncode
            print(f"Method build + preflight exit code: {status}")
            pause()
            continue
        if upper == "S":
            acceptance_params = dict(RUNNER_DEFAULTS)
            acceptance_params["cuda_visible_devices"] = os.environ.get(
                "CUDA_VISIBLE_DEVICES", ""
            )
            if not select_gpu(acceptance_params):
                continue
            print()
            print("Each method will run train -> render -> eval on three dataset")
            print("families with iterations=1, and must produce finite PSNR, SSIM,")
            print("and LPIPS before it is accepted.")
            confirmation = input(
                "Verify all confirmed metrics interfaces now? [Y/N]: "
            ).strip().lower()
            if confirmation not in {"y", "yes"}:
                continue
            status = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_method_metrics.py",
                    "--run-real",
                ],
                cwd=PROJECT_ROOT,
                env=build_env(acceptance_params),
            ).returncode
            print(f"Metrics acceptance exit code: {status}")
            pause()
            continue
        if upper == "V":
            status = subprocess.run(
                [sys.executable, "scripts/audit_method_metrics_acceptance.py"],
                cwd=PROJECT_ROOT,
                env=build_env(RUNNER_DEFAULTS),
            ).returncode
            print(f"Metrics acceptance audit exit code: {status}")
            pause()
            continue
        if upper == "E":
            status = subprocess.run(
                [sys.executable, "scripts/audit_method_e2e_readiness.py"],
                cwd=PROJECT_ROOT,
                env=build_env(RUNNER_DEFAULTS),
            ).returncode
            print(f"Static render/evaluate blocker audit exit code: {status}")
            pause()
            continue

        selected: Optional[Dict[str, str]] = None
        for index, method in enumerate(methods, 1):
            if upper in {to_roman(index), str(index), method["key"].upper()}:
                selected = method
                break
        if selected is None:
            print_error("Choose a method, A, B, C, P, S, V, E, or 0.")
            continue

        print_header(selected["title"])
        print(f"Key:          {selected['key']}")
        print(f"Venue:        {selected['venue']}")
        print(f"Source status: {selected['source_status']}")
        print(f"Local path:   {selected['local_path']}")
        print(f"Repository:   {selected.get('repository') or 'not publicly confirmed'}")
        print(f"Project page: {selected.get('project_page') or 'not recorded'}")
        if selected.get("repository"):
            print()
            print("Server clone command:")
            print(
                f'git clone --recursive "{selected["repository"]}" '
                f'"{selected["local_path"]}"'
            )
        pause()


def show_dataset_menu() -> Optional[List[DatasetFamily]]:
    datasets = load_dataset_families()
    while True:
        print_header("Dataset Selection")
        print("Choose one or more dataset families. Examples: A, AC, A C, ALL")
        print("Input 0 to return to the method menu.")
        print()
        for family in datasets:
            scene_labels = ", ".join(label.split("/", 1)[1] for _, label in family.scenes)
            print(f"  {family.letter}) {family.title}")
            print(f"     {scene_labels}")
        print()
        raw = input("Datasets: ").strip().upper().replace(",", " ")
        if raw in {"0", "B", "BACK"}:
            # B is a dataset letter, so only exact BACK or 0 returns.
            if raw == "B":
                return [datasets[1]]
            return None
        if raw == "ALL":
            return list(datasets)
        tokens = raw.split() if " " in raw else list(raw)
        selected: List[DatasetFamily] = []
        letters = {family.letter: family for family in datasets}
        invalid: List[str] = []
        for token in tokens:
            if token in letters and letters[token] not in selected:
                selected.append(letters[token])
            elif token:
                invalid.append(token)
        if invalid:
            print_error(f"Unknown dataset option(s): {', '.join(invalid)}")
            continue
        if selected:
            return selected
        print_error("Select at least one dataset family.")


def configure_parameters(
    method: MethodOption,
    datasets: Sequence[DatasetFamily],
    existing: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    if existing is None:
        config = load_config(PROJECT_ROOT / method.config)
        params: Dict[str, object] = {}
        params.update(config)
        params.update({key: value for key, value in RUNNER_DEFAULTS.items() if key not in params})
    else:
        params = dict(existing)

    while True:
        print_parameter_summary(method, datasets, params)
        print("Accept these parameters?")
        print("  Y = start training")
        print("  N = edit parameters")
        print("  0 = return to dataset menu")
        raw = input("Choice [Y/N/0]: ").strip().lower()
        if raw in {"y", "yes", ""}:
            return params
        if raw in {"0", "b", "back"}:
            return None
        if raw in {"n", "no"}:
            edit_parameters(params)
            continue
        print_error("Please input Y, N, or 0.")


def edit_parameters(params: Dict[str, object]) -> None:
    editable_keys = sorted(params)
    while True:
        print_header("Edit Parameters")
        print("Type key=value to edit. Type show to list values, done to continue, 0 to cancel editing.")
        print("Common examples: iterations=30000, resolution=-1, cuda_visible_devices=5, min_free_gb=5")
        print()
        raw = input("edit> ").strip()
        if raw.lower() in {"done", "d"}:
            return
        if raw in {"0", "back"}:
            return
        if raw.lower() == "show":
            print_key_values(params)
            continue
        if "=" not in raw:
            print_error("Use key=value.")
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            print_error("Empty key.")
            continue
        if key not in editable_keys:
            print("New parameter key. It will be passed as a --set override when supported.")
        params[key] = parse_scalar(value.strip())
        print(f"Set {key} = {params[key]!r}")


def run_jobs(
    method: MethodOption,
    datasets: Sequence[DatasetFamily],
    params: Dict[str, object],
) -> None:
    if method.key in CATALOG_METHOD_KEYS:
        run_catalog_training_jobs(method, datasets, params)
        return

    validation_root = Path(str(params.get("validation_root") or DEFAULT_VALIDATION_ROOT))
    iterations = int(params.get("iterations") or DEFAULT_ITERATION)
    total_stages = sum(len(family.scenes) for family in datasets) * 3
    completed = 0

    print_header("Preflight")
    env = build_env(params)
    if bool_value(params.get("check_runtime_deps")):
        run_checked([sys.executable, "scripts/check_runtime_dependencies.py"], env=env)
        run_checked([sys.executable, "scripts/check_dataset_write_guard.py"], env=env)

    if bool_value(params.get("auto_patch_readers")):
        run_checked([sys.executable, "scripts/patch_third_party_readers.py", "--method", method.key], env=env)
        if method.key == "3dhgs":
            run_checked([sys.executable, "scripts/patch_3dhgs_render_gt.py"], env=env)

    if method.key == "3dgs_mcmc":
        run_checked(
            [
                sys.executable,
                "scripts/prepare_mcmc_scene_configs.py",
                "--validation-root",
                str(validation_root),
                "--iterations",
                str(params["iterations"]),
                "--resolution",
                str(params["resolution"]),
            ],
            env=env,
        )
    if method.key == "sss":
        run_checked(
            [
                sys.executable,
                "scripts/prepare_sss_scene_configs.py",
                "--validation-root",
                str(validation_root),
                "--iterations",
                str(params["iterations"]),
                "--resolution",
                str(params["resolution"]),
            ],
            env=env,
        )

    run_checked(
        [
            sys.executable,
            "scripts/check_method_scene_readiness.py",
            "--methods",
            method.key,
            "--validation-root",
            str(validation_root),
            "--iterations",
            str(params["iterations"]),
            "--resolution",
            str(params["resolution"]),
        ],
        env=env,
    )

    print_header("Training")
    for family in datasets:
        for data_path, label in family.scenes:
            pair_root = validation_root / method.key / label
            output = pair_root / "method_outputs"
            scene_params = dict(params)
            actual_iteration = iterations
            if bool_value(params.get("force")):
                reset_pair_for_clean_rerun(pair_root, validation_root)
            pair_root.mkdir(parents=True, exist_ok=True)
            output.mkdir(parents=True, exist_ok=True)

            for stage in ("train", "render", "eval"):
                if stage in {"render", "eval"}:
                    actual_iteration = iterations
                    scene_params["render_iteration"] = actual_iteration
                    print(f"Resolved final iteration for {method.key}/{label}: {actual_iteration}")
                marker = pair_root / f".{stage}.done"
                log_path = pair_root / f"{stage}.log"
                if (
                    not marker.exists()
                    and not bool_value(params.get("force"))
                    and recover_stage_marker(stage, output, actual_iteration, marker, env)
                ):
                    if stage in {"train", "render"}:
                        verify_effective_protocol(
                            stage, output, data_path, label, actual_iteration, env
                        )
                    completed += 1
                    print_progress(completed, total_stages, f"recover {method.key}/{label}/{stage}")
                    continue
                if marker.exists() and not bool_value(params.get("force")):
                    if stage in {"train", "render"}:
                        verify_effective_protocol(
                            stage, output, data_path, label, actual_iteration, env
                        )
                    completed += 1
                    print_progress(completed, total_stages, f"skip {method.key}/{label}/{stage}")
                    continue

                cmd = build_stage_command(
                    stage=stage,
                    method=method,
                    data_path=data_path,
                    output=output,
                    label=label,
                    params=scene_params,
                    validation_root=validation_root,
                )
                print_progress(completed, total_stages, f"run {method.key}/{label}/{stage}")
                status = run_streaming(cmd, log_path, env)
                if status != 0:
                    print_error(f"Stage failed: {method.key}/{label}/{stage}. See {log_path}")
                    raise SystemExit(status)

                verify_stage(stage, output, actual_iteration, env)
                if stage in {"train", "render"}:
                    verify_effective_protocol(
                        stage, output, data_path, label, actual_iteration, env
                    )
                marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
                completed += 1
                print_progress(completed, total_stages, f"done {method.key}/{label}/{stage}")

                if stage == "eval" and bool_value(params.get("aggregate_after_eval")):
                    aggregate(validation_root, env, actual_iteration)

    aggregate(validation_root, env, actual_iteration)
    audit(method, datasets, validation_root, env)
    print_header("Complete")
    print(f"Results: {validation_root / method.key}")
    print(f"Metrics: {validation_root / method.key / 'metrics_summary.md'}")
    print(f"Audit:   {validation_root / 'benchmark_protocol_audit.csv'}")


def run_catalog_training_jobs(
    method: MethodOption,
    datasets: Sequence[DatasetFamily],
    params: Dict[str, object],
) -> None:
    iterations = int(params.get("iterations") or DEFAULT_ITERATION)
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    root = Path(str(params.get("validation_root") or DEFAULT_VALIDATION_ROOT))
    env = build_env(params)
    scenes = [
        (data_path, label)
        for family in datasets
        for data_path, label in family.scenes
    ]
    print_header("Unified Train / Render / Evaluate")
    print(f"Method: {method.title} [{method.key}]")
    print(f"Requested iterations: {iterations}")
    print("Required result: paired test renders plus finite PSNR, SSIM, and LPIPS")
    print(f"Selected scenes: {len(scenes)}")

    completed = 0
    total_stages = len(scenes) * 3
    for data_path, label in scenes:
        pair_root = root / method.key / label
        output_root = pair_root / "method_outputs"
        if bool_value(params.get("force")):
            reset_pair_for_clean_rerun(pair_root, root)
        pair_root.mkdir(parents=True, exist_ok=True)
        for stage in ("train", "render", "eval"):
            marker = pair_root / f".{stage}.done"
            log_path = pair_root / f"{stage}.log"
            if (
                not marker.is_file()
                and not bool_value(params.get("force"))
                and recover_stage_marker(
                    stage,
                    output_root,
                    iterations,
                    marker,
                    env,
                )
            ):
                completed += 1
                print_progress(
                    completed,
                    total_stages,
                    f"recover {method.key}/{label}/{stage}",
                )
                continue
            if marker.is_file() and not bool_value(params.get("force")):
                completed += 1
                print_progress(
                    completed,
                    total_stages,
                    f"skip {method.key}/{label}/{stage}",
                )
                continue
            cmd = build_stage_command(
                stage,
                method,
                data_path,
                output_root,
                label,
                params,
                root,
            )
            print_progress(
                completed,
                total_stages,
                f"run {method.key}/{label}/{stage}",
            )
            status = run_streaming(cmd, log_path, env)
            if status != 0:
                print_error(
                    f"Stage failed: {method.key}/{label}/{stage}. See {log_path}"
                )
                raise SystemExit(status)
            verify_stage(stage, output_root, iterations, env)
            marker.write_text(
                time.strftime("%Y-%m-%d %H:%M:%S") + "\n",
                encoding="utf-8",
            )
            completed += 1
            print_progress(
                completed,
                total_stages,
                f"done {method.key}/{label}/{stage}",
            )
            if stage == "eval" and bool_value(params.get("aggregate_after_eval")):
                aggregate(root, env, iterations)

    aggregate(root, env, iterations)
    print_header("Unified Workflow Complete")
    print(f"Results: {root / method.key}")
    print(f"Metrics: {root / method.key / 'metrics_summary.md'}")


def reset_pair_for_clean_rerun(pair_root: Path, validation_root: Path) -> None:
    resolved_root = validation_root.expanduser().resolve()
    resolved_pair = pair_root.expanduser().resolve()
    try:
        relative = resolved_pair.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"refusing to clean output outside validation root: {resolved_pair}"
        ) from exc
    if len(relative.parts) < 3 or resolved_pair == resolved_root:
        raise ValueError(f"refusing unsafe validation cleanup target: {resolved_pair}")
    if pair_root.is_symlink():
        raise ValueError(f"refusing to clean symlinked validation output: {pair_root}")
    if pair_root.exists():
        print(f"Force clean rerun: removing {pair_root}")
        shutil.rmtree(pair_root)


def build_stage_command(
    stage: str,
    method: MethodOption,
    data_path: str,
    output: Path,
    label: str,
    params: Dict[str, object],
    validation_root: Path,
) -> List[str]:
    config = method.config
    common = [
        "--method",
        method.key,
        "--config",
        config,
        "--data",
        data_path,
        "--output",
        str(output),
    ]
    if stage == "train":
        cmd = [sys.executable, "train_all.py"] + common
    elif stage == "render":
        cmd = [sys.executable, "render_all.py"] + common
    elif stage == "eval":
        cmd = [sys.executable, "eval_all.py"] + common
    else:
        raise ValueError(stage)

    overrides = stage_overrides(stage, method, label, params, validation_root)
    for key, value in overrides:
        cmd.extend(["--set", f"{key}={format_override_value(value)}"])
    return cmd


def stage_overrides(
    stage: str,
    method: MethodOption,
    label: str,
    params: Dict[str, object],
    validation_root: Path,
) -> List[Tuple[str, object]]:
    overrides: Dict[str, object] = {}
    if stage in {"train", "render"}:
        overrides["images"] = official_images_for_label(label)
        overrides["resolution"] = params.get("resolution", DEFAULT_RESOLUTION)
    if stage == "train":
        overrides["iterations"] = params.get("iterations", DEFAULT_ITERATION)
        overrides["test_iterations"] = params.get("test_iterations", -1)
        overrides["dataset_label"] = label
        if method.key in {"3dgs_mcmc", "sss"}:
            overrides["generated_scene_config_dir"] = validation_root / "generated_configs" / method.key
    if stage == "render":
        overrides["render_iteration"] = params.get(
            "render_iteration",
            params.get("iterations", DEFAULT_ITERATION),
        )
    if stage == "eval":
        overrides["render_iteration"] = params.get(
            "render_iteration",
            params.get("iterations", DEFAULT_ITERATION),
        )
        overrides["timeout_seconds"] = params.get("timeout_seconds", 0)
        overrides["heartbeat_seconds"] = params.get("heartbeat_seconds", 30)

    for key, value in params.items():
        if key in RUNNER_ONLY_KEYS:
            continue
        if key in {"dataset_path", "output_path", "method", "third_party_repo"}:
            continue
        if key in overrides:
            continue
        if stage == "eval" and key not in {
            "render_iteration",
            "timeout_seconds",
            "heartbeat_seconds",
        }:
            continue
        overrides[key] = value
    return list(overrides.items())


def official_images_for_label(label: str) -> str:
    if label in {"mip360/bicycle", "mip360/flowers", "mip360/garden", "mip360/stump", "mip360/treehill"}:
        return "images_4"
    if label in {"mip360/room", "mip360/counter", "mip360/kitchen", "mip360/bonsai"}:
        return "images_2"
    return "images"


def build_env(params: Dict[str, object]) -> Dict[str, str]:
    env = os.environ.copy()
    cuda_home = str(params.get("cuda_home") or "/usr/local/cuda-11.8")
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = cuda_home
    env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["TORCH_CUDA_ARCH_LIST"] = str(params.get("torch_cuda_arch_list") or "8.6")
    env["MAX_JOBS"] = str(params.get("max_jobs") or 8)
    gpu = str(params.get("cuda_visible_devices") or "").strip()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["UNIFIED3DGS_READONLY_DATASET_ROOT"] = readonly_dataset_root_env()
    return env


def verify_stage(stage: str, output: Path, iterations: int, env: Dict[str, str]) -> None:
    run_checked(
        [
            sys.executable,
            "scripts/verify_scene_outputs.py",
            "--stage",
            stage,
            "--output",
            str(output),
            "--iteration",
            str(iterations),
        ],
        env=env,
    )


def verify_effective_protocol(
    stage: str,
    output: Path,
    data_path: str,
    label: str,
    iterations: int,
    env: Dict[str, str],
) -> None:
    family, scene = label.split("/", 1)
    run_checked(
        [
            sys.executable,
            "scripts/verify_effective_protocol.py",
            "--stage",
            stage,
            "--output",
            str(output),
            "--data",
            data_path,
            "--family",
            family,
            "--scene",
            scene,
            "--iteration",
            str(iterations),
        ],
        env=env,
    )


def recover_stage_marker(
    stage: str,
    output: Path,
    iterations: int,
    marker: Path,
    env: Dict[str, str],
) -> bool:
    cmd = [
        sys.executable,
        "scripts/verify_scene_outputs.py",
        "--stage",
        stage,
        "--output",
        str(output),
        "--iteration",
        str(iterations),
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    print(f"Recovered done marker from valid existing output: {marker}")
    return True


def aggregate(
    validation_root: Path,
    env: Dict[str, str],
    iteration: int = DEFAULT_ITERATION,
) -> None:
    run_checked(
        [
            sys.executable,
            "scripts/aggregate_metrics.py",
            "--validation-root",
            str(validation_root),
            "--iteration",
            str(iteration),
        ],
        env=env,
    )


def run_metrics_aggregation_menu() -> None:
    options = {
        "A": (("scene",), "Scene-Level summaries"),
        "B": (("dataset",), "Dataset-Level averages"),
        "C": (("method",), "Method-Level averages"),
        "D": (("scene", "dataset", "method"), "All three levels"),
    }
    while True:
        print_header("Metrics Aggregation")
        print("  A) Scene-Level: individual method/dataset/scene metrics")
        print("  B) Dataset-Level: average over scenes in each dataset")
        print("  C) Method-Level: equal-weight average over dataset-family averages")
        print("  D) All three levels")
        print("  0) Return to main menu")
        print()
        raw = input("Aggregation level: ").strip().upper()
        if raw in {"0", "BACK"}:
            return
        if raw not in options:
            print_error("Choose A, B, C, D, or 0.")
            continue

        root_raw = input("Validation root [outputs/validation]: ").strip()
        validation_root = Path(root_raw or "outputs/validation")
        levels, title = options[raw]
        cmd = [
            sys.executable,
            "scripts/aggregate_metrics.py",
            "--validation-root",
            str(validation_root),
            "--iteration",
            str(DEFAULT_ITERATION),
            "--levels",
            *levels,
        ]
        print(f"$ {' '.join(cmd)}")
        run_checked(cmd, env=build_env(RUNNER_DEFAULTS))
        print(f"{title} written under: {validation_root}")
        pause()
        return


def run_resume_menu() -> None:
    while True:
        print_header("Resume Incomplete Experiments")
        print("This resumes the formal 5-method x 13-scene matrix.")
        print("Verified completed stages are reused; missing or failed stages run again.")
        print("All metric CSV files are rebuilt after completion.")
        print("Existing results are read from: outputs/validation")
        print()
        print("  Y) Continue")
        print("  0) Return to main menu")
        raw = input("Choice [Y/0]: ").strip().lower()
        if raw in {"0", "back"}:
            return
        if raw not in {"", "y", "yes"}:
            print_error("Choose Y or 0.")
            continue

        validation_root = "outputs/validation"
        resume_params = dict(RUNNER_DEFAULTS)
        resume_params["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not select_gpu(resume_params):
            continue
        env = build_env(resume_params)
        env.update(
            {
                "RUN_REAL": "1",
                "VALIDATION_ROOT": validation_root,
            }
        )
        print()
        print(f"Validation root: {validation_root}")
        print("Starting resumable full validation. Press Ctrl-C to stop safely.")
        print("Rerun this menu option later to continue from verified outputs.")
        run_checked(["bash", "scripts/resume_all_experiments.sh"], env=env)
        pause()
        return


def query_gpu_status() -> Tuple[List[str], List[str], str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return [], [], "nvidia-smi is not available in PATH."
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed."
        return [], [], message

    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    indices = [row.split(",", 1)[0].strip() for row in rows]
    return indices, rows, ""


def print_welcome() -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "not active")
    print("Welcome. Ready to manage Unified 3DGS experiments.")
    print(f"Time: {timestamp}")
    print(f"User / Host: {getpass.getuser()}@{socket.gethostname()}")
    print(f"Conda environment: {conda_env}")
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"GitHub repository: {PROJECT_REPOSITORY_URL}")


def format_bytes(value: int) -> str:
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} PiB"


def memory_status() -> List[Tuple[str, str]]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return [("System memory", "Unavailable on this platform")]

    values: Dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) * 1024

    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    return [
        ("RAM total", format_bytes(total)),
        ("RAM available", format_bytes(available)),
        ("RAM used estimate", format_bytes(max(total - available, 0))),
        ("Swap total", format_bytes(swap_total)),
        ("Swap free", format_bytes(swap_free)),
    ]


def directory_size(path: Path) -> str:
    if not path.exists():
        return "not present"
    try:
        result = subprocess.run(
            ["du", "-sh", str(path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"
    if result.returncode != 0 or not result.stdout.strip():
        return "unavailable"
    return result.stdout.split()[0]


def disk_status(path: Path) -> List[Tuple[str, str]]:
    usage = shutil.disk_usage(path)
    used_percent = (usage.used / usage.total * 100.0) if usage.total else 0.0
    return [
        ("Filesystem total", format_bytes(usage.total)),
        ("Filesystem used", f"{format_bytes(usage.used)} ({used_percent:.1f}%)"),
        ("Filesystem free", format_bytes(usage.free)),
        ("Project directory size", directory_size(PROJECT_ROOT)),
        (
            "Validation outputs size",
            directory_size(PROJECT_ROOT / "outputs" / "validation"),
        ),
    ]


def run_resource_status_menu() -> None:
    while True:
        print_header("System Resource Status")
        print(f"Checked at: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Current project directory: {PROJECT_ROOT}")
        print()
        print("Memory:")
        for label, value in memory_status():
            print(f"  {label:<25} {value}")
        print()
        print("Current Project Filesystem / Directory Usage:")
        for label, value in disk_status(PROJECT_ROOT):
            print(f"  {label:<25} {value}")
        print()
        print("GPU:")
        _, rows, error = query_gpu_status()
        if rows:
            print("  Index | Name | Total MiB | Free MiB | Used MiB | Util % | Temp C")
            print("  " + "-" * 82)
            for row in rows:
                print("  " + " | ".join(part.strip() for part in row.split(",")))
        else:
            print_error(error or "Unable to query GPU status.")
        print()
        print("Input R to refresh or 0 to return to the main menu.")
        raw = input("Choice [R/0]: ").strip().upper()
        if raw in {"0", "BACK"}:
            return
        if raw in {"", "R", "REFRESH"}:
            continue
        print_error("Choose R or 0.")


def select_gpu(params: Dict[str, object]) -> bool:
    while True:
        print_header("GPU Selection")
        indices, rows, error = query_gpu_status()
        if rows:
            print("Index | Name | Total MiB | Free MiB | Used MiB | Util % | Temp C")
            print("-" * 84)
            for row in rows:
                print(" | ".join(part.strip() for part in row.split(",")))
        else:
            print_error(error or "Unable to query GPU status.")
            indices = []

        current = str(
            params.get("cuda_visible_devices")
            or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ).strip()
        print()
        print("Input one GPU index, for example 5.")
        print("Multiple comma-separated indices are accepted when intentionally needed.")
        print("Input R to refresh GPU status, or B to return without starting training.")
        prompt = f"GPU [{current}]: " if current else "GPU: "
        raw = input(prompt).strip()
        if raw.upper() == "R":
            continue
        if raw.upper() in {"B", "BACK"}:
            return False
        selected = raw or current
        if not selected:
            print_error("Select a GPU before starting training.")
            continue
        selected_indices = [item.strip() for item in selected.split(",") if item.strip()]
        if not selected_indices or any(not item.isdigit() for item in selected_indices):
            print_error("GPU selection must contain numeric indices, for example 5 or 5,6.")
            continue
        if indices:
            invalid = [item for item in selected_indices if item not in indices]
            if invalid:
                print_error(f"GPU index not reported by nvidia-smi: {', '.join(invalid)}")
                continue
        params["cuda_visible_devices"] = ",".join(selected_indices)
        print(f"Selected physical GPU(s): {params['cuda_visible_devices']}")
        print("Inside the launched process, the first selected GPU is visible as cuda:0.")
        return True


def audit(
    method: MethodOption,
    datasets: Sequence[DatasetFamily],
    validation_root: Path,
    env: Dict[str, str],
) -> None:
    cmd = [
        sys.executable,
        "scripts/audit_benchmark_protocol.py",
        "--validation-root",
        str(validation_root),
        "--output",
        str(validation_root / "benchmark_protocol_audit.csv"),
        "--methods",
        method.key,
        "--dataset-families",
    ]
    cmd.extend(family.key for family in datasets)
    run_checked(cmd, env=env)


def run_self_check() -> None:
    print_header("Environment / Method Self-Check")
    env = build_env(RUNNER_DEFAULTS)
    commands = [
        [sys.executable, "scripts/check_runtime_dependencies.py"],
        [sys.executable, "scripts/check_dataset_write_guard.py"],
        [sys.executable, "scripts/check_method_scene_readiness.py", "--methods"]
        + [method.key for method in EXTENDED_WORKFLOW_METHODS],
        [sys.executable, "scripts/check_method_preflight.py"],
    ]
    for cmd in commands:
        print(f"$ {' '.join(cmd)}")
        status = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
        if status != 0:
            print_error(f"Self-check command failed with exit code {status}.")
            return
    acceptance_root = PROJECT_ROOT / "outputs" / "validation" / "_method_acceptance"
    if any(acceptance_root.glob("attempts/*/metrics_acceptance_results.json")):
        cmd = [sys.executable, "scripts/audit_method_metrics_acceptance.py"]
        print(f"$ {' '.join(cmd)}")
        status = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
        if status != 0:
            print_error(
                "Method metrics acceptance audit failed "
                f"with exit code {status}."
            )
            return
    else:
        print_error(
            "Method metrics acceptance evidence is absent. "
            "Use the Method setup / capability status menu and run action S."
        )
        return
    print("Self-check passed.")


def run_checked(cmd: Sequence[str], env: Dict[str, str]) -> None:
    status = subprocess.run(list(cmd), cwd=PROJECT_ROOT, env=env).returncode
    if status != 0:
        raise SystemExit(status)


def run_streaming(cmd: Sequence[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(str(part) for part in cmd)}")
    print(f"Log: {log_path}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            list(cmd),
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def print_parameter_summary(
    method: MethodOption,
    datasets: Sequence[DatasetFamily],
    params: Dict[str, object],
) -> None:
    print_header("Parameter Review")
    print(f"Method: {method.title} [{method.key}]")
    print("Datasets:")
    for family in datasets:
        print(f"  {family.letter}) {family.title}")
        for _, label in family.scenes:
            print(f"     - {label}, images={official_images_for_label(label)}")
    print()
    print("Special notes:")
    for note in method.notes:
        print(f"  - {note}")
    print()
    print_key_values(params)


def print_key_values(values: Dict[str, object]) -> None:
    for key in sorted(values):
        print(f"  {key:<28} {values[key]!r}")
    print()


def print_progress(done: int, total: int, label: str) -> None:
    width = 30
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{bar}] {done:>3}/{total:<3} {label}")


def print_header(title: str) -> None:
    line = "=" * 72
    print()
    print(line)
    print(title.center(72))
    print(line)


def print_error(message: str) -> None:
    print(f"ERROR: {message}")


def pause() -> None:
    input("Press Enter to continue...")


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def format_override_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


if __name__ == "__main__":
    try:
        os.chdir(PROJECT_ROOT)
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")
        raise SystemExit(130)
