from __future__ import annotations

import ast
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from unified3dgs.dataset_config import (
    acceptance_dataset,
    readonly_dataset_root_env,
)
from unified3dgs.dataset_overlay import prepare_dataset_overlay
from unified3dgs.method_backend import check_official_backend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "configs" / "method_catalog.json"
PROFILE_PATH = PROJECT_ROOT / "configs" / "method_profiles.json"
DEFAULT_DATASET = acceptance_dataset()
DEFAULT_OUTPUT_ROOT = Path("outputs/validation/_method_acceptance")
DEFAULT_MIN_FREE_DISK_GB = 5.0
IMAGE_ALIASES = ("images", "images_2", "images_4", "images_8")
ROBUST_PLY_PATCH_MARKER = "Unified 3DGS robust PLY reader patch"
NO_DATASET_WRITE_PATCH_MARKER = "Unified 3DGS no dataset-write point cloud patch"

SOURCE_FLAGS = ("--source_path", "--data", "--data_path", "--dataset_path", "-s")
OUTPUT_FLAGS = ("--model_path", "--output", "--output_path", "--save_dir", "-m")
ITERATION_FLAGS = ("--iterations", "--iteration", "--max_steps", "--num_iterations")
SAVE_FLAGS = ("--save_iterations", "--save_iteration", "--checkpoint_iterations")
TEST_FLAGS = ("--test_iterations", "--test_iteration")
EVAL_FLAGS = ("--eval",)
IMAGES_FLAGS = ("--images", "-i")
PROTECTED_RUN_FLAGS = set(
    SOURCE_FLAGS + OUTPUT_FLAGS + ITERATION_FLAGS + SAVE_FLAGS
)


def method_specific_extra_arg_errors(key: str, extra_args: Sequence[object]) -> List[str]:
    flags = {str(value).split("=", 1)[0] for value in extra_args}
    errors: List[str] = []
    if key == "hac_plus" and "--n_offsets" in flags:
        errors.append(
            "hac_plus: extra_args must preserve upstream n_offsets=10; "
            "HAC++ initializes n_offsets + 1 masks but exposes a fixed 10-mask slice"
        )
    return errors


@dataclass
class MethodPreflight:
    key: str
    title: str
    repo: Path
    entry: Optional[Path] = None
    command_flags: Dict[str, Optional[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "repo": str(self.repo),
            "entry": str(self.entry) if self.entry else None,
            "command_flags": self.command_flags,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def load_json_list(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [dict(item) for item in data]


def load_confirmed_catalog() -> List[Dict[str, object]]:
    return [
        method
        for method in load_json_list(CATALOG_PATH)
        if method.get("source_status") == "confirmed" and method.get("repository")
    ]


def load_profiles() -> Dict[str, Dict[str, object]]:
    return {
        str(profile["key"]): profile
        for profile in load_json_list(PROFILE_PATH)
    }


def select_methods(keys: Sequence[str]) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    catalog = load_confirmed_catalog()
    profiles = load_profiles()
    catalog_by_key = {str(method["key"]): method for method in catalog}
    selected_keys = list(keys) if keys else sorted(catalog_by_key)
    unknown = sorted(set(selected_keys) - set(catalog_by_key))
    if unknown:
        raise ValueError(f"Unknown confirmed method(s): {', '.join(unknown)}")
    missing_profiles = sorted(set(selected_keys) - set(profiles))
    if missing_profiles:
        raise ValueError(f"Missing acceptance profile(s): {', '.join(missing_profiles)}")
    return [(catalog_by_key[key], profiles[key]) for key in selected_keys]


def dataset_family(dataset: Path, label: Optional[str] = None) -> Optional[str]:
    if label and "/" in label:
        return label.split("/", 1)[0]
    parts = {part.lower() for part in dataset.parts}
    scene_name = dataset.name.lower()
    if {"mip360", "mipnerf360"} & parts:
        return "mip360"
    if "tandt" in parts or scene_name in {"train", "truck"}:
        return "tandt"
    if "deep_blending" in parts or scene_name in {"drjohnson", "playroom"}:
        return "deep_blending"
    return None


def dataset_scene_label(dataset: Path, label: Optional[str] = None) -> Optional[str]:
    if label and "/" in label:
        return label
    family = dataset_family(dataset)
    if family is None:
        return None
    return f"{family}/{dataset.name}"


def official_dataset_args(
    profile: Dict[str, object], dataset: Path, label: Optional[str] = None
) -> List[str]:
    family = dataset_family(dataset, label)
    label = dataset_scene_label(dataset, label)
    scene_configured = profile.get("official_scene_args", {})
    if label and isinstance(scene_configured, dict):
        scene_values = scene_configured.get(label, [])
        if not isinstance(scene_values, list):
            raise ValueError(
                f"official_scene_args[{label!r}] must be a list"
            )
        if scene_values:
            return [str(value) for value in scene_values]
    configured = profile.get("official_dataset_args", {})
    if family is None or not isinstance(configured, dict):
        return []
    values = configured.get(family, [])
    if not isinstance(values, list):
        raise ValueError(
            f"official_dataset_args[{family!r}] must be a list"
        )
    return [str(value) for value in values]


def resolve_project_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_method_env(
    key: str, repo: Path, runtime_namespace: str = "method_acceptance"
) -> Dict[str, str]:
    env = os.environ.copy()
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda-11.8")
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = env.get("CUDA_PATH", cuda_home)
    env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "8.6")
    env["MAX_JOBS"] = env.get("MAX_JOBS", "8")
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["WANDB_CONSOLE"] = "off"
    legacy_allocator_config = env.pop("PYTORCH_CUDA_ALLOC_CONF", "")
    env["PYTORCH_ALLOC_CONF"] = env.get(
        "PYTORCH_ALLOC_CONF", legacy_allocator_config or "max_split_size_mb:128"
    )
    env["UNIFIED3DGS_READONLY_DATASET_ROOT"] = readonly_dataset_root_env()
    if key == "beta_splatting":
        env["UNIFIED3DGS_PY38_FUNCTOOLS_CACHE"] = "1"
    if key == "octree_gs":
        env["UNIFIED3DGS_NUMPY_LEGACY_ALIASES"] = "1"
    profile = load_profiles().get(key, {})
    if profile.get("colmap_sparse_zero_fallback") is True:
        env["UNIFIED3DGS_COLMAP_SPARSE_ZERO_FALLBACK"] = "1"
    runtime_root = (
        PROJECT_ROOT / "third_party_build" / "runtime" / runtime_namespace / key
    )
    shared_torch_home = (
        PROJECT_ROOT
        / "third_party_build"
        / "runtime"
        / runtime_namespace
        / "shared_torch_cache"
    )
    shared_torch_home.mkdir(parents=True, exist_ok=True)
    env["TORCH_HOME"] = str(shared_torch_home)
    runtime_paths = {
        "XDG_CACHE_HOME": runtime_root / "cache",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
        "TORCH_EXTENSIONS_DIR": runtime_root / "torch_extensions",
        "PIP_CACHE_DIR": runtime_root / "pip_cache",
        "TMPDIR": runtime_root / "tmp",
        "TMP": runtime_root / "tmp",
        "TEMP": runtime_root / "tmp",
    }
    for name, path in runtime_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    paths = [
        str((PROJECT_ROOT / "unified3dgs" / "runtime_guard").resolve()),
        str((PROJECT_ROOT / "third_party_build" / key / "site-packages").resolve()),
        str(repo),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def first_present_flag(help_text: str, candidates: Iterable[str]) -> Optional[str]:
    for flag in candidates:
        if re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text):
            return flag
    return None


def first_declared_flag(
    declared: Iterable[str], candidates: Iterable[str]
) -> Optional[str]:
    declared_set = set(declared)
    for flag in candidates:
        if flag in declared_set:
            return flag
    return None


def required_flags_from_usage(help_text: str) -> List[str]:
    usage_lines: List[str] = []
    collecting = False
    for line in help_text.splitlines():
        if line.lower().startswith("usage:"):
            collecting = True
        elif collecting and line and not line[0].isspace():
            break
        if collecting:
            usage_lines.append(line)
    usage = " ".join(usage_lines)
    previous = None
    while previous != usage:
        previous = usage
        usage = re.sub(r"\[[^\[\]]*\]", " ", usage)
    return sorted(set(re.findall(r"(?<![\w-])--[a-zA-Z0-9_-]+", usage)))


def choose_entry(repo: Path, candidates: Sequence[object]) -> Optional[Path]:
    for candidate in candidates:
        path = repo / str(candidate)
        if path.is_file():
            return path
    return None


def static_cli_text(repo: Path, entry: Path) -> str:
    preferred: List[Path] = [entry]
    for arguments_dir in (repo / "arguments", entry.parent / "arguments"):
        if arguments_dir.is_dir():
            preferred.extend(sorted(arguments_dir.rglob("*.py")))
    seen = set()
    texts: List[str] = []
    for path in preferred:
        try:
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "add_argument" in text or "ArgumentParser" in text:
            texts.append(text)
    return "\n".join(texts)


def static_required_flags(repo: Path, entry: Path) -> List[str]:
    preferred: List[Path] = [entry]
    for arguments_dir in (repo / "arguments", entry.parent / "arguments"):
        if arguments_dir.is_dir():
            preferred.extend(sorted(arguments_dir.rglob("*.py")))
    required = set()
    seen = set()
    for path in preferred:
        try:
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else (
                function.id if isinstance(function, ast.Name) else ""
            )
            if name != "add_argument":
                continue
            is_required = any(
                keyword.arg == "required"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            if not is_required:
                continue
            for argument in node.args:
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.startswith("--")
                ):
                    required.add(argument.value)
    return sorted(required)


def static_declared_flags(repo: Path, entry: Path) -> List[str]:
    preferred: List[Path] = [entry]
    for arguments_dir in (repo / "arguments", entry.parent / "arguments"):
        if arguments_dir.is_dir():
            preferred.extend(sorted(arguments_dir.rglob("*.py")))
    declared = set()
    seen = set()
    for path in preferred:
        try:
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        has_argument_registration = any(
            isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                )
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "add_argument"
                )
            )
            for node in ast.walk(tree)
        )
        if not has_argument_registration:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else function.id if isinstance(function, ast.Name) else ""
                )
                if name == "add_argument":
                    for argument in node.args:
                        if (
                            isinstance(argument, ast.Constant)
                            and isinstance(argument.value, str)
                            and argument.value.startswith("-")
                        ):
                            declared.add(argument.value)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        option = target.attr.lstrip("_")
                        if option:
                            declared.add(f"--{option}")
    return sorted(declared)


def has_standard_3dgs_arguments(repo: Path, entry: Path) -> bool:
    return any(
        (root / "arguments" / "__init__.py").is_file()
        for root in (repo, entry.parent)
    )


def concise_failure(output: str, limit: int = 8) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "no error output captured"
    return " | ".join(lines[-limit:])


def last_json_value(output: str) -> Any:
    """Return the last complete JSON line while ignoring runtime warnings."""
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"probe emitted no valid JSON payload: {concise_failure(output)}")


def signature_value_matches(name: str, actual: object, expected: object) -> bool:
    if name == "python_executable":
        try:
            return Path(str(actual)).resolve() == Path(str(expected)).resolve()
        except (OSError, RuntimeError):
            return str(actual) == str(expected)
    return actual == expected


def resolve_local_module(search_roots: Sequence[Path], module: str) -> Optional[Path]:
    parts = module.split(".")
    for root in search_roots:
        file_candidate = root.joinpath(*parts).with_suffix(".py")
        if file_candidate.is_file():
            return file_candidate
        package_candidate = root.joinpath(*parts) / "__init__.py"
        if package_candidate.is_file():
            return package_candidate
    return None


def resolve_relative_module(path: Path, level: int, module: Optional[str]) -> Optional[Path]:
    root = path.parent
    for _ in range(max(0, level - 1)):
        root = root.parent
    target = root.joinpath(*(module or "").split("."))
    if target.with_suffix(".py").is_file():
        return target.with_suffix(".py")
    if (target / "__init__.py").is_file():
        return target / "__init__.py"
    return None


def entry_external_imports(repo: Path, entry: Path) -> List[str]:
    search_roots = (repo, entry.parent)
    pending = [entry]
    visited = set()
    external = set()
    while pending and len(visited) < 500:
        path = pending.pop()
        try:
            resolved = path.resolve()
            if resolved in visited:
                continue
            visited.add(resolved)
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = resolve_local_module(search_roots, alias.name)
                    if local is not None:
                        pending.append(local)
                    else:
                        external.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative_candidates = [node.module] if node.module else []
                    relative_candidates.extend(
                        f"{node.module}.{alias.name}" if node.module else alias.name
                        for alias in node.names
                        if alias.name != "*"
                    )
                    for candidate in relative_candidates:
                        local = resolve_relative_module(path, node.level, candidate)
                        if local is not None:
                            pending.append(local)
                    continue
                if node.module:
                    local = resolve_local_module(search_roots, node.module)
                    if local is None:
                        external.add(node.module.split(".", 1)[0])
                        continue
                    pending.append(local)
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        imported = resolve_local_module(
                            search_roots, f"{node.module}.{alias.name}"
                        )
                        if imported is not None:
                            pending.append(imported)
    return sorted(external)


def run_capture(
    command: Sequence[str],
    cwd: Path,
    env: Dict[str, str],
    timeout: int,
) -> Tuple[int, str]:
    def output_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = output_text(exc.stdout) + "\n" + output_text(exc.stderr)
        return 124, output.strip()
    except OSError as exc:
        return 127, repr(exc)
    output = output_text(result.stdout) + "\n" + output_text(result.stderr)
    return result.returncode, output.strip()


def normalize_repository_url(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized[len("git@github.com:") :]
    normalized = normalized.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def git_details(repo: Path, result: MethodPreflight, expected_repository: str) -> None:
    if not (repo / ".git").exists():
        result.errors.append("repository is missing Git metadata")
        return
    code, output = run_capture(["git", "rev-parse", "HEAD"], repo, os.environ.copy(), 15)
    if code != 0:
        result.errors.append(f"cannot resolve repository commit: {output}")
    else:
        result.details["commit"] = output.splitlines()[-1]
    code, output = run_capture(
        ["git", "remote", "get-url", "origin"], repo, os.environ.copy(), 15
    )
    if code != 0:
        result.errors.append(f"cannot resolve origin repository: {output}")
    else:
        origin = output.splitlines()[-1].rstrip("/")
        result.details["origin"] = origin
        if normalize_repository_url(origin) != normalize_repository_url(expected_repository):
            result.errors.append(
                f"origin mismatch: expected {expected_repository}, got {origin}"
            )
    code, output = run_capture(
        ["git", "submodule", "status", "--recursive"], repo, os.environ.copy(), 120
    )
    if code != 0:
        result.errors.append(f"cannot inspect submodules: {output}")
    else:
        missing = [line for line in output.splitlines() if line.startswith("-")]
        conflicts = [line for line in output.splitlines() if line.startswith("+")]
        if missing:
            result.errors.append(f"uninitialized submodules: {missing}")
        if conflicts:
            result.warnings.append(f"submodules differ from recorded commits: {conflicts}")
        result.details["submodules"] = output.splitlines()


def extension_sources(
    repo: Path, prefixes: Optional[Sequence[object]] = None
) -> Tuple[List[str], List[str]]:
    found: List[str] = []
    modules: List[str] = []
    normalized_prefixes = [
        str(prefix).replace("\\", "/").lstrip("./") for prefix in prefixes or ()
    ]
    for setup in repo.rglob("setup.py"):
        relative = setup.relative_to(repo)
        if len(relative.parts) > 5:
            continue
        source = relative.parent.as_posix()
        if normalized_prefixes and not any(
            source.startswith(prefix) for prefix in normalized_prefixes
        ):
            continue
        try:
            text = setup.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "CUDAExtension" in text or "CppExtension" in text:
            found.append(source)
            for match in re.finditer(r"""name\s*=\s*["']([^"']+)["']""", text):
                modules.append(match.group(1).replace("-", "_"))
    return sorted(set(found)), sorted(set(modules))


def catalog_extension_spec(
    repo: Path, profile: Dict[str, object]
) -> Tuple[List[str], List[str], List[str]]:
    prefixes = profile.get("extension_prefixes", [])
    if not isinstance(prefixes, list):
        prefixes = []
    sources, modules = extension_sources(repo, prefixes)
    errors: List[str] = []
    archives = profile.get("archived_extensions", [])
    if not isinstance(archives, list):
        return sources, modules, ["profile archived_extensions must be a list"]
    for item in archives:
        if not isinstance(item, dict) or not item.get("archive"):
            errors.append("archived extension entry must contain an archive path")
            continue
        relative = str(item["archive"]).replace("\\", "/").lstrip("./")
        archive = repo / relative
        label = f"archive:{relative}"
        sources.append(label)
        if not archive.is_file():
            errors.append(f"archived extension is missing: {archive}")
        archive_modules = item.get("modules", [])
        if not isinstance(archive_modules, list) or not archive_modules:
            errors.append(f"archived extension has no declared modules: {relative}")
        else:
            modules.extend(str(module) for module in archive_modules)
    external_extensions = profile.get("external_extensions", [])
    if not isinstance(external_extensions, list):
        errors.append("profile external_extensions must be a list")
        external_extensions = []
    for item in external_extensions:
        if not isinstance(item, dict) or not item.get("path"):
            errors.append("external extension entry must contain a project-relative path")
            continue
        relative = str(item["path"]).replace("\\", "/").lstrip("./")
        source = resolve_project_path(relative)
        label = f"external:{relative}"
        sources.append(label)
        if not (source / "setup.py").is_file():
            errors.append(f"external extension has no setup.py: {source}")
        external_modules = item.get("modules", [])
        if not isinstance(external_modules, list) or not external_modules:
            errors.append(f"external extension has no declared modules: {relative}")
        else:
            modules.extend(str(module) for module in external_modules)
    aliases = profile.get("module_aliases", [])
    if not isinstance(aliases, list):
        errors.append("profile module_aliases must be a list")
        aliases = []
    for item in aliases:
        if (
            not isinstance(item, dict)
            or not item.get("source")
            or not item.get("target")
        ):
            errors.append("module alias entry must contain source and target modules")
            continue
        target = str(item["target"])
        modules.append(target)
        submodules = item.get("submodules", [])
        if not isinstance(submodules, list):
            errors.append(f"module alias submodules must be a list: {target}")
            continue
        modules.extend(f"{target}.{submodule}" for submodule in submodules)
    return sorted(set(sources)), sorted(set(modules)), errors


def extension_import_script(modules: Sequence[str]) -> str:
    names = list(modules)
    return (
        "import torch, importlib; "
        f"names={names!r}; "
        "loaded=[]; "
        "[loaded.append(importlib.import_module(name).__name__) for name in names]; "
        "print('extension imports:', ', '.join(names))"
    )


def runtime_signature_values(python: Path, env: Dict[str, str]) -> Dict[str, object]:
    code = (
        "import json, sys, torch; "
        "print(json.dumps({"
        "'python_executable': sys.executable,"
        "'python_version': sys.version,"
        "'torch_version': torch.__version__,"
        "'torch_cuda': torch.version.cuda,"
        "'cuda_available': torch.cuda.is_available()"
        "}, sort_keys=True))"
    )
    exit_code, output = run_capture([str(python), "-c", code], PROJECT_ROOT, env, 30)
    if exit_code != 0:
        raise RuntimeError(concise_failure(output))
    payload = last_json_value(output)
    if not isinstance(payload, dict):
        raise ValueError(f"runtime signature probe returned {type(payload).__name__}, expected object")
    return dict(payload)


def detect_dataset_write_risks(repo: Path) -> List[str]:
    risks: List[str] = []
    patterns = (
        re.compile(r"source_path\s*[,/+]"),
        re.compile(r"os\.path\.join\([^)]*source_path"),
        re.compile(r"Path\([^)]*source_path"),
    )
    for path in repo.rglob("*.py"):
        relative = path.relative_to(repo)
        if len(relative.parts) > 5 or any(part.startswith(".") for part in relative.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not any(token in text for token in ("write", "save", "storePly", "mkdir", "makedirs")):
            continue
        if any(pattern.search(text) for pattern in patterns):
            risks.append(relative.as_posix())
    return sorted(set(risks))[:30]


def undefined_scene_callback_names(reader: Path) -> List[str]:
    if not reader.is_file():
        return []
    tree = ast.parse(reader.read_text(encoding="utf-8"), filename=str(reader))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    callbacks: Set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "sceneLoadTypeCallbacks"
            for target in targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            callbacks.update(
                value.id for value in node.value.values if isinstance(value, ast.Name)
            )
    return sorted(callbacks - defined)


def reader_safety_errors(reader: Path, method_key: str = "<method>") -> List[str]:
    if not reader.is_file():
        return []
    text = reader.read_text(encoding="utf-8", errors="replace")
    repair = (
        "Repair with: python scripts/patch_third_party_readers.py "
        f"--method {method_key}"
    )
    errors: List[str] = []
    if "def fetchPly(" in text and ROBUST_PLY_PATCH_MARKER not in text:
        errors.append(f"standard dataset reader is missing the robust PLY patch. {repair}")
    if (
        "points3D.ply" in text
        and "storePly(" in text
        and (
            NO_DATASET_WRITE_PATCH_MARKER not in text
            or "pcd = fetchColmapPointCloud" not in text
        )
    ):
        errors.append(
            "standard dataset reader may write generated point clouds into the "
            f"dataset or contains an incomplete no-write patch. {repair}"
        )
    try:
        missing_callbacks = undefined_scene_callback_names(reader)
    except Exception as exc:
        errors.append(f"cannot validate scene reader syntax and callbacks: {exc}")
    else:
        if missing_callbacks:
            errors.append(
                "sceneLoadTypeCallbacks references undefined function(s): "
                + ", ".join(missing_callbacks)
                + f". {repair}"
            )
    return errors


def system_errors(dataset: Path, output_root: Path) -> List[str]:
    errors: List[str] = []
    try:
        output_root.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        errors.append(f"acceptance output root must stay inside the project: {output_root}")
    if sys.version_info[:2] != (3, 8):
        errors.append(f"expected Python 3.8, got {sys.version.split()[0]}")
    try:
        import torch

        if not torch.cuda.is_available():
            errors.append("PyTorch CUDA is unavailable")
        if torch.version.cuda != "11.8":
            errors.append(f"expected PyTorch CUDA 11.8, got {torch.version.cuda}")
    except Exception as exc:
        errors.append(f"cannot import/check torch: {exc!r}")
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-11.8"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        errors.append(f"CUDA compiler is missing: {cuda_home / 'bin' / 'nvcc'}")
    for tool in ("git", "gcc", "g++", "cmake", "ninja", "timeout"):
        if shutil.which(tool) is None:
            errors.append(f"required tool is missing from PATH: {tool}")
    if not dataset.is_dir():
        errors.append(f"acceptance-test dataset is missing: {dataset}")
    else:
        missing_image_aliases = [
            name for name in IMAGE_ALIASES if not (dataset / name).is_dir()
        ]
        if missing_image_aliases:
            errors.append(
                "catalog acceptance compatibility dataset is missing low-resolution "
                "image aliases required to prevent method-specific full-resolution "
                f"loads: {missing_image_aliases}"
            )
        if not (dataset / "sparse" / "0").is_dir():
            errors.append(f"dataset has no COLMAP sparse/0 directory: {dataset}")
        else:
            sparse = dataset / "sparse" / "0"
            for name in ("cameras.bin", "images.bin"):
                if not (sparse / name).is_file():
                    errors.append(f"dataset is missing required COLMAP file: {sparse / name}")
            if not any(
                (sparse / name).is_file()
                for name in ("points3D.ply", "points3D.bin", "points3D.txt")
            ):
                errors.append(
                    f"dataset has no usable initial point cloud under: {sparse}"
                )
        for name in ("cameras.bin", "images.bin", "points3D.ply"):
            direct = dataset / "sparse" / name
            if not direct.is_file():
                errors.append(
                    "catalog acceptance compatibility dataset is missing root-level "
                    f"sparse file required by some methods: {direct}"
                )
        point_cloud = dataset / "sparse" / "0" / "points3D.ply"
        if point_cloud.is_file():
            try:
                from plyfile import PlyData

                names = set(PlyData.read(str(point_cloud))["vertex"].data.dtype.names or ())
                required = {"x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"}
                missing = sorted(required - names)
                if missing:
                    errors.append(
                        "catalog acceptance point cloud lacks compatibility fields "
                        f"{missing}: {point_cloud}"
                    )
            except Exception as exc:
                errors.append(
                    f"cannot validate catalog acceptance point-cloud fields: {exc!r}"
                )
    output_parent = output_root if output_root.exists() else output_root.parent
    if not output_parent.exists() or not os.access(str(output_parent), os.W_OK):
        errors.append(f"acceptance output parent is not writable: {output_parent}")
    try:
        free = shutil.disk_usage(output_parent).free
        if free < DEFAULT_MIN_FREE_DISK_GB * 1024**3:
            errors.append(
                f"less than {DEFAULT_MIN_FREE_DISK_GB:g} GiB free for acceptance outputs: "
                f"{output_parent}"
            )
    except OSError as exc:
        errors.append(f"cannot inspect output filesystem: {exc}")
    for script in ("check_runtime_dependencies.py", "check_dataset_write_guard.py"):
        code, output = run_capture(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script)],
            PROJECT_ROOT,
            os.environ.copy(),
            120,
        )
        if code != 0:
            errors.append(f"{script} failed before method training: {output}")
    return errors


def preflight_method(
    method: Dict[str, object],
    profile: Dict[str, object],
    dataset: Path,
) -> MethodPreflight:
    key = str(method["key"])
    repo = resolve_project_path(method["local_path"])
    result = MethodPreflight(key=key, title=str(method["title"]), repo=repo)
    official_backend = profile.get("official_backend")
    if isinstance(official_backend, dict):
        extension_rebuild_hint = f"bash scripts/setup_{key}_official_backend.sh"
    else:
        extension_rebuild_hint = (
            f"python scripts/install_catalog_method_extensions.py "
            f"--method {key} --run-real"
        )
    if not repo.is_dir():
        result.errors.append(f"repository is missing: {repo}")
        return result

    git_details(repo, result, str(method["repository"]))
    readmes = sorted(
        path.relative_to(repo).as_posix()
        for path in repo.glob("README*")
        if path.is_file()
    )
    licenses = sorted(
        path.relative_to(repo).as_posix()
        for path in repo.glob("LICENSE*")
        if path.is_file()
    )
    manifests = sorted(
        path.relative_to(repo).as_posix()
        for pattern in ("requirements*.txt", "environment*.yml", "environment*.yaml", "pyproject.toml")
        for path in repo.glob(pattern)
        if path.is_file()
    )
    result.details["readmes"] = readmes
    result.details["licenses"] = licenses
    result.details["dependency_manifests"] = manifests
    if not readmes:
        result.warnings.append("repository has no top-level README")
    if not licenses:
        result.warnings.append("repository has no top-level LICENSE file")
    if not manifests:
        result.warnings.append("repository has no top-level dependency manifest")
    candidates = profile.get("entry_candidates", [])
    if not isinstance(candidates, list):
        result.errors.append("profile entry_candidates must be a list")
        return result
    result.entry = choose_entry(repo, candidates)
    if result.entry is None:
        result.errors.append(
            "no supported training entry found; checked: "
            + ", ".join(str(item) for item in candidates)
        )
        return result

    try:
        compile(result.entry.read_text(encoding="utf-8"), str(result.entry), "exec")
    except Exception as exc:
        result.errors.append(f"training entry does not compile: {exc}")

    reader = repo / "scene" / "dataset_readers.py"
    reader_errors = reader_safety_errors(reader, key)
    result.details["reader_safety_errors"] = reader_errors
    result.errors.extend(reader_errors)
    try:
        missing_callbacks = undefined_scene_callback_names(reader)
    except Exception:
        missing_callbacks = []
        reader_callbacks_ready = False
    else:
        reader_callbacks_ready = not missing_callbacks
    result.details["undefined_scene_callbacks"] = missing_callbacks
    result.details["reader_callbacks_ready"] = reader_callbacks_ready

    env = build_method_env(key, repo)
    runtime_python = Path(sys.executable).resolve()
    backend_check = check_official_backend(key, profile, PROJECT_ROOT)
    if backend_check.official:
        result.details["official_backend"] = {
            "python": str(backend_check.python),
            "runtime": backend_check.runtime,
            "errors": backend_check.errors,
        }
        if backend_check.errors:
            result.errors.extend(backend_check.errors)
        else:
            runtime_python = backend_check.python
            for name in ("CUDA_HOME", "CUDA_PATH", "PATH", "LD_LIBRARY_PATH"):
                if name in backend_check.environment:
                    env[name] = backend_check.environment[name]
    result.details["runtime_python"] = str(runtime_python)
    if not backend_check.official:
        try:
            shared_runtime = runtime_signature_values(runtime_python, env)
            result.details["shared_runtime"] = shared_runtime
            expected_runtime = {
                "python_version": "3.8",
                "torch_version": "2.0.0",
                "torch_cuda": "11.8",
            }
            mismatches = [
                name
                for name, expected in expected_runtime.items()
                if not str(shared_runtime.get(name, "")).startswith(expected)
            ]
            if shared_runtime.get("cuda_available") is not True:
                mismatches.append("cuda_available")
            if mismatches:
                result.errors.append(
                    "shared framework runtime does not match environment.yml for: "
                    + ", ".join(mismatches)
                    + ". Activate the unified-3dgs Conda environment before using "
                    "the menu or rebuilding method extensions."
                )
        except Exception as exc:
            result.errors.append(
                "cannot validate the shared framework runtime: "
                f"{exc}. Activate the unified-3dgs Conda environment."
            )

    external_imports = entry_external_imports(repo, result.entry)
    result.details["entry_external_imports"] = external_imports
    import_probe = (
        "import importlib.util, json; "
        f"names={external_imports!r}; "
        "missing=[]; "
        "\nfor name in names:\n"
        "    try:\n"
        "        found = importlib.util.find_spec(name) is not None\n"
        "    except Exception:\n"
        "        found = False\n"
        "    if not found: missing.append(name)\n"
        "print(json.dumps(missing))"
    )
    import_code, import_output = run_capture(
        [str(runtime_python), "-c", import_probe], result.entry.parent, env, 60
    )
    if import_code != 0:
        result.errors.append(f"static external-import probe failed: {concise_failure(import_output)}")
    else:
        try:
            missing_imports = last_json_value(import_output)
            if not isinstance(missing_imports, list):
                raise ValueError("external-import probe did not return a JSON list")
        except Exception:
            missing_imports = []
            result.errors.append(
                f"could not parse external-import probe output: {concise_failure(import_output)}"
            )
        result.details["missing_external_imports"] = missing_imports
        if missing_imports:
            result.errors.append(
                "missing external Python modules reachable from training entry: "
                + ", ".join(missing_imports)
            )

    static_text = static_cli_text(repo, result.entry)
    static_options = static_declared_flags(repo, result.entry)
    result.details["static_declared_cli_options"] = static_options
    static_help_only = profile.get("static_help_only", False)
    if not isinstance(static_help_only, bool):
        result.errors.append("profile static_help_only must be a boolean")
        static_help_only = False
    if static_help_only:
        reason = profile.get("static_help_reason", "")
        if not isinstance(reason, str) or not reason.strip():
            result.errors.append(
                "profile static_help_only requires a non-empty static_help_reason"
            )
            reason = "unspecified upstream side effect"
        runtime_help_text = ""
        help_text = static_text
        result.details["help_exit_code"] = None
        result.details["help_failure"] = ""
        result.details["help_tail"] = ""
        result.details["static_help_only"] = True
        result.details["static_help_reason"] = reason
        result.warnings.append(
            "runtime --help skipped because of declared upstream side effect: "
            f"{reason}; static CLI scan and real iteration=1 execution remain required"
        )
    else:
        code, runtime_help_text = run_capture(
            [str(runtime_python), str(result.entry), "--help"], result.entry.parent, env, 60
        )
        help_text = runtime_help_text if code == 0 else runtime_help_text + "\n" + static_text
        result.details["help_exit_code"] = code
        result.details["help_failure"] = concise_failure(runtime_help_text) if code != 0 else ""
        result.details["help_tail"] = runtime_help_text[-4000:]
        if code != 0:
            result.errors.append(
                f"training entry --help failed with exit {code}: "
                f"{concise_failure(runtime_help_text)}"
            )

    flags = {
        "source": first_present_flag(help_text, SOURCE_FLAGS),
        "output": first_present_flag(help_text, OUTPUT_FLAGS),
        "iterations": first_present_flag(help_text, ITERATION_FLAGS),
        "save": first_present_flag(help_text, SAVE_FLAGS),
        "test": first_present_flag(help_text, TEST_FLAGS),
        "eval": first_present_flag(help_text, EVAL_FLAGS),
        "images": first_present_flag(help_text, IMAGES_FLAGS),
    }
    declared_fallbacks = {
        "source": SOURCE_FLAGS,
        "output": OUTPUT_FLAGS,
        "iterations": ITERATION_FLAGS,
        "save": SAVE_FLAGS,
        "test": TEST_FLAGS,
        "eval": EVAL_FLAGS,
        "images": IMAGES_FLAGS,
    }
    for name, candidates in declared_fallbacks.items():
        if flags[name] is None:
            flags[name] = first_declared_flag(static_options, candidates)
    if has_standard_3dgs_arguments(repo, result.entry):
        standard_defaults = {
            "source": "-s",
            "output": "-m",
            "iterations": "--iterations",
            "save": "--save_iterations",
        }
        for name, value in standard_defaults.items():
            if flags[name] is None:
                flags[name] = value
        result.details["standard_3dgs_cli_fallback"] = True
    overrides = profile.get("flags", {})
    if isinstance(overrides, dict):
        for name, value in overrides.items():
            if name in flags and value:
                flags[name] = str(value)
    result.command_flags = flags
    for required in ("source", "output", "iterations"):
        if flags[required] is None:
            result.errors.append(
                f"could not resolve required {required} argument from training --help"
            )
    if flags["save"] is None:
        if profile.get("auto_saves_final") is True:
            result.warnings.append(
                "no explicit save-iteration argument; profile declares automatic final save"
            )
        else:
            result.errors.append(
                "no save/checkpoint-iteration argument detected; iteration=1 result saving "
                "cannot be guaranteed"
            )

    extra_args = profile.get("extra_args", [])
    if not isinstance(extra_args, list):
        result.errors.append("profile extra_args must be a list")
        extra_args = []
    result.details["profile_extra_args"] = extra_args
    conflicting_extra_args = sorted(
        {
            str(value)
            for value in extra_args
            if str(value).split("=", 1)[0] in PROTECTED_RUN_FLAGS
        }
    )
    if conflicting_extra_args:
        result.errors.append(
            "profile extra_args overrides protected acceptance-run arguments: "
            + ", ".join(conflicting_extra_args)
        )
    result.errors.extend(method_specific_extra_arg_errors(result.key, extra_args))
    declared_extra_flags = [
        str(value).split("=", 1)[0]
        for value in extra_args
        if str(value).startswith("--")
    ]
    unknown_extra_flags = [
        flag
        for flag in declared_extra_flags
        if flag not in static_options
        and not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text)
    ]
    if unknown_extra_flags:
        result.errors.append(
            "profile extra_args contain option(s) absent from the training CLI: "
            + ", ".join(sorted(set(unknown_extra_flags)))
        )
    result_globs = profile.get("result_globs", [])
    if not isinstance(result_globs, list):
        result.errors.append("profile result_globs must be a list")
    else:
        unsafe_globs = [str(pattern) for pattern in result_globs if "{iteration}" not in str(pattern)]
        if unsafe_globs:
            result.errors.append(
                "profile result_globs must contain {iteration} for exact save verification: "
                + ", ".join(unsafe_globs)
            )
    output_globs = profile.get("output_globs", [])
    result.details["profile_output_globs"] = output_globs
    if not isinstance(output_globs, list):
        result.errors.append("profile output_globs must be a list")
    else:
        unsafe_output_globs = [
            str(pattern)
            for pattern in output_globs
            if not str(pattern).startswith("{output}")
        ]
        if unsafe_output_globs:
            result.errors.append(
                "profile output_globs must start with {output}: "
                + ", ".join(unsafe_output_globs)
            )
    supplied_flags = {
        str(value)
        for value in flags.values()
        if isinstance(value, str) and value.startswith("--")
    }
    supplied_flags.update(
        str(value) for value in extra_args if str(value).startswith("--")
    )
    required_options = sorted(
        set(required_flags_from_usage(runtime_help_text))
        | set(static_required_flags(repo, result.entry))
    )
    result.details["required_cli_options"] = required_options
    unresolved_required = [flag for flag in required_options if flag not in supplied_flags]
    if unresolved_required:
        result.errors.append(
            "training CLI has unresolved required option(s); add them to the acceptance profile: "
            + ", ".join(unresolved_required)
        )

    extension_prefixes = profile.get("extension_prefixes", [])
    if not isinstance(extension_prefixes, list):
        result.errors.append("profile extension_prefixes must be a list")
        extension_prefixes = []
    result.details["extension_prefixes"] = extension_prefixes
    sources, modules, extension_spec_errors = catalog_extension_spec(repo, profile)
    result.errors.extend(extension_spec_errors)
    result.details["cuda_extension_sources"] = sources
    result.details["cuda_extension_modules"] = modules
    extension_target = PROJECT_ROOT / "third_party_build" / key / "site-packages"
    extension_manifest = extension_target / ".unified3dgs_extension_build.json"
    extension_build_ready = not sources
    if sources and not extension_manifest.is_file():
        state = "missing" if not extension_target.is_dir() else "incomplete or unverified"
        result.errors.append(
            f"method has CUDA/C++ extension sources but isolated build target is {state}: "
            f"{extension_target}. Build with: {extension_rebuild_hint}"
        )
    if sources and extension_manifest.is_file():
        extension_build_ready = True
        try:
            manifest = json.loads(extension_manifest.read_text(encoding="utf-8"))
            result.details["extension_build_manifest"] = manifest
            if sorted(manifest.get("sources", [])) != sources:
                extension_build_ready = False
                result.errors.append(
                    "isolated extension build manifest does not match current extension sources; "
                    f"rebuild with: {extension_rebuild_hint}"
                )
            signature = manifest.get("signature")
            if not isinstance(signature, dict):
                extension_build_ready = False
                result.errors.append(
                    "isolated extension build manifest has no environment signature; "
                    f"rebuild with: {extension_rebuild_hint}"
                )
            else:
                try:
                    runtime_values = runtime_signature_values(runtime_python, env)
                    expected_signature = {
                        "repository_commit": result.details.get("commit", ""),
                        "python_executable": runtime_values["python_executable"],
                        "python_version": runtime_values["python_version"],
                        "torch_version": runtime_values["torch_version"],
                        "torch_cuda": runtime_values["torch_cuda"],
                        "cuda_home": env.get("CUDA_HOME", ""),
                        "torch_cuda_arch_list": env.get("TORCH_CUDA_ARCH_LIST", ""),
                        "sources": sources,
                        "modules": modules,
                    }
                    mismatches = sorted(
                        name
                        for name, expected in expected_signature.items()
                        if not signature_value_matches(
                            name, signature.get(name), expected
                        )
                    )
                    if mismatches:
                        extension_build_ready = False
                        result.errors.append(
                            "isolated extension build signature is stale for: "
                            + ", ".join(mismatches)
                            + f". Rebuild with: {extension_rebuild_hint}"
                        )
                except Exception as exc:
                    extension_build_ready = False
                    result.errors.append(
                        "cannot validate isolated extension environment signature: "
                        f"{exc}. Rebuild with: {extension_rebuild_hint}"
                    )
        except Exception as exc:
            extension_build_ready = False
            result.errors.append(f"cannot read isolated extension build manifest: {exc!r}")
    if sources and extension_manifest.is_file() and modules and extension_build_ready:
        import_script = extension_import_script(modules)
        code, output = run_capture([str(runtime_python), "-c", import_script], repo, env, 30)
        if code != 0:
            result.errors.append(
                "isolated CUDA/C++ extension import probe failed: "
                f"{concise_failure(output)}. Rebuild with: {extension_rebuild_hint}. "
                f"The build and training runtime must both use {runtime_python}."
            )
        else:
            result.details["extension_import_probe"] = output

    local_import_modules = profile.get("preflight_import_modules")
    if local_import_modules is None:
        local_import_modules = ["scene.dataset_readers"] if reader.is_file() else []
    if not isinstance(local_import_modules, list):
        result.errors.append("profile preflight_import_modules must be a list")
        local_import_modules = []
    result.details["preflight_import_modules"] = local_import_modules
    if local_import_modules and extension_build_ready and reader_callbacks_ready:
        local_import_script = (
            "import importlib; "
            f"names={list(map(str, local_import_modules))!r}; "
            "[importlib.import_module(name) for name in names]; "
            "print('local imports:', ', '.join(names))"
        )
        code, output = run_capture(
            [str(runtime_python), "-c", local_import_script], repo, env, 60
        )
        if code != 0:
            result.errors.append(
                "training-local module import probe failed before training: "
                + concise_failure(output)
            )
        else:
            result.details["local_module_import_probe"] = output

    risks = detect_dataset_write_risks(repo)
    result.details["dataset_write_risk_files"] = risks
    if risks:
        result.warnings.append(
            "possible dataset-write code detected; runtime write guard will block writes: "
            + ", ".join(risks)
        )

    code, output = run_capture(
        [
            str(runtime_python),
            "-c",
            "from pathlib import Path; "
            f"assert Path({str(dataset)!r}).is_dir(); "
            "import torch; assert torch.cuda.is_available(); "
            "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))",
        ],
        repo,
        env,
        30,
    )
    if code != 0:
        result.errors.append(f"method-isolated Python/CUDA probe failed: {output}")
    else:
        result.details["python_cuda_probe"] = output
    return result


def preflight_all(
    selected: Sequence[Tuple[Dict[str, object], Dict[str, object]]],
    dataset: Path,
    output_root: Path,
) -> Tuple[List[str], List[MethodPreflight]]:
    overlay_errors: List[str] = []
    preflight_dataset = dataset
    try:
        preflight_dataset = prepare_dataset_overlay(dataset, output_root)
    except Exception as exc:
        overlay_errors.append(
            "cannot prepare catalog acceptance dataset overlay: "
            f"{exc!r}"
        )
    global_errors = overlay_errors + system_errors(preflight_dataset, output_root)
    results: List[MethodPreflight] = []
    for method, profile in selected:
        try:
            results.append(preflight_method(method, profile, preflight_dataset))
        except Exception as exc:
            key = str(method["key"])
            repo = resolve_project_path(method["local_path"])
            result = MethodPreflight(key=key, title=str(method["title"]), repo=repo)
            result.errors.append(f"unexpected preflight exception: {exc!r}")
            results.append(result)
    return global_errors, results


def build_acceptance_command(
    result: MethodPreflight,
    dataset: Path,
    output: Path,
    profile: Optional[Dict[str, object]] = None,
) -> List[str]:
    return build_training_command(
        result=result,
        dataset=dataset,
        output=output,
        iterations=1,
        profile=profile,
        images="images_8" if (dataset / "images_8").is_dir() else None,
        eval_enabled=True,
        test_iterations=-1,
        use_acceptance_profile_args=True,
    )


def build_training_command(
    result: MethodPreflight,
    dataset: Path,
    output: Path,
    iterations: int,
    profile: Optional[Dict[str, object]] = None,
    images: Optional[str] = None,
    resolution: Optional[object] = None,
    eval_enabled: bool = False,
    test_iterations: Optional[int] = None,
    extra_args: Sequence[object] = (),
    use_acceptance_profile_args: bool = False,
) -> List[str]:
    if result.entry is None or not result.passed:
        raise ValueError(f"Cannot build training command for failed preflight: {result.key}")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    flags = result.command_flags
    command = [
        str(result.details.get("runtime_python") or sys.executable),
        str(result.entry),
        str(flags["source"]),
        str(dataset),
        str(flags["output"]),
        str(output),
        str(flags["iterations"]),
        str(iterations),
    ]
    if flags.get("save"):
        command.extend([str(flags["save"]), str(iterations)])
    if flags.get("test") and test_iterations is not None:
        command.extend([str(flags["test"]), str(test_iterations)])
    if flags.get("eval") and eval_enabled:
        command.append(str(flags["eval"]))
    if flags.get("images") and images:
        command.extend([str(flags["images"]), images])
    declared = result.details.get("static_declared_cli_options", [])
    if (
        resolution is not None
        and isinstance(declared, list)
        and "--resolution" in declared
    ):
        command.extend(["--resolution", str(resolution)])
    if profile:
        profile_key = "extra_args" if use_acceptance_profile_args else "training_extra_args"
        profile_args = profile.get(profile_key)
        if profile_args is None and not use_acceptance_profile_args:
            profile_args = profile.get("extra_args", [])
        if isinstance(profile_args, list):
            replacements = {
                "{dataset}": str(dataset),
                "{output}": str(output),
                "{iteration}": str(iterations),
            }
            for value in profile_args:
                rendered = str(value)
                for token, replacement in replacements.items():
                    rendered = rendered.replace(token, replacement)
                command.append(rendered)
    command.extend(str(value) for value in extra_args)
    return command


def result_output_roots(
    output: Path,
    profile: Optional[Dict[str, object]] = None,
) -> List[Path]:
    roots = [output]
    if not profile:
        return roots
    patterns = profile.get("output_globs", [])
    if not isinstance(patterns, list):
        return roots
    for pattern in patterns:
        rendered = str(pattern).replace("{output}", str(output))
        for match in glob.glob(rendered):
            candidate = Path(match)
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    return roots


def saved_result_files(
    output: Path,
    expected_iteration: int,
    newer_than: Optional[float] = None,
    extra_globs: Optional[Sequence[str]] = None,
) -> List[Path]:
    if not output.is_dir():
        return []
    preferred: List[Path] = []
    for path in output.rglob("*"):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        if newer_than is not None and path.stat().st_mtime < newer_than:
            continue
        relative = path.relative_to(output).as_posix().lower()
        if relative.endswith("input.ply"):
            continue
        iteration_patterns = (
            rf"(^|/)iteration_{expected_iteration}(/|$)",
            rf"(^|/)(chkpnt|checkpoint)[_-]?{expected_iteration}(\D|$)",
            rf"(^|/){expected_iteration}\.(pt|pth|ckpt)$",
        )
        if any(re.search(pattern, relative) for pattern in iteration_patterns):
            preferred.append(path)
    for pattern in extra_globs or ():
        rendered_pattern = str(pattern).replace("{iteration}", str(expected_iteration))
        for path in output.glob(rendered_pattern):
            if (
                path.is_file()
                and path.stat().st_size > 0
                and (newer_than is None or path.stat().st_mtime >= newer_than)
                and path not in preferred
            ):
                preferred.append(path)
    return sorted(preferred)


def unexpected_iteration_artifacts(
    output: Path, expected_iteration: int, newer_than: Optional[float] = None
) -> List[Path]:
    if not output.is_dir():
        return []
    unexpected: List[Path] = []
    patterns = (
        re.compile(r"^iteration_(\d+)$"),
        re.compile(r"^(?:chkpnt|checkpoint)[_-]?(\d+)(?:\D.*)?$"),
    )
    for path in output.rglob("*"):
        if newer_than is not None and path.stat().st_mtime < newer_than:
            continue
        for pattern in patterns:
            match = pattern.match(path.name.lower())
            if match and int(match.group(1)) > expected_iteration:
                unexpected.append(path)
                break
    return sorted(unexpected)
