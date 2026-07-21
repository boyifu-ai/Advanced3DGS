from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _last_json_object(output: str) -> Dict[str, object]:
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return dict(payload)
    raise ValueError("backend probe emitted no valid JSON object")


@dataclass
class BackendCheck:
    method: str
    python: Path
    environment: Dict[str, str]
    runtime: Dict[str, object]
    errors: List[str]
    official: bool

    @property
    def passed(self) -> bool:
        return not self.errors


def _backend_environment(
    backend: Dict[str, object], project_root: Path, base: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    env = dict(base or os.environ)
    cuda_home = str(backend.get("cuda_home", "")).strip()
    if cuda_home:
        cuda_path = Path(cuda_home).expanduser()
        if not cuda_path.is_absolute():
            cuda_path = project_root / cuda_path
        cuda_home = str(cuda_path.resolve())
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home
        current_path = env.get("PATH", "")
        env["PATH"] = (
            f"{cuda_home}/bin:{current_path}" if current_path else f"{cuda_home}/bin"
        )
        current = env.get("LD_LIBRARY_PATH", "")
        lib_paths = [f"{cuda_home}/lib64", f"{cuda_home}/lib"]
        env["LD_LIBRARY_PATH"] = (
            ":".join(lib_paths + ([current] if current else []))
        )
    return env


def _backend_python(
    backend: Dict[str, object], project_root: Path
) -> Path:
    env_name = str(backend.get("python_env_var", "")).strip()
    configured = os.environ.get(env_name, "").strip() if env_name else ""
    value = configured or str(backend.get("default_python", "")).strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _probe_runtime(python: Path, env: Dict[str, str]) -> Dict[str, object]:
    code = (
        "import json, sys, torch; "
        "print(json.dumps({"
        "'python': '.'.join(str(v) for v in sys.version_info[:3]),"
        "'python_executable': sys.executable,"
        "'torch': torch.__version__,"
        "'torch_cuda': torch.version.cuda,"
        "'cuda_available': torch.cuda.is_available()"
        "}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "official backend probe failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return _last_json_object(completed.stdout)


def _matches(actual: object, expected: object) -> bool:
    actual_text = str(actual or "")
    expected_text = str(expected or "")
    return (
        actual_text == expected_text
        or actual_text.startswith(expected_text + "+")
        or actual_text.startswith(expected_text + ".")
    )


def check_official_backend(
    method: str,
    profile: Dict[str, object],
    project_root: Path,
) -> BackendCheck:
    backend = profile.get("official_backend")
    if not isinstance(backend, dict):
        return BackendCheck(
            method=method,
            python=Path(sys.executable).resolve(),
            environment=os.environ.copy(),
            runtime={},
            errors=[],
            official=False,
        )

    python = _backend_python(backend, project_root)
    env = _backend_environment(backend, project_root)
    errors: List[str] = []
    runtime: Dict[str, object] = {}
    if not python.is_file():
        errors.append(
            f"official backend Python is missing: {python}. "
            f"Run: bash scripts/setup_{method}_official_backend.sh"
        )
    else:
        try:
            runtime = _probe_runtime(python, env)
        except Exception as exc:
            errors.append(str(exc))

    requirements = backend.get("requirements", {})
    if runtime and isinstance(requirements, dict):
        fields = {
            "python": "python",
            "torch": "torch",
            "torch_cuda": "torch_cuda",
        }
        for requirement, runtime_field in fields.items():
            expected = requirements.get(requirement)
            if expected is not None and not _matches(
                runtime.get(runtime_field), expected
            ):
                errors.append(
                    f"official backend {requirement} mismatch: "
                    f"expected {expected}, got {runtime.get(runtime_field)}"
                )
        if runtime.get("cuda_available") is not True:
            errors.append("official backend cannot access CUDA")

    return BackendCheck(
        method=method,
        python=python,
        environment=env,
        runtime=runtime,
        errors=errors,
        official=True,
    )


def activate_official_backend(
    method: str,
    profile: Dict[str, object],
    project_root: Path,
    script: Path,
    argv: Sequence[str],
) -> BackendCheck:
    check = check_official_backend(method, profile, project_root)
    if not check.official or not check.passed:
        return check

    active = os.environ.get("UNIFIED3DGS_OFFICIAL_BACKEND", "")
    if active == method:
        return check

    env = dict(check.environment)
    env["UNIFIED3DGS_OFFICIAL_BACKEND"] = method
    os.execve(
        str(check.python),
        [str(check.python), str(script)] + list(argv),
        env,
    )
    raise AssertionError("os.execve returned unexpectedly")


def classify_failure(
    output: str,
    backend: BackendCheck,
    official_protocol: bool,
) -> Dict[str, object]:
    lowered = output.lower()
    if backend.errors:
        return {
            "category": "environment_mismatch",
            "objective_limit": False,
            "reason": "; ".join(backend.errors),
        }

    program_markers = (
        "integer multiplication overflow",
        "illegal memory access",
        "indexerror",
        "shape of the mask",
        "permissionerror",
        "modulenotfounderror",
        "nan",
    )
    marker = next((value for value in program_markers if value in lowered), None)
    if marker:
        return {
            "category": "program_error",
            "objective_limit": False,
            "reason": f"detected program/runtime failure marker: {marker}",
        }

    oom = re.search(
        r"tried to allocate\s+([0-9.]+)\s+gib.*?"
        r"([0-9.]+)\s+gib total capacity",
        lowered,
        re.DOTALL,
    )
    if oom:
        requested = float(oom.group(1))
        capacity = float(oom.group(2))
        if official_protocol and backend.official and requested > capacity:
            return {
                "category": "hardware_limit_confirmed",
                "objective_limit": True,
                "reason": (
                    f"official runtime requested {requested:.2f} GiB in one "
                    f"allocation on a {capacity:.2f} GiB GPU"
                ),
                "requested_gib": requested,
                "capacity_gib": capacity,
            }
        return {
            "category": "resource_or_program_error",
            "objective_limit": False,
            "reason": (
                "CUDA OOM occurred before both the official runtime and official "
                "protocol were verified"
            ),
            "requested_gib": requested,
            "capacity_gib": capacity,
        }

    return {
        "category": "program_error",
        "objective_limit": False,
        "reason": "training exited without a verified objective hardware limit",
    }
