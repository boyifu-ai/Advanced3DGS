from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from unified3dgs.dataset_config import readonly_dataset_root_env
from unified3dgs.utils.paths import assert_output_is_safe
from unified3dgs.utils.subprocess_utils import run_command


@dataclass
class MethodRunConfig:
    method: str
    action: str
    dataset_path: Path
    output_path: Path
    config_path: Path
    project_root: Path
    values: Dict[str, object] = field(default_factory=dict)
    dry_run: bool = False
    extra_args: List[str] = field(default_factory=list)


@dataclass
class CommandSpec:
    command: List[str]
    cwd: Path
    env: Dict[str, str]


class BaseMethodAdapter:
    method_name: str = ""
    third_party_repo: str = ""
    train_entry: str = "train.py"
    render_entry: str = "render.py"
    eval_entry: str = "metrics.py"

    def train(self, config: MethodRunConfig) -> int:
        return self._run("train", config)

    def render(self, config: MethodRunConfig) -> int:
        return self._run("render", config)

    def evaluate(self, config: MethodRunConfig) -> int:
        return self._run("evaluate", config)

    def _run(self, action: str, config: MethodRunConfig) -> int:
        assert_output_is_safe(config.output_path)
        spec = self.build_command(action, config)
        if (
            action == "train"
            and not config.dry_run
            and not spec.env.get("CUDA_VISIBLE_DEVICES", "").strip()
        ):
            raise RuntimeError(
                "Refusing to start training without an explicit CUDA_VISIBLE_DEVICES "
                "selection on the shared server."
            )
        if not config.dry_run:
            config.output_path.mkdir(parents=True, exist_ok=True)
        return run_command(spec.command, cwd=spec.cwd, env=spec.env, dry_run=config.dry_run)

    def build_command(self, action: str, config: MethodRunConfig) -> CommandSpec:
        repo_path = self.repo_path(config)
        entry_path = repo_path / self.entry_for(action, config)
        if not config.dry_run and not entry_path.exists():
            raise FileNotFoundError(
                f"Third-party entry script is missing for {self.method_name}: "
                f"{entry_path}. Clone/install the method under third_party first."
            )

        command = [sys.executable, str(entry_path)]
        command.extend(self.build_action_args(action, config))
        command.extend(config.extra_args)

        return CommandSpec(command=command, cwd=repo_path, env=self.build_env(config, repo_path))

    def repo_path(self, config: MethodRunConfig) -> Path:
        repo_value = str(config.values.get("third_party_repo") or self.third_party_repo)
        repo_path = Path(repo_value).expanduser()
        if not repo_path.is_absolute():
            repo_path = config.project_root / repo_path
        return repo_path.resolve()

    def entry_for(self, action: str, config: MethodRunConfig) -> str:
        key = {
            "train": "train_entry",
            "render": "render_entry",
            "evaluate": "eval_entry",
        }[action]
        return str(config.values.get(key) or getattr(self, key))

    def build_action_args(self, action: str, config: MethodRunConfig) -> List[str]:
        if action == "train":
            args = ["-s", str(config.dataset_path), "-m", str(config.output_path)]
            self._append_optional(args, config, "images", "--images")
            self._append_optional(args, config, "iterations", "--iterations")
            self._append_optional(args, config, "resolution", "--resolution")
            self._append_optional(args, config, "test_iterations", "--test_iterations")
            self._append_optional(args, config, "checkpoint_path", "--start_checkpoint")
            self._append_eval_flag(args, config)
            return args
        if action == "render":
            args = ["-s", str(config.dataset_path), "-m", str(config.output_path)]
            self._append_optional(args, config, "images", "--images")
            self._append_optional(args, config, "resolution", "--resolution")
            self._append_optional(args, config, "render_iteration", "--iteration")
            self._append_optional(args, config, "checkpoint_path", "--checkpoint")
            self._append_eval_flag(args, config)
            return args
        if action == "evaluate":
            return ["-m", str(config.output_path)]
        raise ValueError(f"Unsupported action: {action}")

    def build_env(self, config: MethodRunConfig, repo_path: Path) -> Dict[str, str]:
        env = os.environ.copy()
        env["UNIFIED3DGS_OUTPUT_PATH"] = str(config.output_path)

        readonly_dataset_root = config.values.get("readonly_dataset_root")
        if readonly_dataset_root:
            env["UNIFIED3DGS_READONLY_DATASET_ROOT"] = str(readonly_dataset_root)
        else:
            env["UNIFIED3DGS_READONLY_DATASET_ROOT"] = readonly_dataset_root_env()

        cuda_home = config.values.get("cuda_home")
        if cuda_home:
            env["CUDA_HOME"] = str(cuda_home)
            env["CUDA_PATH"] = str(cuda_home)
            cuda_lib64 = str(Path(str(cuda_home)) / "lib64")
            old_ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                cuda_lib64 if not old_ld_path else f"{cuda_lib64}:{old_ld_path}"
            )

        arch_list = config.values.get("torch_cuda_arch_list")
        if arch_list:
            env["TORCH_CUDA_ARCH_LIST"] = str(arch_list)

        max_jobs = config.values.get("max_jobs")
        if max_jobs:
            env["MAX_JOBS"] = str(max_jobs)

        python_paths: List[str] = [
            str((config.project_root / "unified3dgs" / "runtime_guard").resolve())
        ]
        extension_path = config.values.get("extension_path")
        if extension_path:
            resolved_extension_path = Path(str(extension_path)).expanduser()
            if not resolved_extension_path.is_absolute():
                resolved_extension_path = config.project_root / resolved_extension_path
            python_paths.append(str(resolved_extension_path.resolve()))

        python_paths.append(str(repo_path))
        old_python_path = env.get("PYTHONPATH", "")
        if old_python_path:
            python_paths.append(old_python_path)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        return env

    @staticmethod
    def _append_optional(
        args: List[str],
        config: MethodRunConfig,
        key: str,
        flag: str,
    ) -> None:
        value: Optional[object] = config.values.get(key)
        if value is None or value == "":
            return
        args.extend([flag, str(value)])

    @staticmethod
    def _append_eval_flag(args: List[str], config: MethodRunConfig) -> None:
        if _is_eval_enabled(config.values.get("eval")):
            args.append("--eval")


def _is_eval_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
