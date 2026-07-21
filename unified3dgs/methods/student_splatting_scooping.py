from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

from unified3dgs.methods.base import (
    BaseMethodAdapter,
    CommandSpec,
    MethodRunConfig,
)


class StudentSplattingScoopingAdapter(BaseMethodAdapter):
    method_name = "sss"
    third_party_repo = "third_party/3D-student-splatting-and-scooping"
    train_entry = "train.py"
    render_entry = "render.py"
    eval_entry = "metrics.py"

    def build_command(self, action: str, config: MethodRunConfig) -> CommandSpec:
        if action != "evaluate":
            return super().build_command(action, config)

        repo_path = self.repo_path(config)
        evaluator = config.project_root / "scripts" / "evaluate_render_pairs_official.py"
        if not config.dry_run and not evaluator.is_file():
            raise FileNotFoundError(f"Unified official-metrics evaluator is missing: {evaluator}")
        iteration = int(
            config.values.get("render_iteration")
            or config.values.get("iterations")
            or 30000
        )
        command = [
            sys.executable,
            str(evaluator),
            "--method",
            self.method_name,
            "--repo",
            str(repo_path),
            "--style",
            "standard_3dgs",
            "--output",
            str(config.output_path),
            "--iteration",
            str(iteration),
            "--results-output",
            str(config.output_path / "results.json"),
        ]
        command.extend(config.extra_args)
        return CommandSpec(
            command=command,
            cwd=config.project_root,
            env=self.build_env(config, repo_path),
        )

    def build_action_args(self, action: str, config: MethodRunConfig) -> List[str]:
        if action == "train":
            args = super().build_action_args(action, config)
            scene_config = self._scene_config_path(config)
            if scene_config.is_file():
                self._validate_scene_config(scene_config, config)
                args.extend(["--config", str(scene_config)])
                return args
            if config.dry_run and not self.repo_path(config).is_dir():
                args.extend(["--config", str(scene_config)])
                return args
            raise FileNotFoundError(
                "SSS requires an upstream or generated scene config. Missing: "
                f"{scene_config}. Run scripts/prepare_sss_scene_configs.py or "
                "set scene_config_path explicitly."
            )

        if action == "render":
            args = ["-m", str(config.output_path)]
            self._append_optional(args, config, "render_iteration", "--iteration")
            if _as_bool(config.values.get("render_skip_train"), default=True):
                args.append("--skip_train")
            return args

        return super().build_action_args(action, config)

    def _scene_config_path(self, config: MethodRunConfig) -> Path:
        configured = config.values.get("scene_config_path")
        if configured:
            path = Path(str(configured)).expanduser()
            if not path.is_absolute():
                path = self.repo_path(config) / path
            return path.resolve()

        scene_name = config.dataset_path.name
        generated_dir = Path(
            str(
                config.values.get("generated_scene_config_dir")
                or "outputs/validation/generated_configs/sss"
            )
        ).expanduser()
        if not generated_dir.is_absolute():
            generated_dir = config.project_root / generated_dir
        generated_path = (generated_dir / f"{scene_name}.json").resolve()
        if generated_path.is_file():
            return generated_path

        if _as_bool(config.values.get("allow_upstream_scene_config"), default=False):
            config_dir = str(config.values.get("scene_config_dir") or "configs")
            repo_path = self.repo_path(config)
            upstream_path = (repo_path / config_dir / f"{scene_name}.json").resolve()
            if upstream_path.is_file() or not repo_path.is_dir():
                return upstream_path
        return generated_path

    @staticmethod
    def _validate_scene_config(path: Path, config: MethodRunConfig) -> None:
        data: Dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        forbidden = (
            "source_path",
            "model_path",
            "images",
            "eval",
            "test_iterations",
            "save_iterations",
            "checkpoint_iterations",
            "start_checkpoint",
        )
        present = [key for key in forbidden if key in data]
        if present:
            raise ValueError(
                f"SSS scene config {path} overrides unified runner keys: {present}"
            )
        for key in ("iterations", "resolution"):
            expected = config.values.get(key)
            if expected is not None and data.get(key) != expected:
                raise ValueError(
                    f"SSS scene config {path} has {key}={data.get(key)!r}, "
                    f"but the unified run requested {expected!r}. SSS applies JSON "
                    "after CLI parsing, so training is blocked before it can use "
                    "a silently overridden protocol."
                )


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
