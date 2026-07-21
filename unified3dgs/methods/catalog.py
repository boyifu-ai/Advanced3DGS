from __future__ import annotations

import sys
from typing import Dict

from unified3dgs.methods.base import BaseMethodAdapter, CommandSpec, MethodRunConfig
from unified3dgs.method_catalog import build_method_env


class CatalogMethodAdapter(BaseMethodAdapter):
    """User-facing adapter for a confirmed method."""

    def __init__(self, method: Dict[str, object]) -> None:
        self.method_name = str(method["key"])
        self.third_party_repo = str(method["local_path"])

    def build_command(self, action: str, config: MethodRunConfig) -> CommandSpec:
        if action != "train":
            return self._build_stage_command(action, config)
        iterations = int(config.values.get("iterations") or 30000)
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")

        command = [
            sys.executable,
            "-u",
            str(config.project_root / "scripts" / "run_method_training.py"),
            "--method",
            self.method_name,
            "--dataset",
            str(config.dataset_path),
            "--output",
            str(config.output_path),
            "--iterations",
            str(iterations),
            "--min-free-disk-gb",
            str(float(config.values.get("min_free_disk_gb") or 5)),
            "--min-free-vram-gb",
            str(float(config.values.get("min_free_vram_gb") or 4)),
        ]
        dataset_label = config.values.get("dataset_label")
        if dataset_label:
            command.extend(["--dataset-label", str(dataset_label)])
        timeout_seconds = int(config.values.get("timeout_seconds") or 0)
        if timeout_seconds > 0:
            command.extend(["--timeout-seconds", str(timeout_seconds)])
        heartbeat_seconds = int(config.values.get("heartbeat_seconds") or 30)
        if heartbeat_seconds > 0:
            command.extend(["--heartbeat-seconds", str(heartbeat_seconds)])
        images = config.values.get("images")
        if images:
            command.extend(["--images", str(images)])
        resolution = config.values.get("resolution")
        if resolution is not None:
            command.extend(["--resolution", str(resolution)])
        test_iterations = config.values.get("test_iterations")
        if test_iterations is not None:
            command.extend(["--test-iterations", str(test_iterations)])
        if str(config.values.get("eval", "")).lower() in {"1", "true", "yes", "on"}:
            command.append("--eval")
        if config.extra_args:
            command.append("--")
            command.extend(config.extra_args)
        repo = self.repo_path(config)
        env = build_method_env(self.method_name, repo, runtime_namespace="method_runtime")
        return CommandSpec(command=command, cwd=config.project_root, env=env)

    def _build_stage_command(self, action: str, config: MethodRunConfig) -> CommandSpec:
        stage = {"render": "render", "evaluate": "eval"}[action]
        iteration = int(
            config.values.get("render_iteration")
            or config.values.get("iteration")
            or config.values.get("iterations")
            or 30000
        )
        command = [
            sys.executable,
            "-u",
            str(config.project_root / "scripts" / "run_method_stage.py"),
            "--stage",
            stage,
            "--method",
            self.method_name,
            "--dataset",
            str(config.dataset_path),
            "--output",
            str(config.output_path),
            "--iteration",
            str(iteration),
        ]
        timeout_seconds = int(config.values.get("timeout_seconds") or 0)
        if timeout_seconds > 0:
            command.extend(["--timeout-seconds", str(timeout_seconds)])
        heartbeat_seconds = int(config.values.get("heartbeat_seconds") or 30)
        if heartbeat_seconds > 0:
            command.extend(["--heartbeat-seconds", str(heartbeat_seconds)])
        images = config.values.get("images")
        if images:
            command.extend(["--images", str(images)])
        resolution = config.values.get("resolution")
        if resolution is not None:
            command.extend(["--resolution", str(resolution)])
        if str(config.values.get("eval", "")).lower() in {"1", "true", "yes", "on"}:
            command.append("--eval")
        if config.extra_args:
            command.append("--")
            command.extend(config.extra_args)
        repo = self.repo_path(config)
        env = build_method_env(self.method_name, repo, runtime_namespace="method_runtime")
        return CommandSpec(command=command, cwd=config.project_root, env=env)
