from __future__ import annotations

import json
from pathlib import Path

from unified3dgs.methods.base import BaseMethodAdapter


class ThreeDGSMCMCAdapter(BaseMethodAdapter):
    method_name = "3dgs_mcmc"
    third_party_repo = "third_party/3dgs-mcmc"
    train_entry = "train.py"
    render_entry = "render.py"
    eval_entry = "metrics.py"

    def build_action_args(self, action, config):
        args = super().build_action_args(action, config)
        if action != "train":
            return args

        scene_config = self._scene_config_path(config)
        if scene_config.is_file():
            self._validate_scene_config(scene_config, config)
            args.extend(["--config", str(scene_config)])
            return args
        if config.dry_run and not self.repo_path(config).is_dir():
            args.extend(["--config", str(scene_config)])
            return args

        cap_max = config.values.get("cap_max")
        if cap_max is None or cap_max == "":
            raise FileNotFoundError(
                "3DGS-MCMC requires either an upstream scene config or an explicit "
                f"cap_max. Missing scene config: {scene_config}. Use --set cap_max=N "
                "only after determining the scene's target Gaussian count."
            )

        args.extend(["--cap_max", str(cap_max)])
        for key in ("scale_reg", "opacity_reg", "noise_lr", "init_type"):
            value = config.values.get(key)
            if value is not None and value != "":
                args.extend([f"--{key}", str(value)])
        return args

    def _scene_config_path(self, config) -> Path:
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
                or "outputs/validation/generated_configs/3dgs_mcmc"
            )
        ).expanduser()
        if not generated_dir.is_absolute():
            generated_dir = config.project_root / generated_dir
        generated_path = (generated_dir / f"{scene_name}.json").resolve()
        if generated_path.is_file():
            return generated_path

        if _as_bool(config.values.get("allow_upstream_scene_config"), default=False):
            config_dir = str(config.values.get("scene_config_dir") or "configs")
            upstream_path = (self.repo_path(config) / config_dir / f"{scene_name}.json").resolve()
            if upstream_path.is_file() or not self.repo_path(config).is_dir():
                return upstream_path
        return generated_path

    @staticmethod
    def _validate_scene_config(path, config):
        data = json.loads(path.read_text(encoding="utf-8"))
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
                f"3DGS-MCMC scene config {path} overrides unified runner keys: {present}"
            )
        for key in ("iterations", "resolution"):
            expected = config.values.get(key)
            if expected is not None and data.get(key) != expected:
                raise ValueError(
                    f"3DGS-MCMC scene config {path} has {key}={data.get(key)!r}, "
                    f"but the unified run requested {expected!r}. MCMC applies JSON "
                    "after CLI parsing, so training is blocked before it can use "
                    "a silently overridden protocol."
                )


def _as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
