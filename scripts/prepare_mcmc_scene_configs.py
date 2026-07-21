from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


SCENE_TEMPLATES: Dict[str, str] = {
    "bicycle": "bicycle",
    "bonsai": "bonsai",
    "counter": "counter",
    "garden": "garden",
    "kitchen": "kitchen",
    "room": "room",
    "stump": "stump",
    "treehill": "garden",
    "flowers": "garden",
    "train": "train",
    "truck": "truck",
    "drjohnson": "drjohnson",
    "playroom": "playroom",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare unified 3DGS-MCMC scene configs.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument(
        "--validation-root",
        default=Path("outputs/validation"),
        type=Path,
    )
    parser.add_argument("--iterations", default=30000, type=int)
    parser.add_argument("--resolution", default=-1, type=int)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    project_root = args.project_root.resolve()
    validation_root = args.validation_root
    if not validation_root.is_absolute():
        validation_root = (project_root / validation_root).resolve()

    upstream_configs = project_root / "third_party" / "3dgs-mcmc" / "configs"
    generated_root = validation_root / "generated_configs" / "3dgs_mcmc"
    generated_root.mkdir(parents=True, exist_ok=True)

    for scene, template_scene in SCENE_TEMPLATES.items():
        template_path = upstream_configs / f"{template_scene}.json"
        if not template_path.is_file():
            raise FileNotFoundError(f"Missing MCMC template config: {template_path}")

        config = json.loads(template_path.read_text(encoding="utf-8"))
        cap_max = config.get("cap_max")
        if not isinstance(cap_max, int) or cap_max <= 0:
            raise ValueError(f"Template config has invalid cap_max: {template_path}")
        config["iterations"] = args.iterations
        config["resolution"] = args.resolution

        output_path = generated_root / f"{scene}.json"
        output_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"Generated unified MCMC config: {output_path} "
            f"template={template_path.name} iterations={args.iterations} "
            f"resolution={args.resolution} cap_max={cap_max}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
