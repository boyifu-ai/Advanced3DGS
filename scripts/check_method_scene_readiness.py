from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


REPOS = {
    "vanilla_3dgs": "third_party/gaussian-splatting",
    "2dgs": "third_party/2d-gaussian-splatting",
    "3dgs_mcmc": "third_party/3dgs-mcmc",
    "3dhgs": "third_party/3DHGS",
    "sss": "third_party/3D-student-splatting-and-scooping",
}
SCENES = (
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
    "train",
    "truck",
    "drjohnson",
    "playroom",
)
PATCH_MARKER = "Unified 3DGS robust PLY reader patch"
MCMC_RANDOM_MARKER = "Unified 3DGS no dataset-write random initialization patch v2"
THREEDHGS_RENDER_GT_MARKER = "Unified 3DGS 3DHGS render GT patch"
MCMC_REQUIRED_KEYS = ("cap_max",)
SSS_REQUIRED_KEYS = ("cap_max",)
FORBIDDEN_PROTOCOL_KEYS = (
    "source_path",
    "model_path",
    "images",
    "eval",
    "test_iterations",
    "save_iterations",
    "checkpoint_iterations",
    "start_checkpoint",
)
MCMC_FORBIDDEN_PROTOCOL_KEYS = FORBIDDEN_PROTOCOL_KEYS
SSS_FORBIDDEN_PROTOCOL_KEYS = FORBIDDEN_PROTOCOL_KEYS
def check_reader(method: str, repo: Path, errors: List[str]) -> None:
    reader = repo / "scene" / "dataset_readers.py"
    if not reader.is_file():
        errors.append(f"{method}: missing reader {reader}")
        return

    text = reader.read_text(encoding="utf-8", errors="replace")
    if "def fetchPly(" in text and PATCH_MARKER not in text:
        errors.append(f"{method}: robust PLY reader patch is missing")
    if method == "3dgs_mcmc" and MCMC_RANDOM_MARKER not in text:
        errors.append(f"{method}: random initialization output redirection v2 is missing")


def check_mcmc_configs(
    repo: Path,
    generated_root: Path,
    expected_iterations: int,
    expected_resolution: int,
    errors: List[str],
) -> None:
    for scene in SCENES:
        path = generated_root / f"{scene}.json"
        if not path.is_file():
            errors.append(f"3dgs_mcmc: missing unified scene config for {scene}: {path}")
            continue
        try:
            data: Dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"3dgs_mcmc: invalid JSON config {path}: {exc}")
            continue
        missing = [key for key in MCMC_REQUIRED_KEYS if key not in data]
        if missing:
            errors.append(f"3dgs_mcmc: {path} is missing keys {missing}")
        forbidden = [key for key in MCMC_FORBIDDEN_PROTOCOL_KEYS if key in data]
        if forbidden:
            errors.append(
                f"3dgs_mcmc: {path} contains protocol-critical keys that must remain "
                f"controlled by the unified runner: {forbidden}"
            )
        if data.get("iterations") != expected_iterations:
            errors.append(
                f"3dgs_mcmc: {path} must set iterations={expected_iterations}, "
                f"got {data.get('iterations')!r}"
            )
        if data.get("resolution") != expected_resolution:
            errors.append(
                f"3dgs_mcmc: {path} must set resolution={expected_resolution}, "
                f"got {data.get('resolution')!r}. MCMC loads JSON after CLI arguments."
            )


def check_3dhgs_render_patch(repo: Path, errors: List[str]) -> None:
    render_py = repo / "render.py"
    if not render_py.is_file():
        return
    text = render_py.read_text(encoding="utf-8", errors="replace")
    if THREEDHGS_RENDER_GT_MARKER not in text:
        errors.append("3dhgs: render.py is missing GT output patch")


def check_sss_configs(
    repo: Path,
    generated_root: Path,
    expected_iterations: int,
    expected_resolution: int,
    errors: List[str],
) -> None:
    for scene in SCENES:
        path = generated_root / f"{scene}.json"
        if not path.is_file():
            errors.append(f"sss: missing unified scene config for {scene}: {path}")
            continue
        try:
            data: Dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"sss: invalid JSON config {path}: {exc}")
            continue
        missing = [key for key in SSS_REQUIRED_KEYS if key not in data]
        if missing:
            errors.append(f"sss: {path} is missing keys {missing}")
        forbidden = [key for key in SSS_FORBIDDEN_PROTOCOL_KEYS if key in data]
        if forbidden:
            errors.append(
                f"sss: {path} contains protocol-critical keys that must remain "
                f"controlled by the unified runner: {forbidden}"
            )
        if data.get("iterations") != expected_iterations:
            errors.append(
                f"sss: {path} must set iterations={expected_iterations}, "
                f"got {data.get('iterations')!r}"
            )
        if data.get("resolution") != expected_resolution:
            errors.append(
                f"sss: {path} must set resolution={expected_resolution}, "
                f"got {data.get('resolution')!r}. SSS loads JSON after CLI arguments, "
                "so this value controls the effective training resolution."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check third-party readiness for full validation.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(REPOS),
        default=sorted(REPOS),
    )
    parser.add_argument(
        "--validation-root",
        default=Path("outputs/validation"),
        type=Path,
    )
    parser.add_argument("--iterations", default=30000, type=int)
    parser.add_argument("--resolution", default=-1, type=int)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    validation_root = args.validation_root
    if not validation_root.is_absolute():
        validation_root = (project_root / validation_root).resolve()
    errors: List[str] = []
    for method in args.methods:
        repo = project_root / REPOS[method]
        entries = ["train.py", "render.py"]
        if method != "3dhgs":
            entries.append("metrics.py")
        for entry in entries:
            if not (repo / entry).is_file():
                errors.append(f"{method}: missing entry script {repo / entry}")
        if method == "3dhgs":
            fallback = project_root / REPOS["vanilla_3dgs"] / "metrics.py"
            if not fallback.is_file():
                errors.append(f"{method}: missing fallback metric script {fallback}")
        check_reader(method, repo, errors)
        if method == "3dhgs":
            check_3dhgs_render_patch(repo, errors)
        if method == "3dgs_mcmc":
            check_mcmc_configs(
                repo,
                validation_root / "generated_configs" / "3dgs_mcmc",
                args.iterations,
                args.resolution,
                errors,
            )
        if method == "sss":
            check_sss_configs(
                repo,
                validation_root / "generated_configs" / "sss",
                args.iterations,
                args.resolution,
                errors,
            )

    if errors:
        print("Method/scene readiness check failed:")
        for error in errors:
            print(f"- {error}")
        return 2

    print(f"Method/scene readiness passed for: {' '.join(args.methods)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
