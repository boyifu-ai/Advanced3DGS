from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.utils.iterations import resolve_output_iteration


METHOD_REPOS = {
    "vanilla_3dgs": "third_party/gaussian-splatting",
    "2dgs": "third_party/2d-gaussian-splatting",
    "3dgs_mcmc": "third_party/3dgs-mcmc",
    "3dhgs": "third_party/3DHGS",
    "sss": "third_party/3D-student-splatting-and-scooping",
}
METRICS = ("PSNR", "SSIM", "LPIPS")
EXPECTED_ITERATION = 30000
MIP360_OUTDOOR = {"bicycle", "garden", "stump"}
MIP360_INDOOR = {"room", "counter", "kitchen", "bonsai"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_SCENES = {
    "mip360": (
        "bicycle",
        "bonsai",
        "counter",
        "garden",
        "kitchen",
        "room",
        "stump",
    ),
    "tandt": ("train", "truck"),
    "deep_blending": ("drjohnson", "playroom"),
}


def parse_cfg_args(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        expression = ast.parse(
            path.read_text(encoding="utf-8", errors="replace").strip(),
            mode="eval",
        ).body
    except SyntaxError:
        return {}
    if not isinstance(expression, ast.Call):
        return {}
    values: Dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            continue
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            continue
    return values


def count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def first_image(path: Path) -> Optional[Path]:
    if not path.is_dir():
        return None
    return next(
        (
            item
            for item in sorted(path.iterdir())
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ),
        None,
    )


def image_size(path: Optional[Path]) -> Optional[tuple[int, int]]:
    if path is None:
        return None
    with Image.open(path) as image:
        return image.size


def expected_effective_size(path: Optional[Path]) -> Optional[tuple[int, int]]:
    size = image_size(path)
    if size is None:
        return None
    width, height = size
    if width <= 1600:
        return width, height
    scale = width / 1600.0
    return int(width / scale), int(height / scale)


def finite_metrics(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict) or key not in data or not isinstance(data[key], dict):
        return False
    values = data[key]
    return all(
        isinstance(values.get(name), (int, float)) and math.isfinite(float(values[name]))
        for name in METRICS
    )


def iter_scene_dirs(validation_root: Path) -> Iterable[tuple[str, str, str, Path]]:
    for method in METHOD_REPOS:
        method_root = validation_root / method
        if not method_root.is_dir():
            continue
        for family_dir in sorted(path for path in method_root.iterdir() if path.is_dir()):
            if family_dir.name in {"generated_configs", "progress", "run_logs"}:
                continue
            for scene_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
                if (scene_dir / "method_outputs").is_dir():
                    yield method, family_dir.name, scene_dir.name, scene_dir


def git_commit(repo: Path) -> str:
    if not (repo / ".git").exists():
        return ""
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def metric_script(project_root: Path, method: str) -> Path:
    if method == "3dhgs":
        return project_root / METHOD_REPOS["vanilla_3dgs"] / "metrics.py"
    return project_root / METHOD_REPOS[method] / "metrics.py"


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def expected_images(family: str, scene: str) -> str:
    if family == "mip360" and scene in MIP360_OUTDOOR:
        return "images_4"
    if family == "mip360" and scene in MIP360_INDOOR:
        return "images_2"
    return "images"


def configured_iteration(
    project_root: Path,
    validation_root: Path,
    method: str,
    scene: str,
    cfg: Dict[str, object],
) -> Optional[int]:
    return EXPECTED_ITERATION


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit formal Unified 3DGS benchmark outputs.")
    parser.add_argument("--validation-root", type=Path, default=Path("outputs/validation"))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_REPOS),
        default=sorted(METHOD_REPOS),
    )
    parser.add_argument(
        "--dataset-families",
        nargs="+",
        choices=sorted(EXPECTED_SCENES),
        default=sorted(EXPECTED_SCENES),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[],
        help="Optional dataset/scene labels. When set, audit only these selected scenes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/benchmark_protocol_audit.csv"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    validation_root = args.validation_root.expanduser().resolve()
    rows = []
    selected_methods = set(args.methods)
    selected_families = set(args.dataset_families)
    selected_scenes = set(args.scenes)
    for method, family, scene, scene_dir in iter_scene_dirs(validation_root):
        if method not in selected_methods or family not in selected_families:
            continue
        if selected_scenes and f"{family}/{scene}" not in selected_scenes:
            continue
        output = scene_dir / "method_outputs"
        cfg = parse_cfg_args(output / "cfg_args")
        actual_iteration = resolve_output_iteration(output, EXPECTED_ITERATION)
        protocol_iteration = configured_iteration(
            project_root,
            validation_root,
            method,
            scene,
            cfg,
        )
        result_key = f"ours_{actual_iteration}"
        renders = output / "test" / result_key / "renders"
        gt = output / "test" / result_key / "gt"
        render_count = count_files(renders)
        gt_count = count_files(gt)
        expected_render_size = expected_effective_size(
            first_image(
                Path(str(cfg.get("source_path") or ""))
                / expected_images(family, scene)
            )
        )
        actual_render_size = image_size(first_image(renders))
        checks = {
            "images_ok": str(cfg.get("images") or "images") == expected_images(family, scene),
            "resolution_ok": str(cfg.get("resolution")) == "-1",
            "iterations_ok": (
                actual_iteration == protocol_iteration
                if protocol_iteration is not None
                else actual_iteration > 0
            ),
            "eval_split_ok": as_bool(cfg.get("eval")),
            "checkpoint_ok": (
                output
                / "point_cloud"
                / f"iteration_{actual_iteration}"
                / "point_cloud.ply"
            ).is_file(),
            "render_pair_ok": render_count > 0 and render_count == gt_count,
            "render_size_ok": (
                actual_render_size is not None
                and actual_render_size == expected_render_size
            ),
            "metrics_ok": finite_metrics(output / "results.json", result_key),
        }
        rows.append(
            {
                "method": method,
                "dataset": family,
                "scene": scene,
                "final_iteration": actual_iteration,
                "protocol_iteration": protocol_iteration or "",
                "status": "pending",
                "method_protocol": "upstream/default",
                "dataset_path": str(cfg.get("source_path") or ""),
                "scene_config": str(cfg.get("config") or ""),
                **checks,
                "test_renders": render_count,
                "test_gt": gt_count,
                "expected_render_size": (
                    f"{expected_render_size[0]}x{expected_render_size[1]}"
                    if expected_render_size
                    else ""
                ),
                "actual_render_size": (
                    f"{actual_render_size[0]}x{actual_render_size[1]}"
                    if actual_render_size
                    else ""
                ),
                "method_commit": git_commit(project_root / METHOD_REPOS[method]),
                "metric_script": str(metric_script(project_root, method)),
                "metric_script_sha256": file_sha256(metric_script(project_root, method)),
            }
        )

    existing = {
        (str(row["method"]), str(row["dataset"]), str(row["scene"])) for row in rows
    }
    for method in args.methods:
        for family in args.dataset_families:
            for scene in EXPECTED_SCENES[family]:
                if selected_scenes and f"{family}/{scene}" not in selected_scenes:
                    continue
                if (method, family, scene) in existing:
                    continue
                rows.append(
                    {
                        "method": method,
                        "dataset": family,
                        "scene": scene,
                        "final_iteration": "",
                        "protocol_iteration": "",
                        "status": "fail",
                        "method_protocol": "upstream/default",
                        "dataset_path": "",
                        "scene_config": "",
                        "images_ok": False,
                        "resolution_ok": False,
                        "iterations_ok": False,
                        "eval_split_ok": False,
                        "checkpoint_ok": False,
                        "render_pair_ok": False,
                        "render_size_ok": False,
                        "metrics_ok": False,
                        "test_renders": 0,
                        "test_gt": 0,
                        "expected_render_size": "",
                        "actual_render_size": "",
                        "method_commit": git_commit(project_root / METHOD_REPOS[method]),
                        "metric_script": str(metric_script(project_root, method)),
                        "metric_script_sha256": file_sha256(metric_script(project_root, method)),
                    }
                )

    test_counts: Dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["scene"]))
        if int(row["test_renders"]) > 0:
            test_counts.setdefault(key, set()).add(int(row["test_renders"]))
    for row in rows:
        key = (str(row["dataset"]), str(row["scene"]))
        row["cross_method_test_count_ok"] = len(test_counts.get(key, set())) <= 1
        row["status"] = (
            "pass"
            if all(value is not False for key, value in row.items() if key.endswith("_ok"))
            else "fail"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "method",
        "dataset",
        "scene",
        "status",
        "method_protocol",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failed = [row for row in rows if row["status"] != "pass"]
    extensions = [row for row in rows if row["method_protocol"] == "project_extension_config"]
    print(f"Wrote benchmark protocol audit: {args.output.resolve()}")
    print(f"Audited method/scene pairs: {len(rows)}")
    print(f"Passed: {len(rows) - len(failed)}")
    print(f"Failed: {len(failed)}")
    print(f"Project-extension config results: {len(extensions)}")
    metric_hashes = {
        str(row["metric_script_sha256"]) for row in rows if row["metric_script_sha256"]
    }
    print(f"Metric script variants: {len(metric_hashes)}")
    if len(metric_hashes) > 1:
        print("NOTE: multiple metric script hashes are in use; inspect the CSV before paper reporting.")
    for row in failed:
        failed_checks = [
            key for key, value in row.items() if key.endswith("_ok") and value is False
        ]
        print(f"FAIL {row['method']}/{row['dataset']}/{row['scene']}: {failed_checks}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
