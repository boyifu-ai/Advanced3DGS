from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from unified3dgs.dataset_config import validation_scene_records

SCENES = tuple(
    (label.split("/", 1)[0], label.split("/", 1)[1], str(path))
    for path, label in validation_scene_records()
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIP360_OUTDOOR = {"bicycle", "garden", "stump"}
MIP360_INDOOR = {"room", "counter", "kitchen", "bonsai"}


def official_image_folder(family: str, scene: str) -> str:
    if family == "mip360" and scene in MIP360_OUTDOOR:
        return "images_4"
    if family == "mip360" and scene in MIP360_INDOOR:
        return "images_2"
    return "images"


def first_image_size(folder: Path) -> Optional[Tuple[int, int]]:
    if not folder.is_dir():
        return None
    candidates = sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        return None
    with Image.open(candidates[0]) as image:
        return image.size


def latest_render_size(output: Path) -> Optional[Tuple[int, int]]:
    candidates = sorted(
        path
        for path in (output / "test").glob("ours_*/renders/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        return None
    with Image.open(candidates[-1]) as image:
        return image.size


def parse_cfg_args(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        expression = ast.parse(text, mode="eval").body
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


def expected_effective_size(
    source_size: Optional[Tuple[int, int]], resolution: object
) -> Optional[Tuple[int, int]]:
    if source_size is None:
        return None
    width, height = source_size
    try:
        value = int(resolution)
    except (TypeError, ValueError):
        return None

    if value in (1, 2, 4, 8):
        scale = float(value)
    elif value == -1:
        scale = width / 1600.0 if width > 1600 else 1.0
    elif value > 0:
        scale = width / float(value)
    else:
        return None
    return round(width / scale), round(height / scale)


def iter_method_dirs(validation_root: Path) -> Iterable[Path]:
    if not validation_root.is_dir():
        return ()
    return sorted(
        path
        for path in validation_root.iterdir()
        if path.is_dir() and path.name not in {"generated_configs", "progress", "run_logs"}
    )


def format_size(size: Optional[Tuple[int, int]]) -> str:
    return "" if size is None else f"{size[0]}x{size[1]}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit actual and official image-resolution protocols."
    )
    parser.add_argument(
        "--validation-root", type=Path, default=Path("outputs/validation")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/resolution_protocol_audit.csv"),
    )
    args = parser.parse_args()

    validation_root = args.validation_root.expanduser().resolve()
    rows: List[Dict[str, object]] = []

    for method_dir in iter_method_dirs(validation_root):
        method = method_dir.name
        for family, scene, dataset_value in SCENES:
            pair_root = method_dir / family / scene
            output = pair_root / "method_outputs"
            if not pair_root.is_dir():
                continue

            cfg = parse_cfg_args(output / "cfg_args")
            configured_images = str(cfg.get("images") or "images")
            configured_resolution = cfg.get("resolution", "unknown")
            dataset = Path(str(cfg.get("source_path") or dataset_value))

            configured_source_size = first_image_size(dataset / configured_images)
            official_folder = official_image_folder(family, scene)
            official_resolution = -1
            official_source_size = first_image_size(dataset / official_folder)
            official_effective_size = expected_effective_size(
                official_source_size, official_resolution
            )
            expected_size = expected_effective_size(
                configured_source_size, configured_resolution
            )
            actual_render_size = latest_render_size(output)

            folder_match = configured_images == official_folder
            resolution_match = str(configured_resolution) == str(official_resolution)
            render_size_match = (
                actual_render_size is not None
                and actual_render_size == official_effective_size
            )
            protocol_match = folder_match and resolution_match and render_size_match
            rows.append(
                {
                    "method": method,
                    "dataset": family,
                    "scene": scene,
                    "configured_images": configured_images,
                    "configured_resolution": configured_resolution,
                    "configured_source_size": format_size(configured_source_size),
                    "expected_effective_size": format_size(expected_size),
                    "actual_render_size": format_size(actual_render_size),
                    "official_images": official_folder,
                    "official_resolution": official_resolution,
                    "official_source_size": format_size(official_source_size),
                    "official_effective_size": format_size(official_effective_size),
                    "official_folder_match": folder_match,
                    "official_resolution_match": resolution_match,
                    "official_render_size_match": render_size_match,
                    "official_protocol_match": protocol_match,
                    "cfg_args": str(output / "cfg_args"),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "method",
        "dataset",
        "scene",
        "configured_images",
        "configured_resolution",
        "configured_source_size",
        "expected_effective_size",
        "actual_render_size",
        "official_images",
        "official_resolution",
        "official_source_size",
        "official_effective_size",
        "official_folder_match",
        "official_resolution_match",
        "official_render_size_match",
        "official_protocol_match",
        "cfg_args",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote resolution protocol audit: {args.output.resolve()}")
    if not rows:
        print("No completed or partial method/scene output directories found.")
        return 0

    mismatches = [row for row in rows if not row["official_protocol_match"]]
    print(f"Audited method/scene pairs: {len(rows)}")
    print(f"Official protocol matches: {len(rows) - len(mismatches)}")
    print(f"Official protocol mismatches: {len(mismatches)}")
    print()
    print(
        "method,dataset/scene,used,resolution,expected_effective,actual_render,"
        "official,official_resolution,official_source,official_effective,"
        "render_size_match,match"
    )
    for row in rows:
        print(
            f"{row['method']},{row['dataset']}/{row['scene']},"
            f"{row['configured_images']},{row['configured_resolution']},"
            f"{row['expected_effective_size']},{row['actual_render_size']},"
            f"{row['official_images']},{row['official_resolution']},"
            f"{row['official_source_size']},{row['official_effective_size']},"
            f"{row['official_render_size_match']},{row['official_protocol_match']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
