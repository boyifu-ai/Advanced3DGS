from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image


OUTDOOR = {"bicycle", "flowers", "garden", "stump", "treehill"}
INDOOR = {"room", "counter", "kitchen", "bonsai"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_cfg_args(path: Path) -> Dict[str, object]:
    try:
        expression = ast.parse(
            path.read_text(encoding="utf-8", errors="replace").strip(),
            mode="eval",
        ).body
    except (FileNotFoundError, SyntaxError) as exc:
        raise ValueError(f"Unable to parse effective cfg_args: {path}") from exc
    if not isinstance(expression, ast.Call):
        raise ValueError(f"Unexpected cfg_args format: {path}")
    values: Dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            continue
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            continue
    return values


def expected_images(family: str, scene: str) -> str:
    if family == "mip360" and scene in OUTDOOR:
        return "images_4"
    if family == "mip360" and scene in INDOOR:
        return "images_2"
    return "images"


def first_image(path: Path) -> Path:
    candidates = sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"No input images found under: {path}")
    return candidates[0]


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def expected_effective_size(source: Path) -> Tuple[int, int]:
    width, height = image_size(source)
    if width <= 1600:
        return width, height
    scale = width / 1600.0
    return int(width / scale), int(height / scale)


def verify_cfg(
    output: Path,
    data: Path,
    family: str,
    scene: str,
    iteration: int,
) -> Dict[str, object]:
    cfg = parse_cfg_args(output / "cfg_args")
    expected_folder = expected_images(family, scene)
    checks = {
        "source_path": str(Path(str(cfg.get("source_path", ""))).resolve()) == str(data.resolve()),
        "images": str(cfg.get("images") or "images") == expected_folder,
        "resolution": str(cfg.get("resolution")) == "-1",
        "eval": cfg.get("eval") is True,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        details = ", ".join(f"{key}={cfg.get(key)!r}" for key in failed)
        raise ValueError(f"Effective benchmark protocol mismatch in {output / 'cfg_args'}: {details}")
    return cfg


def verify_render_size(output: Path, data: Path, family: str, scene: str, iteration: int) -> None:
    folder = expected_images(family, scene)
    expected = expected_effective_size(first_image(data / folder))
    render_dir = output / "test" / f"ours_{iteration}" / "renders"
    render = first_image(render_dir)
    actual = image_size(render)
    if actual != expected:
        raise ValueError(
            f"Rendered image size mismatch for {family}/{scene}: "
            f"actual={actual}, expected={expected}, render={render}"
        )
    print(f"Verified effective render size: {actual[0]}x{actual[1]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the effective benchmark parameters recorded by a method."
    )
    parser.add_argument("--stage", choices=("train", "render"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    data = args.data.expanduser().resolve()
    cfg = verify_cfg(output, data, args.family, args.scene, args.iteration)
    print(
        "Verified effective benchmark config: "
        f"images={cfg.get('images')} resolution={cfg.get('resolution')} "
        f"eval={cfg.get('eval')}"
    )
    if args.stage == "render":
        verify_render_size(output, data, args.family, args.scene, args.iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
