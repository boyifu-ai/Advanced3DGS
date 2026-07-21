from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
RENDER_DIR_NAMES = {"renders", "render", "pred", "preds", "prediction", "predictions"}
GT_DIR_NAMES = {
    "gt",
    "gts",
    "target",
    "targets",
    "ground_truth",
    "ground-truth",
}


def is_render_dir_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in RENDER_DIR_NAMES or lowered.startswith(
        ("renders_", "render_", "preds_", "test_preds_", "train_preds_")
    )


def is_gt_dir_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in GT_DIR_NAMES or lowered.startswith(
        ("gt_", "gts_", "test_gt_", "train_gt_")
    )


@dataclass(frozen=True)
class RenderPair:
    root: Path
    renders: Path
    gt: Path
    pairs: Tuple[Tuple[Path, Path], ...]


def image_files(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def pair_images(renders: Path, gt: Path) -> Tuple[Tuple[Path, Path], ...]:
    render_files = image_files(renders)
    gt_files = image_files(gt)
    if not render_files or not gt_files:
        return ()

    gt_by_name = {path.name: path for path in gt_files}
    exact = tuple(
        (render, gt_by_name[render.name])
        for render in render_files
        if render.name in gt_by_name
    )
    if len(exact) == len(render_files) == len(gt_files):
        return exact

    gt_by_stem = {path.stem: path for path in gt_files}
    stems = tuple(
        (render, gt_by_stem[render.stem])
        for render in render_files
        if render.stem in gt_by_stem
    )
    if len(stems) == len(render_files) == len(gt_files):
        return stems

    def numeric_suffix(path: Path) -> str:
        match = re.search(r"(\d+)$", path.stem)
        return match.group(1) if match else ""

    render_by_index = {
        numeric_suffix(path): path for path in render_files if numeric_suffix(path)
    }
    gt_by_index = {
        numeric_suffix(path): path for path in gt_files if numeric_suffix(path)
    }
    if (
        len(render_by_index) == len(render_files)
        and len(gt_by_index) == len(gt_files)
        and set(render_by_index) == set(gt_by_index)
    ):
        return tuple(
            (render_by_index[index], gt_by_index[index])
            for index in sorted(render_by_index)
        )

    if len(render_files) == len(gt_files):
        return tuple(zip(render_files, gt_files))
    return ()


def _candidate_score(parent: Path, iteration: int, pair_count: int) -> Tuple[int, int]:
    text = parent.as_posix().lower()
    score = pair_count
    if f"ours_{iteration}" in text or f"iteration_{iteration}" in text:
        score += 100_000
    if "/test/" in f"/{text}/" or text.endswith("/test"):
        score += 10_000
    return score, -len(parent.parts)


def _candidate_roots(output: Path) -> List[Path]:
    roots = [output]
    report = output / "unified3dgs_training_report.json"
    if report.is_file():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            for value in payload.get("resolved_result_roots", []):
                path = Path(str(value)).expanduser()
                if path.is_dir() and path not in roots:
                    roots.append(path)
        except Exception:
            pass
    return roots


def discover_render_pair(output: Path, iteration: int) -> RenderPair:
    candidates: List[RenderPair] = []
    visited = set()
    for root in _candidate_roots(output.expanduser().resolve()):
        if not root.is_dir():
            continue
        for directory in [root, *root.rglob("*")]:
            if not directory.is_dir() or directory.resolve() in visited:
                continue
            visited.add(directory.resolve())
            if not is_render_dir_name(directory.name):
                continue
            parent = directory.parent
            for sibling in parent.iterdir():
                if not sibling.is_dir() or not is_gt_dir_name(sibling.name):
                    continue
                pairs = pair_images(directory, sibling)
                if pairs:
                    candidates.append(
                        RenderPair(
                            root=parent,
                            renders=directory,
                            gt=sibling,
                            pairs=pairs,
                        )
                    )
    if not candidates:
        raise ValueError(
            "No paired render/GT directories found under training result roots for "
            f"{output}. Expected sibling directories such as renders/ and gt/."
        )
    return max(
        candidates,
        key=lambda item: _candidate_score(item.root, iteration, len(item.pairs)),
    )


def result_roots_from_training_report(output: Path) -> List[Path]:
    roots = _candidate_roots(output.expanduser().resolve())
    return [root for root in roots if root.is_dir()]


def resolved_model_output(output: Path) -> Path:
    output = output.expanduser().resolve()
    roots = result_roots_from_training_report(output)
    report = output / "unified3dgs_training_report.json"
    saved_files: List[Path] = []
    if report.is_file():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            saved_files = [
                Path(str(value)).expanduser().resolve()
                for value in payload.get("saved_files", [])
            ]
        except Exception:
            saved_files = []
    for root in roots:
        if root == output:
            continue
        if any(path == root or root in path.parents for path in saved_files):
            return root
    for root in roots:
        if root != output and (
            any(root.rglob("point_cloud.ply"))
            or any(root.rglob("*.pth"))
            or any(root.rglob("*.ckpt"))
        ):
            return root
    return roots[0] if roots else output
