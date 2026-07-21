from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


ITERATION_DIR_PATTERN = re.compile(r"^iteration_(\d+)$")


def checkpoint_iterations(output: Path) -> List[int]:
    point_cloud_root = output / "point_cloud"
    if not point_cloud_root.is_dir():
        return []

    iterations: List[int] = []
    for iteration_dir in point_cloud_root.iterdir():
        match = ITERATION_DIR_PATTERN.match(iteration_dir.name)
        if not match:
            continue
        point_cloud = iteration_dir / "point_cloud.ply"
        if point_cloud.is_file() and point_cloud.stat().st_size > 0:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def resolve_output_iteration(output: Path, requested: Optional[int] = None) -> int:
    iterations = checkpoint_iterations(output)
    if requested is not None and requested > 0 and requested in iterations:
        return requested
    if iterations:
        return iterations[-1]
    if requested is not None and requested > 0:
        return requested
    raise ValueError(f"No valid checkpoint iteration found under {output}")
