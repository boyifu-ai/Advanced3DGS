from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable


def spherical_harmonic_degree(properties: Iterable[str]) -> int:
    rest_count = sum(str(name).startswith("f_rest_") for name in properties)
    coefficients = rest_count + 3
    if coefficients % 3:
        raise ValueError(f"Invalid Mini-Splatting SH property count: {rest_count}")
    basis_count = coefficients // 3
    degree = math.isqrt(basis_count) - 1
    if degree < 0 or (degree + 1) ** 2 != basis_count:
        raise ValueError(f"Invalid Mini-Splatting SH property count: {rest_count}")
    return degree


def saved_sh_degree(model_path: Path, iteration: int) -> int:
    from plyfile import PlyData

    point_cloud = (
        model_path
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )
    if not point_cloud.is_file():
        raise FileNotFoundError(
            f"Missing Mini-Splatting point cloud for iteration {iteration}: "
            f"{point_cloud}"
        )
    properties = [
        item.name for item in PlyData.read(str(point_cloud))["vertex"].properties
    ]
    return spherical_harmonic_degree(properties)


def main() -> int:
    import render as upstream
    from arguments import ModelParams, PipelineParams, get_combined_args
    from utils.general_utils import safe_state

    parser = argparse.ArgumentParser(description="Unified Mini-Splatting renderer")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    args.eval = True

    degree = saved_sh_degree(Path(args.model_path), args.iteration)
    args.sh_degree = degree
    print(f"Detected saved Mini-Splatting SH degree: {degree}")
    safe_state(args.quiet)
    upstream.render_sets(
        model_params.extract(args),
        args.iteration,
        pipeline_params.extract(args),
        args.skip_train,
        args.skip_test,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
