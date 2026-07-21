from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRICS = ("psnr", "ssim", "lpips")

from unified3dgs.metrics_io import discover_render_pair


def require_newer_than(path: Path, newer_than: Optional[Path]) -> None:
    if newer_than is None:
        return
    if not newer_than.is_file():
        raise ValueError(f"Missing stage-start timestamp: {newer_than}")
    if path.stat().st_mtime_ns <= newer_than.stat().st_mtime_ns:
        raise ValueError(f"Output was not refreshed by the current stage: {path}")


def newest_file(path: Path) -> Path:
    files = [item for item in path.rglob("*") if item.is_file()]
    if not files:
        raise ValueError(f"No files found under: {path}")
    return max(files, key=lambda item: item.stat().st_mtime_ns)


def verify_render(output: Path, requested_iteration: int, newer_than: Optional[Path]) -> None:
    pair = discover_render_pair(output, requested_iteration)
    render_count = len(pair.pairs)
    require_newer_than(newest_file(pair.renders), newer_than)
    require_newer_than(newest_file(pair.gt), newer_than)
    print(
        f"Verified render outputs: {pair.root} "
        f"renders={render_count} gt={render_count}"
    )


def verify_train(output: Path, requested_iteration: int, newer_than: Optional[Path]) -> None:
    iteration = requested_iteration
    point_cloud = output / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if point_cloud.is_file() and point_cloud.stat().st_size > 0:
        require_newer_than(point_cloud, newer_than)
        print(f"Verified training output: {point_cloud}")
        return

    report_path = output / "unified3dgs_training_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        saved_files = [
            Path(str(path))
            for path in report.get("saved_files", [])
            if Path(str(path)).is_file() and Path(str(path)).stat().st_size > 0
        ]
        if (
            report.get("passed") is True
            and int(report.get("iterations", -1)) == iteration
            and saved_files
        ):
            for saved in saved_files:
                require_newer_than(saved, newer_than)
            print(
                f"Verified training output from unified report: "
                f"{len(saved_files)} saved file(s)"
            )
            return
    raise ValueError(
        f"Missing final point cloud and no valid unified training report: {point_cloud}"
    )


def walk_metrics(value: object) -> Iterable[tuple[str, float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            lower_key = str(key).lower()
            if lower_key in METRICS and isinstance(child, (int, float)):
                parsed = float(child)
                if math.isfinite(parsed):
                    yield lower_key, parsed
            yield from walk_metrics(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_metrics(child)


def verify_eval(output: Path, requested_iteration: int, newer_than: Optional[Path]) -> None:
    iteration = requested_iteration
    results_path = output / "results.json"
    if not results_path.is_file():
        raise ValueError(f"Missing evaluation results: {results_path}")
    require_newer_than(results_path, newer_than)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    expected_key = f"ours_{iteration}"
    if not isinstance(data, dict) or expected_key not in data:
        raise ValueError(f"Missing {expected_key} metrics in {results_path}")
    metrics: Dict[str, float] = {}
    for name, value in walk_metrics(data[expected_key]):
        metrics[name] = value
    missing = [name for name in METRICS if name not in metrics]
    if missing:
        raise ValueError(f"Missing or non-finite metrics in {results_path}: {missing}")
    print(
        "Verified evaluation metrics: "
        + ", ".join(f"{name}={metrics[name]:.6f}" for name in METRICS)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("train", "render", "eval"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--newer-than", type=Path, default=None)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if args.stage == "train":
        verify_train(output, args.iteration, args.newer_than)
    elif args.stage == "render":
        verify_render(output, args.iteration, args.newer_than)
    else:
        verify_eval(output, args.iteration, args.newer_than)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
