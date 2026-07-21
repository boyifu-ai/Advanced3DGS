from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.utils.iterations import resolve_output_iteration


METRICS = ("psnr", "ssim", "lpips")
METHOD_ORDER = ("vanilla_3dgs", "2dgs", "3dgs_mcmc", "3dhgs", "sss")
AGGREGATION_EXCLUDED_SCENES = {
    "mip360": {"flowers", "treehill"},
}
LOG_PATTERN = re.compile(
    r"\b(?P<name>psnr|ssim|lpips)\b\s*[:=]?\s*(?P<value>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _float_or_none(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _walk_json_metrics(obj: object) -> Iterable[Tuple[str, float]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lower_key = str(key).lower()
            if lower_key in METRICS:
                parsed = _float_or_none(value)
                if parsed is not None:
                    yield lower_key, parsed
            yield from _walk_json_metrics(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json_metrics(item)


def _metrics_from_json(
    path: Path,
    iteration: Optional[int] = None,
) -> Tuple[Dict[str, float], Optional[str]]:
    metrics: Dict[str, float] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return metrics, None
    selected = data
    if iteration is not None and isinstance(data, dict):
        expected_key = f"ours_{iteration}"
        if expected_key not in data:
            return metrics, None
        selected = data[expected_key]
    for name, value in _walk_json_metrics(selected):
        metrics[name] = value
    return metrics, str(path)


def _metrics_from_log(path: Path) -> Tuple[Dict[str, float], Optional[str]]:
    metrics: Dict[str, float] = {}
    if not path.exists():
        return metrics, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return metrics, None
    for match in LOG_PATTERN.finditer(text):
        metrics[match.group("name").lower()] = float(match.group("value"))
    return metrics, str(path) if metrics else None


def collect_pair_metrics(
    pair_root: Path,
    iteration: Optional[int] = None,
) -> Tuple[Dict[str, float], str]:
    json_names = ("results.json", "metrics.json", "eval.json")
    for json_path in sorted(pair_root.rglob("*.json")):
        if json_path.name not in json_names:
            continue
        metrics, source = _metrics_from_json(json_path, iteration)
        if all(name in metrics for name in METRICS):
            return metrics, source or str(json_path)

    if iteration is not None:
        return {}, ""

    for log_name in ("eval.log", "render.log", "train.log"):
        metrics, source = _metrics_from_log(pair_root / log_name)
        if all(name in metrics for name in METRICS):
            return metrics, source or str(pair_root / log_name)

    return {}, ""


def method_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return METHOD_ORDER.index(path.name), path.name
    except ValueError:
        return len(METHOD_ORDER), path.name


def pair_sort_key(method_dir: Path, pair_dir: Path) -> str:
    return pair_dir.relative_to(method_dir).as_posix()


def included_in_group_averages(row: Dict[str, object]) -> bool:
    dataset_label = str(row["dataset"])
    parts = dataset_label.split("/", 1)
    if len(parts) != 2:
        return True
    dataset_family, scene = parts
    return scene not in AGGREGATION_EXCLUDED_SCENES.get(dataset_family, set())


def is_pair_output_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "method_outputs").is_dir():
        return True
    return any((path / name).is_file() for name in ("train.log", "render.log", "eval.log"))


def iter_pair_output_dirs(method_dir: Path) -> List[Path]:
    pair_dirs = [p for p in method_dir.rglob("*") if is_pair_output_dir(p)]
    if is_pair_output_dir(method_dir):
        pair_dirs.append(method_dir)
    return sorted(set(pair_dirs), key=lambda p: pair_sort_key(method_dir, p))


def collect_rows(
    validation_root: Path,
    iteration: Optional[int] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not validation_root.exists():
        return rows
    for method_dir in sorted(
        [p for p in validation_root.iterdir() if p.is_dir()],
        key=method_sort_key,
    ):
        for pair_dir in iter_pair_output_dirs(method_dir):
            dataset_label = pair_dir.relative_to(method_dir).as_posix()
            output = pair_dir / "method_outputs"
            if iteration is not None:
                resolved_iteration = iteration
            else:
                try:
                    resolved_iteration = resolve_output_iteration(output)
                except ValueError:
                    resolved_iteration = None
            metrics, source = collect_pair_metrics(pair_dir, resolved_iteration)
            row: Dict[str, object] = {
                "level": "scene",
                "method": method_dir.name,
                "dataset": dataset_label,
                "iteration": resolved_iteration,
                "psnr": metrics.get("psnr"),
                "ssim": metrics.get("ssim"),
                "lpips": metrics.get("lpips"),
                "status": "ok" if all(name in metrics for name in METRICS) else "missing",
                "source": source,
                "path": str(pair_dir),
            }
            rows.append(row)
    return rows


def add_average_rows(
    rows: List[Dict[str, object]],
    average_level: str,
    average_label: str,
) -> List[Dict[str, object]]:
    result = list(rows)
    methods = sorted({str(row["method"]) for row in rows}, key=lambda name: method_sort_key(Path(name)))
    input_level = "scene" if average_level == "dataset" else "dataset"
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        result.append(
            make_average_row(
                method_rows,
                method,
                average_label,
                f"mean over {len(method_rows)} {input_level} rows",
                average_level,
            )
        )
    return result


def make_average_row(
    rows: List[Dict[str, object]],
    method: str,
    dataset_label: str,
    source: str,
    level: str,
) -> Dict[str, object]:
    avg_row: Dict[str, object] = {
        "level": level,
        "dataset": dataset_label,
        "method": method,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "status": "missing",
        "source": source,
    }
    all_present = bool(rows) and all(row.get("status") == "ok" for row in rows)
    for metric in METRICS:
        values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
        if values:
            avg_row[metric] = mean(values)
        if len(values) != len(rows) or len(values) == 0:
            all_present = False
    avg_row["status"] = "ok" if all_present else "partial"
    return avg_row


def format_value(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return ""


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "level",
                "dataset",
                "iteration",
                "psnr",
                "ssim",
                "lpips",
                "status",
                "source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "level": row.get("level") or "",
                    "dataset": row["dataset"],
                    "iteration": row.get("iteration") or "",
                    "psnr": format_value(row.get("psnr")),
                    "ssim": format_value(row.get("ssim")),
                    "lpips": format_value(row.get("lpips")),
                    "status": row["status"],
                    "source": row["source"],
                }
            )


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Metrics Summary",
        "",
        "| Method | Level | Dataset / Scope | Final Iteration | PSNR | SSIM | LPIPS | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {level} | {dataset} | {iteration} | {psnr} | {ssim} | {lpips} | {status} |".format(
                method=row["method"],
                level=row.get("level") or "",
                dataset=row["dataset"],
                iteration=row.get("iteration") or "",
                psnr=format_value(row.get("psnr")),
                ssim=format_value(row.get("ssim")),
                lpips=format_value(row.get("lpips")),
                status=row["status"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_row(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "method": row["method"],
        "level": row.get("level"),
        "dataset": row["dataset"],
        "iteration": row.get("iteration"),
        "psnr": row.get("psnr"),
        "ssim": row.get("ssim"),
        "lpips": row.get("lpips"),
        "status": row["status"],
        "source": row.get("source", ""),
    }


def write_rows_json(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_direction": {
            "psnr": "higher_is_better",
            "ssim": "higher_is_better",
            "lpips": "lower_is_better",
        },
        "rows": [compact_row(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_group_summary(
    rows: List[Dict[str, object]],
    output_dir: Path,
    include_average: bool = True,
    average_level: str = "dataset",
    average_label: str = "AVERAGE",
) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = (
        add_average_rows(rows, average_level, average_label)
        if include_average
        else list(rows)
    )
    write_csv(output_rows, output_dir / "metrics_summary.csv")
    write_markdown(output_rows, output_dir / "metrics_summary.md")
    write_rows_json(output_rows, output_dir / "metrics_summary.json")


def write_group_summaries(
    rows: List[Dict[str, object]],
    validation_root: Path,
    write_dataset: bool = True,
    write_method: bool = True,
) -> None:
    by_method_dataset: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

    for row in rows:
        if not included_in_group_averages(row):
            continue
        method = str(row["method"])
        dataset = str(row["dataset"])
        dataset_family = dataset.split("/", 1)[0]
        by_method_dataset.setdefault((method, dataset_family), []).append(row)

    method_dataset_rows: Dict[str, List[Dict[str, object]]] = {}
    for (method, dataset_family), dataset_rows in by_method_dataset.items():
        if write_dataset:
            write_group_summary(
                dataset_rows,
                validation_root / method / dataset_family,
                average_level="dataset",
                average_label="AVERAGE",
            )
        dataset_avg = make_average_row(
            dataset_rows,
            method,
            dataset_family,
            f"mean over {len(dataset_rows)} scene rows",
            "dataset",
        )
        method_dataset_rows.setdefault(method, []).append(dataset_avg)

    if write_method:
        for method, dataset_rows in method_dataset_rows.items():
            output_rows = list(dataset_rows)
            output_rows.append(
                make_average_row(
                    dataset_rows,
                    method,
                    "AVERAGE",
                    f"mean over {len(dataset_rows)} dataset rows",
                    "method",
                )
            )
            output_dir = validation_root / method
            write_csv(output_rows, output_dir / "metrics_summary.csv")
            write_markdown(output_rows, output_dir / "metrics_summary.md")
            write_rows_json(output_rows, output_dir / "metrics_summary.json")


def write_scene_summary(row: Dict[str, object]) -> None:
    pair_dir_value = row.get("path")
    if not isinstance(pair_dir_value, str) or not pair_dir_value:
        return
    pair_dir = Path(pair_dir_value)
    pair_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": row["method"],
        "level": "scene",
        "dataset": row["dataset"],
        "iteration": row.get("iteration"),
        "status": row["status"],
        "metrics": {
            "psnr": row.get("psnr"),
            "ssim": row.get("ssim"),
            "lpips": row.get("lpips"),
        },
        "metric_direction": {
            "psnr": "higher_is_better",
            "ssim": "higher_is_better",
            "lpips": "lower_is_better",
        },
        "source": row.get("source", ""),
    }
    (pair_dir / "metrics_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Scene Metrics",
        "",
        f"- Method: `{row['method']}`",
        f"- Dataset / scene: `{row['dataset']}`",
        f"- Final iteration: `{row.get('iteration', '')}`",
        f"- Status: `{row['status']}`",
        f"- Source: `{row.get('source', '')}`",
        "",
        "| Metric | Value | Better Direction |",
        "| --- | ---: | --- |",
        f"| PSNR | {format_value(row.get('psnr'))} | Higher is better |",
        f"| SSIM | {format_value(row.get('ssim'))} | Higher is better |",
        f"| LPIPS | {format_value(row.get('lpips'))} | Lower is better |",
        "",
    ]
    if row["status"] != "ok":
        lines.append("This scene is missing one or more required metrics.")
        lines.append("")
    (pair_dir / "metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_scene_summaries(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        write_scene_summary(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate PSNR/SSIM/LPIPS validation metrics.")
    parser.add_argument("--validation-root", default="outputs/validation")
    parser.add_argument(
        "--iteration",
        type=int,
        default=30000,
        help="Required formal checkpoint iteration. Default: 30000.",
    )
    parser.add_argument(
        "--no-scene-summaries",
        action="store_true",
        help="Do not write metrics_summary.json/md into each method/dataset/scene directory.",
    )
    parser.add_argument(
        "--no-group-summaries",
        action="store_true",
        help="Do not write per-method or per-dataset metrics_summary files.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=("scene", "dataset", "method"),
        default=("scene", "dataset", "method"),
        help="Summary levels to write. Default: all three levels.",
    )
    args = parser.parse_args()

    validation_root = Path(args.validation_root)
    scene_rows = collect_rows(validation_root, args.iteration)
    levels = set(args.levels)
    if "scene" in levels and not args.no_scene_summaries:
        write_scene_summaries(scene_rows)
    if not args.no_group_summaries:
        write_group_summaries(
            scene_rows,
            validation_root,
            write_dataset="dataset" in levels,
            write_method="method" in levels,
        )
    print(f"Wrote metric summaries under: {validation_root}")
    missing = [row for row in scene_rows if row["status"] in {"missing", "partial"}]
    if missing:
        print(f"Scene rows with missing or partial metrics: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
