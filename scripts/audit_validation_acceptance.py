from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.methods.registry import available_methods


DEFAULT_SCENES = (
    "mip360/garden",
    "tandt/train",
    "deep_blending/drjohnson",
)
METRICS = ("psnr", "ssim", "lpips")


def walk_metrics(value: object) -> Iterable[Tuple[str, float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key).lower()
            if name in METRICS and isinstance(child, (int, float)):
                parsed = float(child)
                if math.isfinite(parsed):
                    yield name, parsed
            yield from walk_metrics(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_metrics(child)


def finite_metrics(pair_root: Path, iteration: int) -> Optional[Dict[str, float]]:
    path = pair_root / "method_outputs" / "results.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = payload.get(f"ours_{iteration}")
    if expected is None:
        return None
    metrics: Dict[str, float] = {}
    for name, value in walk_metrics(expected):
        metrics[name] = value
    return metrics if all(name in metrics for name in METRICS) else None


def training_failure_category(pair_root: Path) -> str:
    path = pair_root / "method_outputs" / "unified3dgs_training_report.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    classification = payload.get("failure_classification", {})
    if not isinstance(classification, dict):
        return ""
    category = str(classification.get("category", ""))
    if classification.get("objective_limit") is True:
        return "hardware_limit"
    if category in {"hardware_limit", "hardware_limit_confirmed"}:
        return "hardware_limit"
    return category


def pair_roots(root: Path, method: str, scene: str) -> List[Path]:
    family, scene_name = scene.split("/", 1)
    candidates = [root / method / family / scene_name]
    if root.is_dir():
        for worker in root.iterdir():
            if worker.is_dir() and worker.name != "progress":
                candidates.append(worker / method / family / scene_name)
    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_dir():
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def failure_tsv_category(worker_root: Path, method: str, scene: str) -> str:
    path = worker_root / "progress" / "validation_failures.tsv"
    try:
        rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    category = ""
    for row in rows[1:]:
        fields = row.split("\t")
        if len(fields) >= 7 and fields[1] == method and fields[2] == scene:
            category = fields[4]
    return category


def classify_pair(
    root: Path,
    method: str,
    scene: str,
    iteration: int,
) -> Dict[str, object]:
    candidates = pair_roots(root, method, scene)
    evidence: List[Dict[str, object]] = []
    passed: Optional[Dict[str, object]] = None
    hardware: Optional[Dict[str, object]] = None
    running: Optional[Dict[str, object]] = None
    failed: Optional[Dict[str, object]] = None

    for pair in candidates:
        metrics = finite_metrics(pair, iteration)
        row: Dict[str, object] = {
            "pair_root": str(pair),
            "metrics": metrics or {},
            "markers": sorted(path.name for path in pair.glob(".*.*")),
        }
        if (pair / ".eval.done").is_file() and metrics is not None:
            row["state"] = "metrics_passed"
            passed = row
        else:
            category = training_failure_category(pair)
            worker_root = pair.parents[2]
            tsv_category = failure_tsv_category(worker_root, method, scene)
            if category == "hardware_limit" or tsv_category == "hardware_limit":
                row["state"] = "hardware_limited"
                row["failure_category"] = "hardware_limit"
                hardware = row
            elif any(pair.glob(".*.running")):
                row["state"] = "running"
                running = row
            elif any(pair.glob(".*.failed")):
                row["state"] = "program_failed"
                row["failure_category"] = category or tsv_category or "program_error"
                failed = row
            else:
                row["state"] = "incomplete"
        evidence.append(row)

    selected = passed or hardware or running or failed
    if selected is None:
        state = "incomplete" if candidates else "not_started"
        selected = {"state": state, "metrics": {}, "pair_root": ""}
    return {
        "method": method,
        "scene": scene,
        "state": selected["state"],
        "metrics": selected.get("metrics", {}),
        "pair_root": selected.get("pair_root", ""),
        "evidence": evidence,
    }


def audit_validation_acceptance(
    root: Path,
    methods: Sequence[str],
    scenes: Sequence[str],
    iteration: int,
) -> Dict[str, object]:
    rows = [
        classify_pair(root, method, scene, iteration)
        for method in methods
        for scene in scenes
    ]
    counts = {
        state: sum(row["state"] == state for row in rows)
        for state in (
            "metrics_passed",
            "hardware_limited",
            "program_failed",
            "running",
            "incomplete",
            "not_started",
        )
    }
    blocking = sum(
        counts[state]
        for state in ("program_failed", "running", "incomplete", "not_started")
    )
    return {
        "root": str(root),
        "iterations": iteration,
        "method_count": len(methods),
        "scene_count": len(scenes),
        "expected_pair_count": len(rows),
        "counts": counts,
        "metrics_complete": counts["metrics_passed"] == len(rows),
        "framework_accepted": blocking == 0,
        "policy": (
            "Objective hardware limits are reported separately and do not count as "
            "framework program errors; they do not count as metrics passes."
        ),
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit representative train-render-eval acceptance while separating "
            "finite-metrics passes, objective hardware limits, and program errors."
        )
    )
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/validation/_acceptance")
    )
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    methods = args.methods or available_methods()
    scenes = args.scenes or list(DEFAULT_SCENES)
    report = args.report or root / "acceptance_summary.json"
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()

    payload = audit_validation_acceptance(root, methods, scenes, args.iterations)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    counts = payload["counts"]
    print("=" * 72)
    print("Unified 3DGS validation acceptance audit")
    print("=" * 72)
    print(f"Methods:          {payload['method_count']}")
    print(f"Scenes:           {payload['scene_count']}")
    print(f"Expected pairs:   {payload['expected_pair_count']}")
    print(f"Metrics passed:   {counts['metrics_passed']}")
    print(f"Hardware limited: {counts['hardware_limited']}")
    print(f"Program failed:   {counts['program_failed']}")
    print(f"Running:          {counts['running']}")
    print(f"Incomplete:       {counts['incomplete']}")
    print(f"Not started:      {counts['not_started']}")
    for state in ("hardware_limited", "program_failed", "running", "incomplete", "not_started"):
        selected = [row for row in payload["results"] if row["state"] == state]
        if selected:
            print(f"\n{state.upper()}:")
            for row in selected:
                print(f" - {row['method']}/{row['scene']}")
    print(f"\nReport: {report}")
    if payload["framework_accepted"]:
        print("FINAL RESULT: FRAMEWORK ACCEPTED (objective hardware limits separated).")
        return 0
    print("FINAL RESULT: FRAMEWORK ACCEPTANCE HAS BLOCKING ITEMS.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
