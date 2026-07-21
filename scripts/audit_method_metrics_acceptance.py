from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import load_confirmed_catalog


DEFAULT_DATASET_COUNT = 3
DEFAULT_DATASET_LABELS = (
    "mip360/garden",
    "tandt/train",
    "deep_blending/drjohnson",
)


def latest_report(root: Path) -> Path:
    pointer = root / "latest_attempt.json"
    if pointer.is_file():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        report = Path(str(payload["report"]))
        if not report.is_absolute():
            report = PROJECT_ROOT / report
        return report.resolve()
    reports = sorted(root.glob("attempts/*/metrics_acceptance_results.json"))
    if not reports:
        raise FileNotFoundError(f"No metrics acceptance report found under {root}")
    return reports[-1].resolve()


def audit(report_path: Path, expected_dataset_count: int = DEFAULT_DATASET_COUNT) -> Dict[str, object]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    official_report = payload.get("protocol_mode") == "official_short"
    confirmed = [
        str(method["key"])
        for method in load_confirmed_catalog()
    ]
    expected_count = len(confirmed) * expected_dataset_count
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        rows = []
    passed = []
    for row in rows:
        if (
            not official_report
            or not isinstance(row, dict)
            or row.get("passed") is not True
            or row.get("official_protocol") is not True
            or row.get("official_runtime_verified") is not True
        ):
            continue
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        values = [metrics.get(name) for name in ("psnr", "ssim", "lpips")]
        if all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values
        ):
            passed.append(row)
    failed = [
        row
        for row in rows
        if not (isinstance(row, dict) and row.get("passed") is True)
    ]
    methods_with_passes = sorted(
        {str(row.get("method")) for row in passed if isinstance(row, dict)}
    )
    missing_methods = sorted(set(confirmed) - set(methods_with_passes))
    return {
        "report": str(report_path),
        "expected_method_count": len(confirmed),
        "expected_dataset_count": expected_dataset_count,
        "expected_result_count": expected_count,
        "observed_result_count": len(rows),
        "passed_result_count": len(passed),
        "all_passed": len(passed) == expected_count and not failed and not missing_methods,
        "missing_methods": missing_methods,
        "failed_results": failed,
    }


def audit_all_reports(
    root: Path,
    expected_dataset_labels: Sequence[str] = DEFAULT_DATASET_LABELS,
) -> Dict[str, object]:
    confirmed = [str(method["key"]) for method in load_confirmed_catalog()]
    expected_pairs = {
        (method, dataset)
        for method in confirmed
        for dataset in expected_dataset_labels
    }
    latest_rows: Dict[tuple[str, str], Dict[str, object]] = {}
    reports = sorted(root.glob("attempts/*/metrics_acceptance_results.json"))
    for report_path in reports:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        official_report = (
            isinstance(payload, dict)
            and payload.get("protocol_mode") == "official_short"
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = (str(row.get("method", "")), str(row.get("dataset", "")))
            if key in expected_pairs:
                latest_rows[key] = {
                    **row,
                    "acceptance_report": str(report_path),
                    "official_report": official_report,
                }

    passed_pairs = set()
    failed_results: List[Dict[str, object]] = []
    for pair in sorted(expected_pairs):
        row = latest_rows.get(pair)
        if row is None:
            continue
        metrics = row.get("metrics", {})
        values = (
            [metrics.get(name) for name in ("psnr", "ssim", "lpips")]
            if isinstance(metrics, dict)
            else []
        )
        valid = (
            row.get("official_report") is True
            and row.get("official_protocol") is True
            and row.get("official_runtime_verified") is True
            and row.get("passed") is True
            and len(values) == 3
            and all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in values
            )
        )
        if valid:
            passed_pairs.add(pair)
        else:
            failed_results.append(row)

    missing_pairs = sorted(expected_pairs - set(latest_rows))
    missing_methods = sorted(
        {
            method
            for method, _ in expected_pairs - passed_pairs
        }
    )
    return {
        "reports_scanned": len(reports),
        "expected_method_count": len(confirmed),
        "expected_dataset_count": len(expected_dataset_labels),
        "expected_result_count": len(expected_pairs),
        "observed_result_count": len(latest_rows),
        "passed_result_count": len(passed_pairs),
        "all_passed": passed_pairs == expected_pairs,
        "missing_methods": missing_methods,
        "missing_pairs": [
            {"method": method, "dataset": dataset}
            for method, dataset in missing_pairs
        ],
        "failed_results": failed_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit method train-render-eval metrics acceptance evidence."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/validation/_method_acceptance"),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expected-dataset-count", type=int, default=DEFAULT_DATASET_COUNT)
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    report_path = args.report.expanduser().resolve() if args.report else None
    payload = (
        audit(report_path, args.expected_dataset_count)
        if report_path
        else audit_all_reports(root)
    )
    output = root / "coverage_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Method metrics acceptance audit")
    print(f"Report: {report_path or 'merged historical attempts'}")
    print(f"Passed: {payload['passed_result_count']}/{payload['expected_result_count']}")
    if payload["missing_methods"]:
        print("Missing method evidence: " + ", ".join(payload["missing_methods"]))
    if payload["failed_results"]:
        print(f"Failed method/dataset results: {len(payload['failed_results'])}")
        for row in payload["failed_results"][:20]:
            if isinstance(row, dict):
                print(f"- {row.get('method')} {row.get('dataset')}: {row.get('metric_error') or row.get('stages')}")
    print(f"Coverage report: {output}")
    return 0 if payload["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
