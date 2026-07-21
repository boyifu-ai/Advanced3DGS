from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    DEFAULT_OUTPUT_ROOT,
    ITERATION_FLAGS,
    load_confirmed_catalog,
)


def read_records(path: Path) -> List[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def command_uses_iteration_one(command: object) -> bool:
    if not isinstance(command, list):
        return False
    values = [str(value) for value in command]
    for index, value in enumerate(values):
        if value in ITERATION_FLAGS and index + 1 < len(values):
            return values[index + 1] == "1"
        for flag in ITERATION_FLAGS:
            if value == f"{flag}=1":
                return True
    return False


def verified_saved_files(record: Dict[str, object]) -> Tuple[List[str], List[str]]:
    saved = record.get("saved_files", [])
    if not isinstance(saved, list) or not saved:
        return [], ["no exact iteration=1 saved files recorded"]
    verified: List[str] = []
    errors: List[str] = []
    for value in saved:
        path = Path(str(value))
        if not path.is_file():
            errors.append(f"saved file is missing: {path}")
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"cannot inspect saved file {path}: {exc!r}")
            continue
        if size <= 0:
            errors.append(f"saved file is empty: {path}")
            continue
        verified.append(str(path))
    return verified, errors


def passing_record_errors(record: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    if record.get("status") != "passed":
        errors.append(f"status is not passed: {record.get('status')!r}")
    if not command_uses_iteration_one(record.get("command")):
        errors.append("training command does not explicitly use iteration=1")
    unexpected = record.get("unexpected_iteration_artifacts", [])
    if not isinstance(unexpected, list) or unexpected:
        errors.append("unexpected higher-iteration artifacts were recorded")
    _, saved_errors = verified_saved_files(record)
    errors.extend(saved_errors)
    return errors


def audit_coverage(
    output_root: Path,
    catalog: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    methods = list(catalog if catalog is not None else load_confirmed_catalog())
    attempts_root = output_root / "attempts"
    reports = sorted(
        attempts_root.glob("*/acceptance_results.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    records_by_method: Dict[str, List[Tuple[Path, Dict[str, object]]]] = {}
    for report in reports:
        for record in read_records(report):
            method = str(record.get("method", "")).strip()
            if method:
                records_by_method.setdefault(method, []).append((report, record))

    rows: List[Dict[str, object]] = []
    for method in methods:
        key = str(method["key"])
        attempts = records_by_method.get(key, [])
        selected: Optional[Tuple[Path, Dict[str, object]]] = None
        rejected: List[Dict[str, object]] = []
        for report, record in attempts:
            errors = passing_record_errors(record)
            if not errors:
                selected = (report, record)
                break
            rejected.append(
                {
                    "report": str(report),
                    "status": record.get("status"),
                    "errors": errors,
                }
            )
        row: Dict[str, object] = {
            "method": key,
            "title": method.get("title", key),
            "passed": selected is not None,
            "attempts_found": len(attempts),
            "rejected_attempts": rejected,
        }
        if selected is not None:
            report, record = selected
            verified, _ = verified_saved_files(record)
            row.update(
                {
                    "report": str(report),
                    "output": record.get("output"),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                    "completion_mode": record.get("completion_mode"),
                    "verified_saved_files": verified,
                }
            )
        rows.append(row)

    passed = [row for row in rows if row["passed"]]
    missing = [str(row["method"]) for row in rows if not row["passed"]]
    return {
        "expected_method_count": len(rows),
        "passed_method_count": len(passed),
        "all_passed": len(passed) == len(rows),
        "missing_or_unverified_methods": missing,
        "attempt_reports_scanned": len(reports),
        "methods": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all historical method acceptance attempts and require one "
            "verifiable iteration=1 training/save pass for every confirmed method."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    report = args.report.expanduser() if args.report else output_root / "coverage_report.json"
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()

    payload = audit_coverage(output_root)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Method iteration=1 coverage audit")
    print(f"Attempt reports scanned: {payload['attempt_reports_scanned']}")
    print(
        f"Verified methods: {payload['passed_method_count']}/"
        f"{payload['expected_method_count']}"
    )
    for row in payload["methods"]:
        state = "PASS" if row["passed"] else "MISSING/UNVERIFIED"
        print(f"[{state}] {row['method']}: attempts={row['attempts_found']}")
        if row["passed"]:
            print(f"  report: {row['report']}")
            print(f"  completion: {row['completion_mode']}")
            print(f"  saved files: {len(row['verified_saved_files'])}")
    print(f"Coverage report: {report}")
    if payload["all_passed"]:
        print("All confirmed methods have verified iteration=1 training/save passes.")
        return 0
    print(
        "Missing or unverified methods: "
        + ", ".join(payload["missing_or_unverified_methods"])
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
