from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import load_confirmed_catalog


def audit(root: Path) -> Dict[str, object]:
    expected = sorted(str(method["key"]) for method in load_confirmed_catalog())
    evidence: Dict[str, List[Dict[str, object]]] = {key: [] for key in expected}
    for report_path in sorted(root.glob("attempts/*/acceptance_results.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in report.get("methods", []):
            key = str(row.get("method", ""))
            if key not in evidence or row.get("exit_code") != 0 or not row.get("verified"):
                continue
            training_report = Path(str(row.get("output", ""))) / "unified3dgs_training_report.json"
            try:
                details = json.loads(training_report.read_text(encoding="utf-8"))
            except Exception:
                continue
            saved = [Path(str(path)) for path in details.get("saved_files", [])]
            if (
                details.get("passed") is True
                and details.get("iterations") == 1
                and saved
                and all(path.is_file() and path.stat().st_size > 0 for path in saved)
            ):
                evidence[key].append(
                    {
                        "acceptance_report": str(report_path),
                        "training_report": str(training_report),
                        "saved_files": [str(path) for path in saved],
                    }
                )
    missing = [key for key in expected if not evidence[key]]
    return {
        "all_passed": not missing,
        "expected_method_count": len(expected),
        "passed_method_count": len(expected) - len(missing),
        "missing_or_unverified_methods": missing,
        "methods": [
            {"method": key, "passed": bool(evidence[key]), "evidence": evidence[key]}
            for key in expected
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit historical iteration=1 acceptance of configurable training interfaces."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/validation/_training_acceptance"),
    )
    args = parser.parse_args()
    root = args.root.expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    payload = audit(root)
    root.mkdir(parents=True, exist_ok=True)
    report = root / "coverage_report.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "Configurable training-interface acceptance: "
        f"{payload['passed_method_count']}/{payload['expected_method_count']}"
    )
    for key in payload["missing_or_unverified_methods"]:
        print(f"- missing or unverifiable: {key}")
    print(f"Report: {report}")
    return 0 if payload["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
