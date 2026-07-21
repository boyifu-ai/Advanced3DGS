from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> object:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc), "path": str(path)}


def log_tail(path: object, lines: int = 20) -> List[str]:
    if not path:
        return []
    log = Path(str(path))
    if not log.is_file():
        return [f"log is missing: {log}"]
    try:
        content = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"could not read log: {exc!r}"]
    return content[-lines:]


def collect_failures(root: Path) -> Dict[str, object]:
    failures: List[Dict[str, object]] = []
    extension = read_json(root / "extension_build_report.json")
    if isinstance(extension, list):
        for method in extension:
            if method.get("status") != "failed":
                continue
            record: Dict[str, object] = {
                "stage": "extension_build",
                "method": method.get("method"),
                "error": method.get("error"),
                "items": [],
            }
            for item in method.get("build_records", []):
                if item.get("status") == "failed":
                    record["items"].append(
                        {
                            "source": item.get("source"),
                            "error": item.get("error"),
                            "log": item.get("log"),
                            "log_tail": log_tail(item.get("log")),
                        }
                    )
            failures.append(record)

    dependencies = read_json(root / "python_dependency_report.json")
    if isinstance(dependencies, list):
        for method in dependencies:
            if method.get("status") != "failed":
                continue
            record = {
                "stage": "python_dependencies",
                "method": method.get("method"),
                "error": method.get("error"),
                "items": [],
            }
            for item in method.get("installs", []):
                if item.get("status") == "failed":
                    record["items"].append(
                        {
                            "package": item.get("package"),
                            "error": item.get("error"),
                            "import_error": item.get("import_error"),
                            "exit_code": item.get("exit_code"),
                            "log": item.get("log"),
                            "log_tail": log_tail(item.get("log")),
                        }
                    )
            failures.append(record)

    preflight = read_json(root / "preflight_report.json")
    if isinstance(preflight, dict):
        global_errors = preflight.get("global_errors", [])
        if global_errors:
            failures.append(
                {"stage": "preflight", "method": "GLOBAL", "errors": global_errors}
            )
        for method in preflight.get("methods", []):
            if method.get("passed") is False:
                failures.append(
                    {
                        "stage": "preflight",
                        "method": method.get("key"),
                        "errors": method.get("errors", []),
                    }
                )
    return {"failure_count": len(failures), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print one complete root-cause summary for method preparation."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/validation/_method_acceptance")
    )
    args = parser.parse_args()
    root = args.root.expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    payload = collect_failures(root)
    report = root / "failure_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    print("=" * 72)
    print("COMPLETE METHOD FAILURE SUMMARY")
    print("=" * 72)
    if not payload["failures"]:
        print("No preparation or preflight failures.")
    for record in payload["failures"]:
        print(f"\n[{record['stage']}] method={record['method']}")
        if record.get("error"):
            print(f"ERROR: {record['error']}")
        for error in record.get("errors", []):
            print(f"ERROR: {error}")
        for item in record.get("items", []):
            label = item.get("source") or item.get("package") or "item"
            print(f"ITEM: {label}")
            if item.get("error"):
                print(f"ERROR: {item['error']}")
            if item.get("import_error"):
                print(f"IMPORT ERROR: {item['import_error']}")
            if item.get("log"):
                print(f"LOG: {item['log']}")
            for line in item.get("log_tail", []):
                print(f"  {line}")
    print(f"\nFailure summary report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
