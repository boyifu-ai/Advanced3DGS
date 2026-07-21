from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    OUTPUT_FLAGS,
    PROFILE_PATH,
    SOURCE_FLAGS,
    load_confirmed_catalog,
    load_json_list,
    resolve_project_path,
    static_declared_flags,
)


DEFAULT_RENDER_ENTRIES = ("render.py",)
DEFAULT_EVAL_ENTRIES = ("metrics.py",)


def choose_existing(repo: Path, candidates: Sequence[object]) -> Optional[Path]:
    for value in candidates:
        candidate = repo / str(value)
        if candidate.is_file():
            return candidate
    return None


def entry_record(
    repo: Path,
    candidates: Sequence[object],
    required_groups: Sequence[Sequence[str]],
) -> Dict[str, object]:
    entry = choose_existing(repo, candidates)
    record: Dict[str, object] = {
        "candidates": [str(value) for value in candidates],
        "entry": str(entry) if entry else None,
        "passed": False,
        "errors": [],
    }
    errors: List[str] = record["errors"]  # type: ignore[assignment]
    if entry is None:
        errors.append("no declared entry candidate exists")
        return record
    try:
        compile(entry.read_text(encoding="utf-8"), str(entry), "exec")
    except Exception as exc:
        errors.append(f"entry does not compile: {exc}")
        return record
    flags = static_declared_flags(repo, entry)
    record["static_declared_cli_options"] = flags
    for group in required_groups:
        if not any(flag in flags for flag in group):
            errors.append(
                "cannot statically verify required CLI option group: "
                + " / ".join(group)
            )
    record["passed"] = not errors
    return record


def read_training_coverage(path: Path) -> Dict[str, bool]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    methods = payload.get("methods", []) if isinstance(payload, dict) else []
    return {
        str(item.get("method")): bool(item.get("passed"))
        for item in methods
        if isinstance(item, dict) and item.get("method")
    }


def audit_e2e_readiness(
    coverage_report: Path,
    catalog: Optional[Sequence[Dict[str, object]]] = None,
    profiles: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    methods = list(catalog if catalog is not None else load_confirmed_catalog())
    profile_rows = list(profiles if profiles is not None else load_json_list(PROFILE_PATH))
    profiles_by_key = {str(item["key"]): item for item in profile_rows}
    training_coverage = read_training_coverage(coverage_report)
    rows: List[Dict[str, object]] = []

    for method in methods:
        key = str(method["key"])
        profile = profiles_by_key.get(key, {})
        repo = resolve_project_path(method["local_path"])
        framework_render_entry = profile.get("framework_render_entry")
        render_candidates = (
            [str((PROJECT_ROOT / str(framework_render_entry)).resolve())]
            if framework_render_entry
            else profile.get("render_entry_candidates", DEFAULT_RENDER_ENTRIES)
        )
        if not isinstance(render_candidates, list):
            render_candidates = list(DEFAULT_RENDER_ENTRIES)
        render_entry = None
        for candidate in render_candidates:
            path = Path(str(candidate))
            if not path.is_absolute():
                path = repo / path
            if path.is_file():
                render_entry = path
                break
        render = entry_record(
            repo,
            [str(render_entry)] if render_entry else render_candidates,
            (SOURCE_FLAGS, OUTPUT_FLAGS),
        )
        evaluator = PROJECT_ROOT / "scripts" / "evaluate_render_pairs.py"
        evaluate = entry_record(
            PROJECT_ROOT,
            [str(evaluator)],
            (("--output",), ("--iteration",)),
        )
        blockers: List[str] = []
        if not render["passed"]:
            blockers.append("render entry/CLI is not statically ready")
        if not evaluate["passed"]:
            blockers.append("unified evaluation entry/CLI is not statically ready")

        rows.append(
            {
                "method": key,
                "title": method.get("title", key),
                "repository": str(repo),
                "short_training_verified": training_coverage.get(key, False),
                "render": render,
                "evaluate": evaluate,
                "static_optional_capability_passed": not blockers,
                "runtime_optional_capability_verified": False,
                "blockers": blockers,
            }
        )

    statically_ready = [row for row in rows if row["static_optional_capability_passed"]]
    return {
        "method_count": len(rows),
        "short_training_verified_count": sum(
            1 for row in rows if row["short_training_verified"]
        ),
        "static_optional_capability_passed_count": len(statically_ready),
        "runtime_optional_capability_verified_count": 0,
        "all_runtime_optional_capabilities_verified": False,
        "note": (
            "Static readiness is required before the metrics acceptance run. Runtime "
            "acceptance still requires train-render-evaluate output with finite metrics."
        ),
        "methods": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit render/evaluate entry readiness for all confirmed methods. "
            "This static blocker audit never starts training."
        )
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/coverage_report.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/e2e_readiness_report.json"),
    )
    args = parser.parse_args()
    coverage_report = args.coverage_report.expanduser()
    report = args.report.expanduser()
    if not coverage_report.is_absolute():
        coverage_report = (PROJECT_ROOT / coverage_report).resolve()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()

    payload = audit_e2e_readiness(coverage_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Method static render/evaluate blocker audit")
    print(
        "Unified short training verified: "
        f"{payload['short_training_verified_count']}/{payload['method_count']}"
    )
    print(
        "Static render/eval ready: "
        f"{payload['static_optional_capability_passed_count']}/{payload['method_count']}"
    )
    print(
        "Runtime train-render-evaluate verified by this static audit: "
        "0/{0}".format(payload["method_count"])
    )
    for row in payload["methods"]:
        state = (
            "STATIC READY"
            if row["static_optional_capability_passed"]
            else "STATIC BLOCKED"
        )
        print(f"[{state}] {row['method']}")
        for blocker in row["blockers"]:
            print(f"  - {blocker}")
        for stage in ("render", "evaluate"):
            for error in row[stage]["errors"]:
                print(f"  - {stage}: {error}")
    print(f"Report: {report}")
    return (
        0
        if payload["static_optional_capability_passed_count"] == payload["method_count"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
