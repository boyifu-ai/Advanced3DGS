from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import resolve_project_path, select_methods


PATCH_MARKER = "# Unified 3DGS Python 3.8 annotation compatibility patch"
UNUSED_IMPORT_MARKER = "# Unified 3DGS removed unused optional upstream import"


def future_import_insertion_index(text: str) -> int:
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return int(getattr(tree.body[0], "end_lineno", tree.body[0].lineno))
    insertion = 0
    if lines and lines[0].startswith("#!"):
        insertion = 1
    if len(lines) > insertion and "coding" in lines[insertion][:80]:
        insertion += 1
    return insertion


def patch_future_annotations(path: Path, run_real: bool) -> Dict[str, object]:
    record: Dict[str, object] = {"path": str(path), "patch": "future_annotations"}
    if not path.is_file():
        record.update(status="failed", error="target file is missing")
        return record
    text = path.read_text(encoding="utf-8")
    if "from __future__ import annotations" in text:
        record["status"] = "already_patched"
        return record
    insertion = future_import_insertion_index(text)
    lines = text.splitlines(keepends=True)
    lines.insert(insertion, f"from __future__ import annotations  {PATCH_MARKER}\n")
    patched = "".join(lines)
    try:
        compile(patched, str(path), "exec")
    except Exception as exc:
        record.update(status="failed", error=f"patched source does not compile: {exc}")
        return record
    if not run_real:
        record["status"] = "catalog"
        return record
    backup = path.with_name(path.name + ".bak_unified3dgs_python38")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    record.update(status="patched", backup=str(backup))
    return record


def import_references(repo: Path, module: str) -> List[Dict[str, object]]:
    references: List[Dict[str, object]] = []
    for path in sorted(repo.rglob("*.py")):
        relative = path.relative_to(repo)
        if any(part.startswith(".") for part in relative.parts) or len(relative.parts) > 6:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if f"import {module}" in line or f"from {module}" in line:
                references.append(
                    {"path": relative.as_posix(), "line": number, "text": line.strip()}
                )
    return references


def remove_unused_module_imports(
    repo: Path, module: str, run_real: bool
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "module": module,
        "patch": "remove_unused_optional_import",
        "patched_files": [],
        "used_imports": [],
    }
    failed = False
    for path in sorted(repo.rglob("*.py")):
        relative = path.relative_to(repo)
        if any(part.startswith(".") for part in relative.parts) or len(relative.parts) > 6:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        loaded_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        removable: List[ast.AST] = []
        for node in ast.walk(tree):
            imported_names: List[str] = []
            if isinstance(node, ast.Import):
                matching = [alias for alias in node.names if alias.name == module]
                if not matching or len(node.names) != len(matching):
                    continue
                imported_names = [alias.asname or module for alias in matching]
            elif isinstance(node, ast.ImportFrom) and node.module == module:
                imported_names = [
                    alias.asname or alias.name for alias in node.names if alias.name != "*"
                ]
                if len(imported_names) != len(node.names):
                    record["used_imports"].append(
                        {"path": relative.as_posix(), "line": node.lineno, "names": ["*"]}
                    )
                    continue
            else:
                continue
            used = sorted(name for name in imported_names if name in loaded_names)
            if used:
                record["used_imports"].append(
                    {"path": relative.as_posix(), "line": node.lineno, "names": used}
                )
            else:
                removable.append(node)
        if not removable:
            continue
        lines = text.splitlines(keepends=True)
        for node in sorted(removable, key=lambda item: item.lineno, reverse=True):
            start = int(node.lineno) - 1
            end = int(getattr(node, "end_lineno", node.lineno))
            indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
            original = " ".join(line.strip() for line in lines[start:end])
            lines[start:end] = [
                f"{indent}pass  {UNUSED_IMPORT_MARKER}: {original}\n"
            ]
        patched = "".join(lines)
        try:
            compile(patched, str(path), "exec")
        except Exception as exc:
            failed = True
            record.setdefault("errors", []).append(
                f"{relative.as_posix()}: patched source does not compile: {exc}"
            )
            continue
        file_record: Dict[str, object] = {
            "path": relative.as_posix(),
            "removed_imports": len(removable),
            "status": "catalog",
        }
        if run_real:
            backup = path.with_name(path.name + f".bak_unified3dgs_unused_{module}")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(patched, encoding="utf-8")
            file_record.update(status="patched", backup=str(backup))
        record["patched_files"].append(file_record)
    if failed:
        record["status"] = "failed"
    elif record["used_imports"]:
        record["status"] = "still_required"
    elif record["patched_files"]:
        record["status"] = "patched" if run_real else "catalog"
    else:
        record["status"] = "not_imported_or_already_patched"
    return record


def prepare_method(
    method: Dict[str, object], run_real: bool
) -> Tuple[Dict[str, object], bool]:
    key = str(method["key"])
    repo = resolve_project_path(method["local_path"])
    record: Dict[str, object] = {
        "method": key,
        "repo": str(repo),
        "actions": [],
        "status": "no_compatibility_action",
    }
    failed = False
    if key == "ges":
        action = patch_future_annotations(repo / "scene" / "__init__.py", run_real)
        record["actions"] = [action]
        failed = action["status"] == "failed"
        record["status"] = "failed" if failed else str(action["status"])
    if key == "beta_splatting":
        optional_import = remove_unused_module_imports(repo, "plas", run_real)
        references = import_references(repo, "plas")
        record["actions"] = [optional_import]
        record["plas_import_references"] = references
        record["status"] = str(optional_import["status"])
        failed = optional_import["status"] == "failed"
    return record, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply narrow, auditable compatibility preparation for methods."
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/compatibility_report.json"),
    )
    args = parser.parse_args()

    records: List[Dict[str, object]] = []
    failures = 0
    selected = select_methods(args.method)
    print("Method compatibility preparation", flush=True)
    print(f"Methods selected: {len(selected)}", flush=True)
    print(f"Apply changes: {args.run_real}", flush=True)
    for method, _ in selected:
        record, failed = prepare_method(method, args.run_real)
        records.append(record)
        failures += int(failed)
        print(f"[{record['status']}] {record['method']}", flush=True)

    report = args.report.expanduser()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Report: {report}")
    print(f"Failures: {failures}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
