from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.install_catalog_method_extensions import disk_error, run_streaming_build
from unified3dgs.method_catalog import (
    DEFAULT_MIN_FREE_DISK_GB,
    build_method_env,
    resolve_project_path,
    select_methods,
)


DEPENDENCIES_PATH = PROJECT_ROOT / "configs" / "method_python_dependencies.json"
PREPARATION_REVISION = "catalog-deps-r13"
PACKAGE_IMPORT_OVERRIDES = {
    "protobuf": "google.protobuf",
    "pyyaml": "yaml",
}


def package_requirement(package: object) -> str:
    if isinstance(package, dict):
        return str(package.get("requirement", "")).strip()
    return str(package).strip()


def package_import_name(package: object) -> str:
    if isinstance(package, dict) and package.get("import_name"):
        return str(package["import_name"]).strip()
    name = re.split(r"[<>=!~;\[]", package_requirement(package), maxsplit=1)[0].strip()
    normalized = name.lower().replace("-", "_").replace(".", "_")
    return PACKAGE_IMPORT_OVERRIDES.get(normalized, normalized)


def package_version_requirement(package: object) -> str:
    if isinstance(package, dict) and "version_requirement" in package:
        return str(package.get("version_requirement", "")).strip()
    return package_requirement(package)


def package_import_probe(
    package: object, env: Dict[str, str], isolated_target: Optional[Path] = None
) -> Tuple[bool, str]:
    module = package_import_name(package)
    version_requirement = package_version_requirement(package)
    require_isolated = isinstance(package, dict) and package.get("require_isolated") is True
    isolated_target_value = str(isolated_target) if isolated_target else ""
    if not module:
        return False, "empty import module name"
    probe = (
        "import importlib, importlib.util, importlib.metadata, os, sys, traceback; "
        "from packaging.requirements import Requirement; "
        f"requirement={version_requirement!r}; "
        f"spec=importlib.util.find_spec({module!r}); "
        "found=spec is not None; "
        f"\nif not found: print('module not found: {module}', file=sys.stderr); sys.exit(1)\n"
        f"isolated_target={isolated_target_value!r}\n"
        f"require_isolated={require_isolated!r}\n"
        "if require_isolated:\n"
        "    candidates=[]\n"
        "    if spec.origin: candidates.append(spec.origin)\n"
        "    if spec.submodule_search_locations: candidates.extend(spec.submodule_search_locations)\n"
        "    root=os.path.realpath(isolated_target)\n"
        "    inside=any(os.path.commonpath((root, os.path.realpath(path))) == root for path in candidates)\n"
        "    if not inside: print('module does not resolve from isolated target', file=sys.stderr); sys.exit(1)\n"
        "try:\n"
        f"    importlib.import_module({module!r})\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
        "if not requirement: sys.exit(0)\n"
        "req=Requirement(requirement)\n"
        "if not req.specifier: sys.exit(0)\n"
        "try:\n"
        "    version=importlib.metadata.version(req.name)\n"
        "except importlib.metadata.PackageNotFoundError:\n"
        "    print('package metadata not found for version check', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "if not req.specifier.contains(version, prereleases=True):\n"
        "    print(f'version mismatch: {version} does not satisfy {req.specifier}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, repr(exc)
    return result.returncode == 0, result.stdout.strip()


def package_is_importable(
    package: object, env: Dict[str, str], isolated_target: Optional[Path] = None
) -> bool:
    return package_import_probe(package, env, isolated_target)[0]


def load_dependencies() -> Dict[str, Dict[str, object]]:
    data = json.loads(DEPENDENCIES_PATH.read_text(encoding="utf-8"))
    return {str(item["key"]): dict(item) for item in data}


def resolved_vcs_commit(log: Path) -> str:
    if not log.is_file():
        return ""
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Resolved \S+ to commit ([0-9a-fA-F]{7,40})", text)
    return matches[-1].lower() if matches else ""


def moving_vcs_ref(package: object) -> bool:
    if not isinstance(package, dict) or package.get("allow_moving_ref") is not True:
        return False
    requirement = package_requirement(package)
    return bool(re.search(r"@(main|master)(?:#|$)", requirement))


def install_method(
    method: Dict[str, object],
    dependency: Dict[str, object],
    run_real: bool,
    timeout_seconds: int,
    min_free_disk_gb: float,
    pip_timeout_seconds: int,
    pip_retries: int,
) -> Dict[str, object]:
    key = str(method["key"])
    repo = resolve_project_path(method["local_path"])
    packages = dependency.get("packages", [])
    blockers = dependency.get("manual_blockers", [])
    record: Dict[str, object] = {
        "method": key,
        "repo": str(repo),
        "packages": packages,
        "manual_blockers": blockers,
        "installs": [],
        "status": "catalog",
    }
    if not isinstance(packages, list) or not isinstance(blockers, list):
        record["status"] = "failed"
        record["error"] = "dependency profile packages/manual_blockers must be lists"
        return record
    if not packages:
        record["status"] = "manual_blockers" if blockers else "no_packages"
        return record
    if not run_real:
        return record

    target = PROJECT_ROOT / "third_party_build" / key / "site-packages"
    target.mkdir(parents=True, exist_ok=True)
    provenance_path = target / ".unified3dgs_python_dependencies.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception:
        provenance = {}
    if not isinstance(provenance, dict):
        provenance = {}
    env = build_method_env(key, repo)
    shared_pip_cache = (
        PROJECT_ROOT / "outputs" / "method_acceptance" / "runtime" / "shared_python_pip_cache"
    )
    shared_pip_cache.mkdir(parents=True, exist_ok=True)
    env["PIP_CACHE_DIR"] = str(shared_pip_cache)
    installs: List[Dict[str, object]] = []
    for index, package in enumerate(packages, 1):
        requirement = package_requirement(package)
        print()
        print(f"[{key}] Python package {index}/{len(packages)}: {requirement}", flush=True)
        if not requirement:
            installs.append(
                {"package": package, "status": "failed", "error": "empty requirement"}
            )
            continue
        previous = provenance.get(requirement)
        requires_commit_record = moving_vcs_ref(package) and not (
            isinstance(previous, dict) and previous.get("resolved_vcs_commit")
        )
        if package_is_importable(package, env, target) and not requires_commit_record:
            print(f"Already importable; skipping download: {requirement}", flush=True)
            package_record = {"package": requirement, "status": "already_importable"}
            if isinstance(previous, dict):
                for name in ("resolved_vcs_commit", "installed_at", "log"):
                    if previous.get(name):
                        package_record[name] = previous[name]
            installs.append(package_record)
            continue
        if requires_commit_record:
            print(
                f"Reinstalling moving VCS dependency to record resolved commit: {requirement}",
                flush=True,
            )
        low_disk = disk_error(min_free_disk_gb)
        if low_disk:
            print(f"[{key}] ERROR: {low_disk}", flush=True)
            installs.append({"package": package, "status": "failed", "error": low_disk})
            continue
        log = PROJECT_ROOT / "third_party_build" / key / f"python_dep_{index:02d}.log"
        command = [
            "timeout",
            "--signal=INT",
            "--kill-after=30s",
            str(timeout_seconds),
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--disable-pip-version-check",
            "--retries",
            str(pip_retries),
            "--timeout",
            str(pip_timeout_seconds),
            "--target",
            str(target),
        ]
        if not (isinstance(package, dict) and package.get("install_dependencies") is True):
            command.append("--no-deps")
        if isinstance(package, dict) and package.get("find_links"):
            command.extend(["--find-links", str(package["find_links"])])
        if isinstance(package, dict) and package.get("only_binary") is True:
            command.extend(["--only-binary", ":all:"])
        if isinstance(package, dict) and package.get("no_build_isolation") is True:
            command.append("--no-build-isolation")
        command.append(requirement)
        print("Command:", " ".join(command), flush=True)
        print(f"Live log: {log}", flush=True)
        status = run_streaming_build(command, env, log, key, requirement)
        import_ok = False
        import_output = ""
        if status == 0:
            import_ok, import_output = package_import_probe(package, env, target)
        install_record = {
            "package": requirement,
            "status": "installed" if import_ok else "failed",
            "exit_code": status,
            "log": str(log),
            "import_validation": "passed" if import_ok else "failed",
        }
        commit = resolved_vcs_commit(log)
        if commit:
            install_record["resolved_vcs_commit"] = commit
        if import_ok:
            install_record["installed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
            provenance[requirement] = dict(install_record)
        elif import_output:
            install_record["import_error"] = import_output[-4000:]
        installs.append(install_record)
        if status == 0 and not import_ok:
            print(
                f"Install command succeeded but import/version validation failed: {requirement}",
                flush=True,
            )
            if import_output:
                print(import_output[-4000:], flush=True)

    # A successful pip install can initially import-fail because another
    # explicitly declared package later in this profile completes its runtime
    # dependency closure. Recheck those records after the full profile so
    # package ordering cannot create a permanent false failure.
    for package, install_record in zip(packages, installs):
        if (
            install_record.get("status") != "failed"
            or install_record.get("exit_code") != 0
        ):
            continue
        import_ok, import_output = package_import_probe(package, env, target)
        if not import_ok:
            if import_output:
                install_record["import_error"] = import_output[-4000:]
            continue
        requirement = package_requirement(package)
        print(
            f"Recovered after complete dependency closure: {requirement}",
            flush=True,
        )
        install_record["status"] = "installed_after_dependency_closure"
        install_record["import_validation"] = "passed"
        install_record["revalidated_after_all_packages"] = True
        install_record.pop("import_error", None)
        install_record["installed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        provenance[requirement] = dict(install_record)

    record["installs"] = installs
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record["provenance_manifest"] = str(provenance_path)
    failures = [item for item in installs if item["status"] == "failed"]
    if failures:
        record["status"] = "failed"
        record["error"] = f"{len(failures)} isolated Python dependency installs failed"
    elif blockers:
        record["status"] = "installed_with_manual_blockers"
    else:
        record["status"] = "installed"
    record["target"] = str(target)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install low-risk method Python dependencies into isolated targets."
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--min-free-disk-gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB
    )
    parser.add_argument(
        "--pip-timeout-seconds",
        type=int,
        default=120,
        help="Per-request pip network timeout. Default: 120 seconds.",
    )
    parser.add_argument(
        "--pip-retries",
        type=int,
        default=10,
        help="Number of pip network retries. Default: 10.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/python_dependency_report.json"),
    )
    args = parser.parse_args()

    selected = select_methods(args.method)
    dependencies = load_dependencies()
    selected_keys = {str(method["key"]) for method, _ in selected}
    missing = sorted(selected_keys - set(dependencies))
    if missing:
        raise ValueError(f"Missing Python dependency profiles: {', '.join(missing)}")

    report = args.report.expanduser()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    print(f"Dependency preparation revision: {PREPARATION_REVISION}", flush=True)
    for index, (method, _) in enumerate(selected, 1):
        print()
        print("=" * 72, flush=True)
        print(
            f"PYTHON DEPENDENCIES {index}/{len(selected)}: "
            f"{method['title']} [{method['key']}]",
            flush=True,
        )
        print("=" * 72, flush=True)
        try:
            record = install_method(
                method,
                dependencies[str(method["key"])],
                args.run_real,
                args.timeout_seconds,
                args.min_free_disk_gb,
                args.pip_timeout_seconds,
                args.pip_retries,
            )
        except Exception as exc:
            record = {
                "method": str(method["key"]),
                "status": "failed",
                "installs": [],
                "manual_blockers": [],
                "error": f"unexpected dependency preparation exception: {exc!r}",
            }
            print(f"[{method['key']}] ERROR: {record['error']}", flush=True)
        records.append(record)
        report.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[{record['status']}] {record['method']}", flush=True)
        for blocker in record.get("manual_blockers", []):
            print(f"  MANUAL BLOCKER: {blocker}", flush=True)
        if record.get("error"):
            print(f"  ERROR: {record['error']}", flush=True)

    failures = [record for record in records if record["status"] == "failed"]
    blockers = [record for record in records if record.get("manual_blockers")]
    print(f"Report: {report}")
    print(f"Install failures: {len(failures)}")
    print(f"Methods with manual compatibility blockers: {len(blockers)}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
