from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from queue import Empty, Queue
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    DEFAULT_MIN_FREE_DISK_GB,
    build_method_env,
    catalog_extension_spec,
    resolve_project_path,
    select_methods,
)


def build_signature(
    repo: Path, sources: List[str], modules: List[str], env: Dict[str, str]
) -> Dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
    except Exception as exc:
        torch_version = f"unavailable: {exc!r}"
        torch_cuda = None
    return {
        "repository_commit": commit,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "cuda_home": env.get("CUDA_HOME", ""),
        "torch_cuda_arch_list": env.get("TORCH_CUDA_ARCH_LIST", ""),
        "sources": sources,
        "modules": modules,
    }


def disk_error(min_free_disk_gb: float) -> str:
    free = shutil.disk_usage(PROJECT_ROOT).free
    if free < min_free_disk_gb * 1024**3:
        return (
            f"free disk is below {min_free_disk_gb:g} GiB before extension build: "
            f"{free / 1024**3:.2f} GiB"
        )
    return ""


def clean_copied_source(path: Path) -> None:
    root = path.resolve()
    for pattern in ("build", "dist", "*.egg-info", "__pycache__"):
        for target in path.glob(pattern):
            target.resolve().relative_to(root)
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()


def extract_archive_safely(archive: Path, destination: Path) -> Path:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            target.relative_to(destination_root)
        handle.extractall(destination)
    setup_files = sorted(destination.rglob("setup.py"), key=lambda path: len(path.parts))
    if not setup_files:
        raise ValueError(f"archive contains no setup.py: {archive}")
    return setup_files[0].parent


def install_module_aliases(
    aliases: object, target: Path, env: Dict[str, str]
) -> List[Dict[str, object]]:
    if not isinstance(aliases, list):
        raise ValueError("profile module_aliases must be a list")
    records: List[Dict[str, object]] = []
    marker = "# Unified 3DGS isolated module alias"
    for item in aliases:
        if (
            not isinstance(item, dict)
            or not item.get("source")
            or not item.get("target")
        ):
            raise ValueError("module alias entry must contain source and target modules")
        source = str(item["source"])
        alias = str(item["target"])
        submodules = item.get("submodules", [])
        if not isinstance(submodules, list):
            raise ValueError(f"module alias submodules must be a list: {alias}")
        alias_dir = target.joinpath(*alias.split("."))
        alias_init = alias_dir / "__init__.py"
        if alias_init.is_file() and marker not in alias_init.read_text(
            encoding="utf-8", errors="replace"
        ):
            raise ValueError(f"refusing to overwrite non-framework module alias: {alias_init}")
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias_init.write_text(
            marker
            + "\n"
            + "import importlib as _importlib\n"
            + "import sys as _sys\n"
            + f"_source = _importlib.import_module({source!r})\n"
            + "for _name in dir(_source):\n"
            + "    if not _name.startswith('__'):\n"
            + "        globals()[_name] = getattr(_source, _name)\n"
            + f"for _submodule in {submodules!r}:\n"
            + "    _sys.modules[__name__ + '.' + _submodule] = "
            + f"_importlib.import_module({source!r} + '.' + _submodule)\n",
            encoding="utf-8",
        )
        probe_names = [alias] + [f"{alias}.{name}" for name in submodules]
        probe = (
            "import importlib; "
            f"names={probe_names!r}; "
            "[importlib.import_module(name) for name in names]; "
            "print(', '.join(names))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            raise ValueError(
                f"module alias import validation failed for {alias}: "
                f"{(result.stdout + result.stderr).strip()}"
            )
        records.append(
            {
                "source": source,
                "target": alias,
                "submodules": submodules,
                "path": str(alias_init),
            }
        )
    return records


def run_streaming_build(
    command: List[str],
    env: Dict[str, str],
    log_path: Path,
    method: str,
    source: str,
    heartbeat_seconds: int = 30,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    queue: Queue[Optional[str]] = Queue()
    build_env = dict(env)
    build_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=build_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    def read_output() -> None:
        for line in process.stdout:
            queue.put(line)
        queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    last_output = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        while True:
            try:
                line = queue.get(timeout=heartbeat_seconds)
            except Empty:
                elapsed = int(time.monotonic() - started)
                quiet = int(time.monotonic() - last_output)
                message = (
                    f"[heartbeat] method={method} extension={source} "
                    f"elapsed={elapsed}s no_output_for={quiet}s\n"
                )
                print(message, end="", flush=True)
                handle.write(message)
                handle.flush()
                continue
            if line is None:
                break
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            last_output = time.monotonic()

    reader.join(timeout=5)
    return process.wait()


def install_method(
    method: Dict[str, object],
    profile: Dict[str, object],
    run_real: bool,
    timeout_seconds: int,
    min_free_disk_gb: float,
) -> Dict[str, object]:
    key = str(method["key"])
    repo = resolve_project_path(method["local_path"])
    record: Dict[str, object] = {
        "method": key,
        "repo": str(repo),
        "status": "catalog",
        "extensions": [],
    }
    if not repo.is_dir():
        record["status"] = "failed"
        record["error"] = f"repository is missing: {repo}"
        return record

    extension_prefixes = profile.get("extension_prefixes", [])
    if not isinstance(extension_prefixes, list):
        record["status"] = "failed"
        record["error"] = "profile extension_prefixes must be a list"
        return record
    sources, modules, extension_spec_errors = catalog_extension_spec(repo, profile)
    record["extensions"] = sources
    record["modules"] = modules
    record["extension_prefixes"] = extension_prefixes
    if extension_spec_errors:
        record["status"] = "failed"
        record["errors"] = extension_spec_errors
        record["error"] = "; ".join(extension_spec_errors)
        return record
    if not sources:
        record["status"] = "no_extensions_detected"
        print(f"No CUDA/C++ extensions detected for {key}.", flush=True)
        return record
    if not run_real:
        return record

    build_root = PROJECT_ROOT / "third_party_build" / key
    target = build_root / "site-packages"
    manifest = target / ".unified3dgs_extension_build.json"
    env = build_method_env(key, repo)
    signature = build_signature(repo, sources, modules, env)
    record["signature"] = signature
    if manifest.is_file():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if existing.get("signature") == signature:
            try:
                alias_records = install_module_aliases(
                    profile.get("module_aliases", []), target, env
                )
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"isolated module alias preparation failed: {exc}"
                return record
            record["status"] = "already_built"
            record["target"] = str(target)
            record["manifest"] = str(manifest)
            record["module_aliases"] = alias_records
            print(f"Reusing verified isolated extension build for {key}: {target}", flush=True)
            return record

    low_disk = disk_error(min_free_disk_gb)
    if low_disk:
        print(f"[{key}] ERROR: {low_disk}", flush=True)
        record["status"] = "failed"
        record["error"] = low_disk
        return record

    source_root = build_root / f"src_{time.strftime('%Y%m%d_%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    logs: List[str] = []
    build_records: List[Dict[str, object]] = []
    if manifest.exists():
        manifest.unlink()

    for index, relative in enumerate(sources, 1):
        print()
        print(
            f"[{key}] extension {index}/{len(sources)}: {relative}",
            flush=True,
        )
        low_disk = disk_error(min_free_disk_gb)
        if low_disk:
            print(f"[{key}] ERROR before {relative}: {low_disk}", flush=True)
            build_records.append(
                {
                    "source": relative,
                    "exit_code": None,
                    "log": None,
                    "status": "failed",
                    "error": low_disk,
                }
            )
            continue
        is_archive = relative.startswith("archive:")
        is_external = relative.startswith("external:")
        if is_archive:
            source = repo / relative[len("archive:") :]
        elif is_external:
            source = resolve_project_path(relative[len("external:") :])
        else:
            source = repo / relative
        source_label = (
            "repo_root"
            if relative in {"", "."}
            else relative.replace("/", "__").replace(":", "__")
        )
        copy_name = f"{index:02d}_{source_label}"
        copied = source_root / copy_name
        print(f"Copying source to: {copied}", flush=True)
        if is_archive:
            copied.mkdir(parents=True, exist_ok=False)
            install_source = extract_archive_safely(source, copied)
        else:
            shutil.copytree(
                source,
                copied,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "build",
                    "dist",
                    "*.egg-info",
                    "__pycache__",
                    "outputs",
                ),
            )
            install_source = copied
        clean_copied_source(install_source)
        log_path = build_root / f"install_{index:02d}_{copy_name}.log"
        command = [
            "timeout",
            "--signal=INT",
            "--kill-after=30s",
            str(timeout_seconds),
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--upgrade",
            "--target",
            str(target),
            str(install_source),
        ]
        print("Command:", " ".join(command), flush=True)
        print(f"Live log: {log_path}", flush=True)
        returncode = run_streaming_build(
            command,
            env,
            log_path,
            method=key,
            source=relative,
        )
        logs.append(str(log_path))
        build_record: Dict[str, object] = {
            "source": relative,
            "exit_code": returncode,
            "log": str(log_path),
            "status": "built" if returncode == 0 else "failed",
        }
        if returncode != 0:
            build_record["error"] = (
                f"extension build failed for {relative}, exit={returncode}; "
                f"see {log_path}"
            )
            build_record["source_copy_retained"] = str(copied)
        else:
            print(f"[{key}] extension build succeeded: {relative}", flush=True)
            shutil.rmtree(copied)
            build_record["source_copy_retained"] = None
        build_records.append(build_record)

    record["build_records"] = build_records
    record["target"] = str(target)
    record["logs"] = logs
    failures = [item for item in build_records if item["status"] == "failed"]
    if failures:
        record["status"] = "failed"
        record["errors"] = [str(item["error"]) for item in failures]
        record["error"] = (
            f"{len(failures)} of {len(build_records)} detected extension builds failed; "
            "see build_records and logs"
        )
        return record
    if source_root.is_dir() and not any(source_root.iterdir()):
        source_root.rmdir()

    try:
        alias_records = install_module_aliases(profile.get("module_aliases", []), target, env)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"isolated module alias preparation failed: {exc}"
        return record
    record["module_aliases"] = alias_records

    manifest.write_text(
        json.dumps(
            {
                "method": key,
                "python": sys.executable,
                "sources": sources,
                "modules": modules,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "signature": signature,
                "build_records": build_records,
                "module_aliases": alias_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    record["status"] = "built"
    record["manifest"] = str(manifest)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or build isolated CUDA/C++ extensions for methods."
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--min-free-disk-gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/extension_build_report.json"),
    )
    args = parser.parse_args()

    selected = select_methods(args.method)
    records: List[Dict[str, object]] = []
    print(f"Method isolated extension builder", flush=True)
    print(f"Methods selected: {len(selected)}", flush=True)
    print(f"Real build: {args.run_real}", flush=True)
    print(f"Minimum free disk: {args.min_free_disk_gb:g} GiB", flush=True)
    for index, (method, profile) in enumerate(selected, 1):
        print()
        print("=" * 72, flush=True)
        print(
            f"METHOD {index}/{len(selected)}: {method['title']} [{method['key']}]",
            flush=True,
        )
        print("=" * 72, flush=True)
        try:
            record = install_method(
                method,
                profile,
                args.run_real,
                args.timeout_seconds,
                args.min_free_disk_gb,
            )
        except Exception as exc:
            record = {
                "method": str(method["key"]),
                "status": "failed",
                "extensions": [],
                "error": f"unexpected extension preparation exception: {exc!r}",
            }
            print(f"[{method['key']}] ERROR: {record['error']}", flush=True)
        records.append(record)
        print(
            f"[{record['status']}] {record['method']}: {record.get('extensions', [])}",
            flush=True,
        )
        if record.get("error"):
            print(f"  ERROR: {record['error']}", flush=True)

    report = args.report.expanduser()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [record for record in records if record["status"] == "failed"]
    print(f"Report: {report}")
    if not args.run_real:
        print("Plan only. Add --run-real to build the detected extensions.")
    print(f"Failed: {len(failed)}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
