from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = Path(__file__).resolve()
FIXED_METHOD_REPOS = {
    "third_party/gaussian-splatting",
    "third_party/2d-gaussian-splatting",
    "third_party/3dgs-mcmc",
    "third_party/3DHGS",
    "third_party/3D-student-splatting-and-scooping",
}
CORE_PATHS = (
    "README.md",
    "environment.yml",
    "requirements.txt",
    "train_all.py",
    "render_all.py",
    "eval_all.py",
    "unified3dgs_menu.py",
    "configs",
    "scripts",
    "tests",
    "unified3dgs",
    "patches/README.md",
    "third_party/README.md",
    "third_party/.gitkeep",
    "third_party_build/README.md",
    "third_party_build/.gitkeep",
    "outputs/.gitkeep",
)
BANNED_TERMS = (
    "planned",
    "smoke",
    "urgent",
    "additional_training",
    "planned_methods",
    "planned_method",
    "run_additional",
    "additional_method",
    "test_planned",
)
PERSONAL_PATH_PATTERNS = (
    re.compile(r"/home/[^/\s\"']+/"),
    re.compile(r"/Users/[^/\s\"']+/"),
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+[\\/]"),
)
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
}
TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}|\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def run_git(args: List[str]) -> List[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def submodule_paths() -> set[str]:
    gitmodules = PROJECT_ROOT / ".gitmodules"
    if not gitmodules.is_file():
        return set()
    lines = run_git(
        ["config", "--file", str(gitmodules), "--get-regexp", r"^submodule\..*\.path$"]
    )
    paths: set[str] = set()
    for line in lines:
        _, separator, path = line.partition(" ")
        if separator and path.strip():
            paths.add(Path(path.strip()).as_posix())
    return paths


def expected_submodule_paths() -> set[str]:
    catalog_path = PROJECT_ROOT / "configs" / "method_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    confirmed = {
        Path(item["local_path"]).as_posix()
        for item in catalog
        if item.get("source_status") == "confirmed" and item.get("repository")
    }
    return FIXED_METHOD_REPOS | confirmed


def iter_files(submodules: set[str]) -> Iterable[Path]:
    for path in PROJECT_ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        relative = rel(path)
        if any(relative == item or relative.startswith(f"{item}/") for item in submodules):
            continue
        if path.is_file():
            yield path


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def text_hits(path: Path, patterns: Iterable[str]) -> List[Dict[str, object]]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    lowered_patterns = [pattern.lower() for pattern in patterns]
    hits: List[Dict[str, object]] = []
    for index, line in enumerate(lines, 1):
        lowered = line.lower()
        if any(pattern in lowered for pattern in lowered_patterns):
            hits.append({"file": rel(path), "line": index, "text": line[:240]})
    return hits


def regex_hits(
    path: Path, patterns: Iterable[re.Pattern[str]]
) -> List[Dict[str, object]]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    hits: List[Dict[str, object]] = []
    for index, line in enumerate(lines, 1):
        if any(pattern.search(line) for pattern in patterns):
            hits.append({"file": rel(path), "line": index, "text": line[:240]})
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the release workspace before syncing or publishing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/_release_audit/workspace_audit.json"),
    )
    args = parser.parse_args()

    output = args.output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    tracked = set(run_git(["ls-files"]))
    submodules = submodule_paths()
    expected_submodules = expected_submodule_paths()
    missing_submodules = sorted(expected_submodules - submodules)
    unexpected_submodules = sorted(submodules - expected_submodules)
    status = run_git(["status", "--short"])
    files = sorted(iter_files(submodules), key=lambda item: rel(item))
    all_files = [
        {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "tracked": rel(path) in tracked,
        }
        for path in files
    ]
    generated_candidates = [
        item
        for item in all_files
        if (
            item["path"].startswith("outputs/")
            and item["path"] != "outputs/.gitkeep"
        )
        or "__pycache__/" in item["path"]
        or item["path"].endswith((".log", ".tmp", ".pyc"))
        or TIMESTAMP_RE.search(Path(item["path"]).name)
    ]
    banned_hits: List[Dict[str, object]] = []
    server_path_hits: List[Dict[str, object]] = []
    tracked_files = [PROJECT_ROOT / path for path in sorted(tracked)]
    for path in tracked_files:
        if not path.is_file():
            continue
        if path.resolve() == AUDIT_SCRIPT:
            continue
        if "__pycache__" in path.parts:
            continue
        banned_hits.extend(text_hits(path, BANNED_TERMS))
        server_path_hits.extend(regex_hits(path, PERSONAL_PATH_PATTERNS))

    core = {
        path: (PROJECT_ROOT / path).exists()
        for path in CORE_PATHS
    }
    tracked_violations = [
        path
        for path in tracked
        if (
            path.startswith("outputs/")
            and path != "outputs/.gitkeep"
        )
        or (
            path.startswith("third_party/")
            and path not in {
                "third_party/README.md",
                "third_party/.gitkeep",
                *submodules,
            }
        )
        or (
            path.startswith("third_party_build/")
            and path not in {"third_party_build/README.md", "third_party_build/.gitkeep"}
        )
        or path.endswith((".log", ".tmp", ".pyc"))
        or path.startswith("docs/")
        or TIMESTAMP_RE.search(Path(path).name)
    ]
    payload = {
        "project_root": str(PROJECT_ROOT),
        "core_paths": core,
        "git_status": status,
        "tracked_file_count": len(tracked),
        "submodule_count": len(submodules),
        "submodule_paths": sorted(submodules),
        "expected_submodule_count": len(expected_submodules),
        "missing_submodules": missing_submodules,
        "unexpected_submodules": unexpected_submodules,
        "all_file_count": len(all_files),
        "all_files": all_files,
        "generated_candidates": generated_candidates,
        "tracked_release_violations": tracked_violations,
        "banned_term_hits": banned_hits,
        "server_path_hits": server_path_hits,
        "passed": (
            all(core.values())
            and not tracked_violations
            and not banned_hits
            and not server_path_hits
            and not missing_submodules
            and not unexpected_submodules
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote release workspace audit: {output}")
    print(f"All files: {len(all_files)}")
    print(f"Tracked files: {len(tracked)}")
    print(f"Submodules: {len(submodules)}/{len(expected_submodules)}")
    print(f"Missing submodules: {len(missing_submodules)}")
    print(f"Unexpected submodules: {len(unexpected_submodules)}")
    print(f"Generated candidates: {len(generated_candidates)}")
    print(f"Tracked release violations: {len(tracked_violations)}")
    print(f"Banned term hits: {len(banned_hits)}")
    print(f"Server path hits: {len(server_path_hits)}")
    print(f"Passed: {payload['passed']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
