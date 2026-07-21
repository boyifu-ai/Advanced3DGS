from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "configs" / "method_catalog.json"


def load_catalog() -> List[Dict[str, str]]:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Method catalog must be a list: {CATALOG_PATH}")
    return data


def confirmed(method: Dict[str, str]) -> bool:
    return method.get("source_status") == "confirmed" and bool(method.get("repository"))


def print_catalog(methods: List[Dict[str, str]]) -> None:
    print(f"Methods: {len(methods)}")
    print("Status: confirmed means the upstream Git repository was verified as reachable.")
    print()
    for index, method in enumerate(methods, 1):
        status = method["source_status"]
        print(f"{index:>2}. {method['title']} [{method['key']}] ({method['venue']})")
        print(f"    status: {status}")
        print(f"    local:  {method['local_path']}")
        if method.get("repository"):
            print(f"    repo:   {method['repository']}")
        if method.get("project_page"):
            print(f"    page:   {method['project_page']}")


def print_clone_commands(methods: List[Dict[str, str]]) -> None:
    print(f"cd {PROJECT_ROOT}")
    print("mkdir -p third_party")
    for method in methods:
        if not confirmed(method):
            continue
        print(
            f'git clone --recursive "{method["repository"]}" '
            f'"{method["local_path"]}"'
        )


def clone_methods(methods: List[Dict[str, str]]) -> int:
    for method in methods:
        if not confirmed(method):
            print(
                f"SKIP unconfirmed source: {method['title']} "
                f"({method['source_status']})"
            )
            continue
        destination = PROJECT_ROOT / method["local_path"]
        if destination.exists():
            print(f"SKIP existing: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "git",
            "clone",
            "--recursive",
            method["repository"],
            str(destination),
        ]
        print(f"CLONE: {method['title']} -> {destination}")
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            print(f"FAILED: {method['title']} exit={result.returncode}")
            return result.returncode
    return 0


def select_methods(
    catalog: List[Dict[str, str]],
    keys: List[str],
    select_all: bool,
) -> List[Dict[str, str]]:
    if select_all:
        return catalog
    wanted = set(keys)
    selected = [method for method in catalog if method["key"] in wanted]
    missing = sorted(wanted - {method["key"] for method in selected})
    if missing:
        raise ValueError(f"Unknown method key(s): {', '.join(missing)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or clone the catalog Advanced3DGS method repositories."
    )
    parser.add_argument(
        "action",
        choices=("list", "commands", "clone"),
        help="List catalog entries, print server clone commands, or clone repositories.",
    )
    parser.add_argument("--method", action="append", default=[], help="Method key.")
    parser.add_argument("--all", action="store_true", help="Select all methods.")
    args = parser.parse_args()

    catalog = load_catalog()
    selected = select_methods(catalog, args.method, args.all or not args.method)

    if args.action == "list":
        print_catalog(selected)
        return 0
    if args.action == "commands":
        print_clone_commands(selected)
        return 0
    if args.action == "clone":
        return clone_methods(selected)
    raise ValueError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
