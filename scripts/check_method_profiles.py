from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    CATALOG_PATH,
    PROFILE_PATH,
    PROTECTED_RUN_FLAGS,
    load_json_list,
    method_specific_extra_arg_errors,
)


DEPENDENCIES_PATH = PROJECT_ROOT / "configs" / "method_python_dependencies.json"


def duplicate_keys(items: Sequence[Dict[str, object]]) -> List[str]:
    seen = set()
    duplicates = set()
    for item in items:
        key = str(item.get("key", ""))
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def audit_configuration() -> Dict[str, object]:
    catalog = [
        item
        for item in load_json_list(CATALOG_PATH)
        if item.get("source_status") == "confirmed" and item.get("repository")
    ]
    profiles = load_json_list(PROFILE_PATH)
    dependencies = load_json_list(DEPENDENCIES_PATH)
    errors: List[str] = []
    warnings: List[str] = []

    for name, items in (
        ("catalog", catalog),
        ("acceptance profiles", profiles),
        ("Python dependency profiles", dependencies),
    ):
        duplicates = duplicate_keys(items)
        if duplicates:
            errors.append(f"{name} contains duplicate keys: {', '.join(duplicates)}")

    catalog_keys = {str(item["key"]) for item in catalog}
    profile_keys = {str(item["key"]) for item in profiles}
    dependency_keys = {str(item["key"]) for item in dependencies}
    if profile_keys != catalog_keys:
        errors.append(
            "acceptance profile keys do not exactly match confirmed catalog keys: "
            f"missing={sorted(catalog_keys - profile_keys)}, "
            f"extra={sorted(profile_keys - catalog_keys)}"
        )
    if dependency_keys != catalog_keys:
        errors.append(
            "Python dependency keys do not exactly match confirmed catalog keys: "
            f"missing={sorted(catalog_keys - dependency_keys)}, "
            f"extra={sorted(dependency_keys - catalog_keys)}"
        )

    for profile in profiles:
        key = str(profile.get("key", "<missing>"))
        candidates = profile.get("entry_candidates", [])
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"{key}: entry_candidates must be a non-empty list")
        if profile.get("static_help_only") is True and not str(
            profile.get("static_help_reason", "")
        ).strip():
            errors.append(f"{key}: static_help_only requires static_help_reason")
        if "stop_after_verified_save" in profile and not isinstance(
            profile["stop_after_verified_save"], bool
        ):
            errors.append(f"{key}: stop_after_verified_save must be a boolean")
        if profile.get("stop_after_verified_save") is True and not str(
            profile.get("stop_after_verified_save_reason", "")
        ).strip():
            errors.append(
                f"{key}: stop_after_verified_save requires "
                "stop_after_verified_save_reason"
            )
        acceptance_resolution = profile.get("acceptance_resolution")
        if acceptance_resolution is not None and (
            not isinstance(acceptance_resolution, int)
            or isinstance(acceptance_resolution, bool)
            or acceptance_resolution == 0
        ):
            errors.append(
                f"{key}: acceptance_resolution must be a non-zero integer"
            )
        official_backend = profile.get("official_backend")
        if official_backend is not None:
            if not isinstance(official_backend, dict):
                errors.append(f"{key}: official_backend must be an object")
            else:
                required_backend_fields = (
                    "python_env_var",
                    "default_python",
                    "cuda_home",
                    "requirements",
                    "source",
                )
                missing = [
                    field
                    for field in required_backend_fields
                    if not official_backend.get(field)
                ]
                if missing:
                    errors.append(
                        f"{key}: official_backend is missing: "
                        + ", ".join(missing)
                    )
                requirements = official_backend.get("requirements")
                if not isinstance(requirements, dict):
                    errors.append(
                        f"{key}: official_backend.requirements must be an object"
                    )
                elif any(
                    not requirements.get(field)
                    for field in ("python", "torch", "torch_cuda")
                ):
                    errors.append(
                        f"{key}: official_backend.requirements must declare "
                        "python, torch, and torch_cuda"
                    )
        acceptance_training_entry = profile.get("acceptance_training_entry")
        if acceptance_training_entry:
            wrapper = (
                PROJECT_ROOT / str(acceptance_training_entry)
            ).resolve()
            try:
                wrapper.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                errors.append(
                    f"{key}: acceptance_training_entry must stay inside the project"
                )
            if not wrapper.is_file():
                errors.append(
                    f"{key}: acceptance_training_entry does not exist: {wrapper}"
                )
            max_iterations = profile.get("acceptance_max_iterations")
            if (
                not isinstance(max_iterations, int)
                or isinstance(max_iterations, bool)
                or max_iterations <= 0
            ):
                errors.append(
                    f"{key}: acceptance_training_entry requires a positive "
                    "acceptance_max_iterations"
                )
            attempts = profile.get("acceptance_training_attempts", 1)
            if (
                not isinstance(attempts, int)
                or isinstance(attempts, bool)
                or attempts <= 0
            ):
                errors.append(
                    f"{key}: acceptance_training_attempts must be a "
                    "positive integer"
                )
        official_dataset_args = profile.get("official_dataset_args", {})
        if not isinstance(official_dataset_args, dict):
            errors.append(f"{key}: official_dataset_args must be an object")
        else:
            for dataset_family, values in official_dataset_args.items():
                if not str(dataset_family).strip():
                    errors.append(
                        f"{key}: official_dataset_args contains an empty dataset key"
                    )
                if not isinstance(values, list):
                    errors.append(
                        f"{key}: official_dataset_args[{dataset_family!r}] "
                        "must be a list"
                    )
        official_scene_args = profile.get("official_scene_args", {})
        if not isinstance(official_scene_args, dict):
            errors.append(f"{key}: official_scene_args must be an object")
        else:
            for scene_label, values in official_scene_args.items():
                if "/" not in str(scene_label).strip("/"):
                    errors.append(
                        f"{key}: official_scene_args key must look like "
                        f"dataset/scene: {scene_label!r}"
                    )
                if not isinstance(values, list):
                    errors.append(
                        f"{key}: official_scene_args[{scene_label!r}] "
                        "must be a list"
                    )
        acceptance_point_caps = profile.get(
            "acceptance_dataset_point_caps", {}
        )
        if not isinstance(acceptance_point_caps, dict):
            errors.append(
                f"{key}: acceptance_dataset_point_caps must be an object"
            )
        else:
            for dataset_family, value in acceptance_point_caps.items():
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                ):
                    errors.append(
                        f"{key}: acceptance_dataset_point_caps"
                        f"[{dataset_family!r}] must be a positive integer"
                    )
        extra_args = profile.get("extra_args", [])
        if not isinstance(extra_args, list):
            errors.append(f"{key}: extra_args must be a list")
        else:
            conflicts = sorted(
                {
                    str(value).split("=", 1)[0]
                    for value in extra_args
                    if str(value).split("=", 1)[0] in PROTECTED_RUN_FLAGS
                }
            )
            if conflicts:
                errors.append(
                    f"{key}: extra_args override protected acceptance flags: {conflicts}"
                )
            errors.extend(method_specific_extra_arg_errors(key, extra_args))
        training_extra_args = profile.get("training_extra_args", extra_args)
        if not isinstance(training_extra_args, list):
            errors.append(f"{key}: training_extra_args must be a list")
        else:
            protected_training_flags = set(PROTECTED_RUN_FLAGS)
            conflicts = sorted(
                {
                    str(value).split("=", 1)[0]
                    for value in training_extra_args
                    if str(value).split("=", 1)[0] in protected_training_flags
                }
            )
            if conflicts:
                errors.append(
                    f"{key}: training_extra_args override framework-owned training "
                    f"flags: {conflicts}"
                )
            errors.extend(method_specific_extra_arg_errors(key, training_extra_args))
        render_candidates = profile.get("render_entry_candidates", ["render.py"])
        if not isinstance(render_candidates, list) or not render_candidates:
            errors.append(f"{key}: render_entry_candidates must be a non-empty list")
        framework_render_entry = profile.get("framework_render_entry")
        if framework_render_entry:
            wrapper = (PROJECT_ROOT / str(framework_render_entry)).resolve()
            try:
                wrapper.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                errors.append(
                    f"{key}: framework_render_entry must stay inside the project"
                )
            if not wrapper.is_file():
                errors.append(
                    f"{key}: framework_render_entry does not exist: {wrapper}"
                )
        render_contracts = profile.get("framework_render_contracts", [])
        if not isinstance(render_contracts, list):
            errors.append(f"{key}: framework_render_contracts must be a list")
        else:
            for contract in render_contracts:
                if not isinstance(contract, dict) or not contract.get("path"):
                    errors.append(
                        f"{key}: invalid framework_render_contracts entry: "
                        f"{contract!r}"
                    )
                    continue
                markers = contract.get("contains", [])
                if not isinstance(markers, list) or not markers:
                    errors.append(
                        f"{key}: framework render contract has no markers: "
                        f"{contract!r}"
                    )
        render_extra_args = profile.get("render_extra_args", [])
        if not isinstance(render_extra_args, list):
            errors.append(f"{key}: render_extra_args must be a list")
        render_flags = profile.get("render_flags", {})
        if not isinstance(render_flags, dict):
            errors.append(f"{key}: render_flags must be an object")
        result_globs = profile.get("result_globs", [])
        if not isinstance(result_globs, list):
            errors.append(f"{key}: result_globs must be a list")
        elif any("{iteration}" not in str(pattern) for pattern in result_globs):
            errors.append(f"{key}: every result_glob must contain {{iteration}}")
        output_globs = profile.get("output_globs", [])
        if not isinstance(output_globs, list):
            errors.append(f"{key}: output_globs must be a list")
        elif any(not str(pattern).startswith("{output}") for pattern in output_globs):
            errors.append(
                f"{key}: every output_glob must start with {{output}} to remain "
                "inside the method attempt directory"
            )
        for field, required_path_key in (
            ("archived_extensions", "archive"),
            ("external_extensions", "path"),
        ):
            entries = profile.get(field, [])
            if not isinstance(entries, list):
                errors.append(f"{key}: {field} must be a list")
                continue
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get(required_path_key):
                    errors.append(f"{key}: invalid {field} entry: {entry!r}")
                    continue
                modules = entry.get("modules", [])
                if not isinstance(modules, list) or not modules:
                    errors.append(
                        f"{key}: {field} entry has no declared modules: {entry!r}"
                    )
                if field == "external_extensions":
                    path = str(entry["path"]).replace("\\", "/").lstrip("./")
                    if not path.startswith("third_party/"):
                        errors.append(
                            f"{key}: external extension must stay under third_party/: {path}"
                        )

    vcs_pattern = re.compile(r"^(?:git\+)?https?://.+@([^#]+)")
    for dependency in dependencies:
        key = str(dependency.get("key", "<missing>"))
        packages = dependency.get("packages", [])
        blockers = dependency.get("manual_blockers", [])
        if not isinstance(packages, list) or not isinstance(blockers, list):
            errors.append(f"{key}: packages/manual_blockers must be lists")
            continue
        for package in packages:
            if not isinstance(package, dict):
                continue
            requirement = str(package.get("requirement", "")).strip()
            if not requirement:
                errors.append(f"{key}: dependency object has empty requirement")
                continue
            match = vcs_pattern.match(requirement)
            if not match:
                continue
            if package.get("require_isolated") is not True:
                errors.append(
                    f"{key}: VCS source dependency must set require_isolated=true: "
                    f"{requirement}"
                )
            reference = match.group(1)
            if reference in {"main", "master"}:
                if package.get("allow_moving_ref") is not True or not str(
                    package.get("moving_ref_reason", "")
                ).strip():
                    errors.append(
                        f"{key}: moving VCS ref {reference!r} requires "
                        "allow_moving_ref=true and moving_ref_reason"
                    )
                else:
                    warnings.append(
                        f"{key}: moving VCS ref {reference!r} is allowed; resolved commit "
                        "must be recorded by dependency preparation"
                    )

    return {
        "confirmed_method_count": len(catalog_keys),
        "confirmed_methods": sorted(catalog_keys),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate all method acceptance configuration before server work."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/configuration_audit.json"),
    )
    args = parser.parse_args()
    report = args.report.expanduser()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()
    payload = audit_configuration()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Method acceptance configuration audit")
    print(f"Confirmed methods: {payload['confirmed_method_count']}")
    print(f"Report: {report}")
    for warning in payload["warnings"]:
        print(f"WARNING: {warning}")
    for error in payload["errors"]:
        print(f"ERROR: {error}")
    print("Configuration audit passed." if payload["passed"] else "Configuration audit failed.")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
