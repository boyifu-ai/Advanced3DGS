from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import (
    PROFILE_PATH,
    load_confirmed_catalog,
    load_json_list,
    resolve_project_path,
)


def detect_metric_files(repo: Path) -> Dict[str, object]:
    candidates = {
        "metrics.py": repo / "metrics.py",
        "full_eval.py": repo / "full_eval.py",
        "eval.py": repo / "eval.py",
    }
    files = {name: path.is_file() for name, path in candidates.items()}
    details: Dict[str, object] = {
        "files": {name: str(path) for name, path in candidates.items() if path.is_file()},
        "has_lpips_pytorch": (repo / "lpipsPyTorch").exists(),
        "has_image_utils": (repo / "utils" / "image_utils.py").is_file(),
        "has_loss_utils": (repo / "utils" / "loss_utils.py").is_file(),
    }
    if files["metrics.py"] and details["has_lpips_pytorch"]:
        details["detected_style"] = "standard_3dgs"
    elif files["eval.py"] and details["has_lpips_pytorch"]:
        details["detected_style"] = "custom_eval"
    else:
        details["detected_style"] = ""
    return details


def audit() -> Dict[str, object]:
    profiles = {
        str(row["key"]): dict(row)
        for row in load_json_list(PROFILE_PATH)
        if isinstance(row, dict) and row.get("key")
    }
    rows: List[Dict[str, object]] = []
    for method in load_confirmed_catalog():
        key = str(method["key"])
        repo = resolve_project_path(method["local_path"])
        profile = profiles.get(key, {})
        detected = detect_metric_files(repo)
        configured_style = str(profile.get("official_metrics_style") or "")
        warnings: List[str] = []
        if detected["detected_style"] == "standard_3dgs" and not configured_style:
            warnings.append(
                "repo appears to provide standard 3DGS metrics, but "
                "official_metrics_style is not configured"
            )
        if configured_style and detected["detected_style"] == "":
            warnings.append(
                "official_metrics_style is configured, but required official "
                "metric files were not found locally"
            )
        rows.append(
            {
                "method": key,
                "title": method.get("title", key),
                "repo": str(repo),
                "configured_official_metrics_style": configured_style,
                **detected,
                "warnings": warnings,
            }
        )
    return {
        "method_count": len(rows),
        "configured_count": sum(
            1 for row in rows if row["configured_official_metrics_style"]
        ),
        "detected_standard_or_custom_count": sum(
            1 for row in rows if row["detected_style"]
        ),
        "warning_count": sum(len(row["warnings"]) for row in rows),
        "methods": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit official metric/evaluation scripts in confirmed third-party "
            "method repositories."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/official_metric_scripts.json"),
    )
    args = parser.parse_args()
    report = args.report.expanduser()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()
    payload = audit()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Official metric script audit")
    print(
        f"Configured official metrics: {payload['configured_count']}/"
        f"{payload['method_count']}"
    )
    print(
        "Detected standard/custom metric scripts: "
        f"{payload['detected_standard_or_custom_count']}/{payload['method_count']}"
    )
    for row in payload["methods"]:
        style = row["configured_official_metrics_style"] or "fallback_unified"
        detected = row["detected_style"] or "none"
        print(f"{row['method']}: configured={style} detected={detected}")
        for warning in row["warnings"]:
            print(f"  - {warning}")
    print(f"Report: {report}")
    return 0 if payload["warning_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
