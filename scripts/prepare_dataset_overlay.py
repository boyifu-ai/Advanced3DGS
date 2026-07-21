from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT
from unified3dgs.dataset_overlay import prepare_dataset_overlay


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a project-local, low-resolution COLMAP compatibility view for "
            "method iteration=1 completeness testing."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validation/_method_acceptance/dataset_overlay_report.json"),
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser()
    if not dataset.is_absolute():
        dataset = (PROJECT_ROOT / dataset).resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    report = args.report.expanduser()
    if not report.is_absolute():
        report = (PROJECT_ROOT / report).resolve()

    overlay = prepare_dataset_overlay(dataset, output_root)
    manifest = json.loads(
        (overlay / "overlay_manifest.json").read_text(encoding="utf-8")
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Method dataset overlay compatibility view")
    print(f"Original read-only dataset: {dataset}")
    print(f"Project-local dataset overlay: {overlay}")
    print(f"Low-resolution image source: {manifest['preferred_images']}")
    print(f"Added PLY fields: {manifest['ply']['added_fields']}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
