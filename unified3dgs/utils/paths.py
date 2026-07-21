from __future__ import annotations

from pathlib import Path

from unified3dgs.dataset_config import readonly_dataset_roots


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_output_is_safe(output_path: Path) -> None:
    for dataset_root in readonly_dataset_roots():
        if _is_relative_to(output_path, dataset_root):
            raise ValueError(
                "Refusing to write outputs under a configured dataset root: "
                f"{dataset_root}"
            )
