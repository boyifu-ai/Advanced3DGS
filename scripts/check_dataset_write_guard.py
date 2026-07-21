from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Python dataset write guard.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    guard_path = project_root / "unified3dgs" / "runtime_guard"
    if not (guard_path / "sitecustomize.py").is_file():
        raise FileNotFoundError(f"Missing runtime guard: {guard_path}")

    with tempfile.TemporaryDirectory(prefix="unified3dgs_guard_") as temp_value:
        temp_root = Path(temp_value)
        readonly_root = temp_root / "shared"
        output_root = temp_root / "outputs"
        readonly_root.mkdir()
        output_root.mkdir()

        env = os.environ.copy()
        env["UNIFIED3DGS_READONLY_DATASET_ROOT"] = str(readonly_root)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(guard_path), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        blocked_target = readonly_root / "must_not_exist.txt"
        blocked_dir = readonly_root / "must_not_exist_dir"
        blocked_existing = readonly_root / "must_not_change.txt"
        linked_readonly = output_root / "linked_shared"
        allowed_target = output_root / "must_exist.txt"
        allowed_dir = output_root / "must_exist_dir"
        blocked_existing.write_text("unchanged", encoding="utf-8")
        symlink_supported = True
        try:
            linked_readonly.symlink_to(readonly_root, target_is_directory=True)
        except OSError as exc:
            symlink_supported = False
            print(f"Symlink guard check skipped on this platform: {exc}")
        blocked_via_symlink = linked_readonly / "must_not_exist_via_symlink.txt"
        code = (
            "from pathlib import Path\n"
            f"blocked = Path({str(blocked_target)!r})\n"
            f"blocked_dir = Path({str(blocked_dir)!r})\n"
            f"blocked_existing = Path({str(blocked_existing)!r})\n"
            f"blocked_via_symlink = Path({str(blocked_via_symlink)!r})\n"
            f"allowed = Path({str(allowed_target)!r})\n"
            f"allowed_dir = Path({str(allowed_dir)!r})\n"
            "allowed.write_text('ok', encoding='utf-8')\n"
            "allowed_dir.mkdir(parents=True, exist_ok=True)\n"
            "try:\n"
            "    blocked.write_text('bad', encoding='utf-8')\n"
            "except PermissionError as exc:\n"
            "    print(exc)\n"
            "else:\n"
            "    raise SystemExit('dataset write was not blocked')\n"
            "try:\n"
            "    blocked_dir.mkdir(parents=True, exist_ok=True)\n"
            "except PermissionError as exc:\n"
            "    print(exc)\n"
            "else:\n"
            "    raise SystemExit('dataset mkdir was not blocked')\n"
            "try:\n"
            "    blocked_existing.unlink()\n"
            "except PermissionError as exc:\n"
            "    print(exc)\n"
            "else:\n"
            "    raise SystemExit('dataset unlink was not blocked')\n"
            "try:\n"
            "    blocked_existing.chmod(0o600)\n"
            "except PermissionError as exc:\n"
            "    print(exc)\n"
            "else:\n"
            "    raise SystemExit('dataset chmod was not blocked')\n"
        )
        if symlink_supported:
            code += (
                "try:\n"
                "    blocked_via_symlink.write_text('bad', encoding='utf-8')\n"
                "except PermissionError as exc:\n"
                "    print(exc)\n"
                "else:\n"
                "    raise SystemExit('dataset write through symlink was not blocked')\n"
            )
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

        if blocked_target.exists():
            raise RuntimeError(f"Guard test unexpectedly wrote: {blocked_target}")
        if blocked_dir.exists():
            raise RuntimeError(f"Guard test unexpectedly created: {blocked_dir}")
        if blocked_existing.read_text(encoding="utf-8") != "unchanged":
            raise RuntimeError(f"Guard test unexpectedly changed: {blocked_existing}")
        if symlink_supported and (readonly_root / blocked_via_symlink.name).exists():
            raise RuntimeError(
                "Guard test unexpectedly wrote through dataset symlink: "
                f"{blocked_via_symlink}"
            )
        if not allowed_target.is_file():
            raise RuntimeError(f"Guard test did not write output: {allowed_target}")
        if not allowed_dir.is_dir():
            raise RuntimeError(f"Guard test did not create output dir: {allowed_dir}")

    print("Dataset write guard passed: dataset writes blocked, output writes allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
