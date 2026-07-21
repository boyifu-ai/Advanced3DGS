from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


PATCH_MARKER = "Unified 3DGS 3DHGS render GT patch"


def _find_gt_variable(text: str) -> str:
    match = re.search(r"(?m)^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=.*view\.original_image", text)
    if match:
        return match.group("name")
    return "gt"


def _patch_render_path(text: str) -> tuple[str, bool]:
    pattern = re.compile(
        r"(?m)^(?P<indent>\s*)render_path\s*=\s*"
        r"(?P<join>os\.path\.join\(.*,\s*)"
        r"(?P<quote>[\"'])renders(?P=quote)"
        r"(?P<suffix>\)\s*)$"
    )

    def replace(match: re.Match[str]) -> str:
        line = match.group(0)
        indent = match.group("indent")
        gt_line = (
            f"{indent}gts_path = "
            f"{match.group('join')}{match.group('quote')}gt{match.group('quote')}"
            f"{match.group('suffix')}"
        )
        return f"{line}\n{gt_line}"

    patched, count = pattern.subn(replace, text, count=1)
    return patched, count > 0


def _patch_makedirs(text: str) -> tuple[str, bool]:
    pattern = re.compile(r"(?m)^(?P<indent>\s*)(?P<call>(?:os\.)?makedirs)\(render_path,\s*exist_ok=True\)\s*$")

    def replace(match: re.Match[str]) -> str:
        line = match.group(0)
        indent = match.group("indent")
        call = match.group("call")
        return f"{line}\n{indent}{call}(gts_path, exist_ok=True)"

    patched, count = pattern.subn(replace, text, count=1)
    return patched, count > 0


def _patch_save_image(text: str, gt_variable: str) -> tuple[str, bool]:
    lines = text.splitlines()
    output = []
    patched = False

    for line in lines:
        output.append(line)
        if patched:
            continue
        if "save_image(" not in line or "render_path" not in line:
            continue

        indent = line[: len(line) - len(line.lstrip())]
        gt_line = line.replace("render_path", "gts_path", 1)
        gt_line = re.sub(r"save_image\(\s*[^,]+,\s*", f"save_image({gt_variable}, ", gt_line, count=1)
        output.append(f"{indent}# {PATCH_MARKER}: save GT images for unified metrics.")
        output.append(gt_line)
        patched = True

    return "\n".join(output) + ("\n" if text.endswith("\n") else ""), patched


def patch_file(path: Path) -> str:
    if not path.is_file():
        return "missing"

    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return "already_patched"

    backup_path = path.with_suffix(path.suffix + ".bak_unified3dgs_gt")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    gt_variable = _find_gt_variable(text)
    patched, render_path_ok = _patch_render_path(text)
    patched, makedirs_ok = _patch_makedirs(patched)
    patched, save_ok = _patch_save_image(patched, gt_variable)

    missing = []
    if not render_path_ok:
        missing.append("render_path")
    if not makedirs_ok:
        missing.append("makedirs(render_path)")
    if not save_ok:
        missing.append("save_image(render_path)")
    if missing:
        raise RuntimeError(
            f"Could not patch {path}; patterns not found: {', '.join(missing)}. "
            "Inspect third_party/3DHGS/render.py and update this patch script."
        )

    path.write_text(patched, encoding="utf-8")
    return f"patched gt_variable={gt_variable} backup={backup_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch 3DHGS render.py to save GT images.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    args = parser.parse_args()

    render_py = args.project_root / "third_party" / "3DHGS" / "render.py"
    print(f"3dhgs render GT patch: {patch_file(render_py)}: {render_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
