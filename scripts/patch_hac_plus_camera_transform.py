from __future__ import annotations

import argparse
import re
from pathlib import Path


PATCH_MARKER = "Unified 3DGS HAC+ camera transform compatibility patch"


REPLACEMENT = f"""        # {PATCH_MARKER}.
        try:
            self.full_proj_transform = (
                self.world_view_transform.contiguous().unsqueeze(0)
                .bmm(self.projection_matrix.contiguous().unsqueeze(0))
            ).squeeze(0)
        except RuntimeError as exc:
            if "CUBLAS_STATUS_NOT_SUPPORTED" not in str(exc):
                raise
            self.full_proj_transform = (
                self.world_view_transform.detach().cpu().matmul(
                    self.projection_matrix.detach().cpu()
                )
            ).to(
                device=self.world_view_transform.device,
                dtype=self.world_view_transform.dtype,
            )"""


TARGET_PATTERN = re.compile(
    r"(?P<indent>[ \t]*)self\.full_proj_transform\s*=\s*"
    r"\(self\.world_view_transform\.unsqueeze\(0\)\.bmm"
    r"\(self\.projection_matrix\.unsqueeze\(0\)\)\)\.squeeze\(0\)"
)


def patch_text(text: str) -> tuple[str, str]:
    if PATCH_MARKER in text:
        return text, "already_patched"
    patched, count = TARGET_PATTERN.subn(REPLACEMENT, text, count=1)
    if count == 0:
        raise ValueError(
            "Could not find the HAC+ full_proj_transform bmm pattern. "
            "Inspect third_party/HAC-plus/scene/cameras.py before retrying."
        )
    return patched, "patched"


def patch_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    patched, status = patch_text(text)
    if patched != text:
        backup = path.with_suffix(path.suffix + ".bak_unified3dgs_camera_transform")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        path.write_text(patched, encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Patch HAC+ camera transform creation to avoid a PyTorch 1.12/cu116 "
            "CUBLAS_STATUS_NOT_SUPPORTED failure on 4x4 bmm."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    target = args.project_root / "third_party" / "HAC-plus" / "scene" / "cameras.py"
    if not target.is_file():
        raise SystemExit(f"Missing HAC+ camera file: {target}")
    status = patch_file(target)
    print(f"HAC+ camera transform patch: {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
