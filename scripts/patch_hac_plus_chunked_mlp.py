from __future__ import annotations

import argparse
import re
from pathlib import Path


PATCH_MARKER = "Unified 3DGS HAC+ chunked MLP compatibility patch"


HELPER = f"""
# {PATCH_MARKER}.
def _unified3dgs_chunked_mlp_forward(module, inputs, chunk_size=32768):
    if inputs.shape[0] <= chunk_size:
        return module(inputs)
    return torch.cat(
        [module(chunk.contiguous()) for chunk in inputs.split(chunk_size, dim=0)],
        dim=0,
    )
"""


REPLACEMENTS = {
    r"bank_weight = pc\.get_featurebank_mlp\(cat_view\)": (
        "bank_weight = _unified3dgs_chunked_mlp_forward(pc.get_featurebank_mlp, cat_view)"
    ),
    r"neural_opacity = pc\.get_opacity_mlp\(cat_local_view\)": (
        "neural_opacity = _unified3dgs_chunked_mlp_forward(pc.get_opacity_mlp, cat_local_view)"
    ),
    r"color = pc\.get_color_mlp\(cat_local_view\)": (
        "color = _unified3dgs_chunked_mlp_forward(pc.get_color_mlp, cat_local_view)"
    ),
    r"scale_rot = pc\.get_cov_mlp\(cat_local_view\)": (
        "scale_rot = _unified3dgs_chunked_mlp_forward(pc.get_cov_mlp, cat_local_view)"
    ),
}


def patch_text(text: str) -> tuple[str, str]:
    if PATCH_MARKER in text:
        return text, "already_patched"
    if "def generate_neural_gaussians(" not in text:
        raise ValueError("Could not find generate_neural_gaussians in HAC+ renderer")

    insertion_point = text.find("def generate_neural_gaussians(")
    patched = text[:insertion_point] + HELPER + "\n" + text[insertion_point:]
    missing = []
    for pattern, replacement in REPLACEMENTS.items():
        patched, count = re.subn(pattern, replacement, patched)
        if count == 0:
            missing.append(pattern)
    if missing:
        raise ValueError(
            "Could not patch HAC+ renderer MLP call(s): " + ", ".join(missing)
        )
    return patched, "patched"


def patch_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    patched, status = patch_text(text)
    if patched != text:
        backup = path.with_suffix(path.suffix + ".bak_unified3dgs_chunked_mlp")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        path.write_text(patched, encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Patch HAC+ renderer MLP calls to split large CUDA linear batches into "
            "gradient-preserving chunks under the official PyTorch 1.12/CUDA 11.6 backend."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    target = (
        args.project_root
        / "third_party"
        / "HAC-plus"
        / "gaussian_renderer"
        / "__init__.py"
    )
    if not target.is_file():
        raise SystemExit(f"Missing HAC+ renderer file: {target}")
    status = patch_file(target)
    print(f"HAC+ chunked MLP patch: {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
