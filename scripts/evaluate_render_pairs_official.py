from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.metrics_io import discover_render_pair


def load_image(path: Path, device, batched: bool, torch):
    from PIL import Image
    import torchvision.transforms.functional as tf

    with Image.open(path) as image:
        tensor = tf.to_tensor(image)[:3, :, :].contiguous().to(device)
    return tensor.unsqueeze(0) if batched else tensor


def import_from_repo(repo: Path, module: str):
    repo_text = str(repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    return importlib.import_module(module)


def metric_functions(repo: Path, style: str):
    image_utils = import_from_repo(repo, "utils.image_utils")
    lpips_module = import_from_repo(repo, "lpipsPyTorch")
    if style == "beta_splatting":
        fused = importlib.import_module("fused_ssim")
        return {
            "batched": False,
            "psnr": image_utils.psnr,
            "ssim": lambda render, gt: fused.fused_ssim(
                render.unsqueeze(0), gt.unsqueeze(0)
            ).mean(),
            "lpips": lambda render, gt: lpips_module.lpips(
                render, gt, net_type="vgg"
            ).mean(),
        }
    if style == "standard_3dgs":
        loss_utils = import_from_repo(repo, "utils.loss_utils")
        return {
            "batched": True,
            "psnr": image_utils.psnr,
            "ssim": lambda render, gt: loss_utils.ssim(render, gt).mean(),
            "lpips": lambda render, gt: lpips_module.lpips(
                render, gt, net_type="vgg"
            ).mean(),
        }
    raise ValueError(f"Unsupported official metrics style: {style}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSNR/SSIM/LPIPS with official repository metric functions, "
            "then write Unified 3DGS-compatible results.json/per_view.json."
        )
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument(
        "--style",
        required=True,
        choices=("standard_3dgs", "beta_splatting"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--results-output", default=None, type=Path)
    args = parser.parse_args()

    import torch

    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    results_output = (
        args.results_output.expanduser().resolve()
        if args.results_output
        else output / "results.json"
    )
    functions = metric_functions(repo, args.style)
    pair = discover_render_pair(output, args.iteration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batched = bool(functions["batched"])

    print(f"Official metrics style: {args.style}")
    print(f"Method: {args.method}")
    print(f"Repository: {repo}")
    print(f"Render directory: {pair.renders}")
    print(f"GT directory: {pair.gt}")
    print(f"Paired images: {len(pair.pairs)}")

    per_view: Dict[str, Dict[str, float]] = {}
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    lpips_values: List[float] = []
    with torch.no_grad():
        for index, (render_path, gt_path) in enumerate(pair.pairs, 1):
            render = load_image(render_path, device, batched, torch)
            gt = load_image(gt_path, device, batched, torch)
            if render.shape != gt.shape:
                raise ValueError(
                    f"Image shape mismatch for {render_path.name}: "
                    f"render={tuple(render.shape)} gt={tuple(gt.shape)}"
                )
            psnr_value = float(functions["psnr"](render, gt).mean().item())
            ssim_value = float(functions["ssim"](render, gt).mean().item())
            lpips_value = float(functions["lpips"](render, gt).mean().item())
            if not all(
                math.isfinite(value)
                for value in (psnr_value, ssim_value, lpips_value)
            ):
                raise ValueError(f"Non-finite metric for image pair {render_path.name}")
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)
            per_view[render_path.name] = {
                "PSNR": psnr_value,
                "SSIM": ssim_value,
                "LPIPS": lpips_value,
            }
            print(
                f"[{index}/{len(pair.pairs)}] {render_path.name}: "
                f"PSNR={psnr_value:.6f} SSIM={ssim_value:.6f} "
                f"LPIPS={lpips_value:.6f}",
                flush=True,
            )

    metrics = {
        "PSNR": sum(psnr_values) / len(psnr_values),
        "SSIM": sum(ssim_values) / len(ssim_values),
        "LPIPS": sum(lpips_values) / len(lpips_values),
    }
    key = f"ours_{args.iteration}"
    results_output.parent.mkdir(parents=True, exist_ok=True)
    results_output.write_text(
        json.dumps({key: metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_output.with_name("per_view.json").write_text(
        json.dumps({key: per_view}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_output.with_name("metric_policy.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "metric_source": "official_repository_functions",
                "official_metrics_style": args.style,
                "repo": str(repo),
                "results": str(results_output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "Official-style metrics: "
        f"PSNR={metrics['PSNR']:.6f} SSIM={metrics['SSIM']:.6f} "
        f"LPIPS={metrics['LPIPS']:.6f}"
    )
    print(f"Results: {results_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
