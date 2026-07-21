from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.metrics_io import discover_render_pair


def gaussian(window_size: int, sigma: float, torch):
    values = torch.tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
            for x in range(window_size)
        ],
        dtype=torch.float32,
    )
    return values / values.sum()


def create_window(window_size: int, channel: int, device, torch):
    one_d = gaussian(window_size, 1.5, torch).unsqueeze(1)
    two_d = one_d.mm(one_d.t()).float().unsqueeze(0).unsqueeze(0)
    return two_d.expand(channel, 1, window_size, window_size).contiguous().to(device)


def ssim(image, target, torch) -> float:
    import torch.nn.functional as functional

    channel = image.shape[0]
    window_size = 11
    window = create_window(window_size, channel, image.device, torch)
    image = image.unsqueeze(0)
    target = target.unsqueeze(0)
    mu1 = functional.conv2d(image, window, padding=window_size // 2, groups=channel)
    mu2 = functional.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2
    sigma1_sq = (
        functional.conv2d(image * image, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        functional.conv2d(target * target, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        functional.conv2d(image * target, window, padding=window_size // 2, groups=channel)
        - mu12
    )
    c1 = 0.01**2
    c2 = 0.03**2
    value = (
        (2 * mu12 + c1)
        * (2 * sigma12 + c2)
        / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    )
    return float(value.mean().item())


def load_rgb(path: Path, device, torch):
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute unified PSNR/SSIM/LPIPS from paired render and GT images."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--results-output", type=Path, default=None)
    args = parser.parse_args()

    import torch
    import lpips

    output = args.output.expanduser().resolve()
    results_output = (
        args.results_output.expanduser().resolve()
        if args.results_output
        else output / "results.json"
    )
    pair = discover_render_pair(output, args.iteration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Render directory: {pair.renders}")
    print(f"GT directory: {pair.gt}")
    print(f"Paired images: {len(pair.pairs)}")
    print(
        "Loading LPIPS VGG model. The first run may download pretrained weights "
        "into the project-owned TORCH_HOME cache.",
        flush=True,
    )
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    per_view: Dict[str, Dict[str, float]] = {}
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    lpips_values: List[float] = []
    with torch.no_grad():
        for index, (render_path, gt_path) in enumerate(pair.pairs, 1):
            render = load_rgb(render_path, device, torch)
            gt = load_rgb(gt_path, device, torch)
            if render.shape != gt.shape:
                raise ValueError(
                    f"Image shape mismatch for {render_path.name}: "
                    f"render={tuple(render.shape)} gt={tuple(gt.shape)}"
                )
            mse = torch.mean((render - gt) ** 2)
            psnr_value = float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item())
            ssim_value = ssim(render, gt, torch)
            lpips_value = float(
                lpips_model(
                    render.unsqueeze(0) * 2.0 - 1.0,
                    gt.unsqueeze(0) * 2.0 - 1.0,
                )
                .mean()
                .item()
            )
            if not all(math.isfinite(value) for value in (psnr_value, ssim_value, lpips_value)):
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
    per_view_path = results_output.with_name("per_view.json")
    per_view_path.write_text(
        json.dumps({key: per_view}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Unified metrics: "
        f"PSNR={metrics['PSNR']:.6f} SSIM={metrics['SSIM']:.6f} "
        f"LPIPS={metrics['LPIPS']:.6f}"
    )
    print(f"Results: {results_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
