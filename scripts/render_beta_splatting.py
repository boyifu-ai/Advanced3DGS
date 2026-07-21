from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import torch
    import torchvision

    from arguments import ModelParams
    from scene import BetaModel, Scene

    parser = argparse.ArgumentParser(description="Unified Beta-Splatting renderer")
    model_params = ModelParams(parser)
    parser.add_argument("--iteration", type=str, required=True)
    args = parser.parse_args()
    args.eval = True

    dataset = model_params.extract(args)
    beta_model = BetaModel(dataset.sh_degree, dataset.sb_number)
    background = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(
        background, dtype=torch.float32, device="cuda"
    )
    scene = Scene(dataset, beta_model, shuffle=False)

    iteration = str(args.iteration)
    point_cloud = (
        Path(dataset.model_path)
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )
    png_model = (
        Path(dataset.model_path)
        / "point_cloud"
        / f"iteration_{iteration}"
        / "png"
    )
    if point_cloud.is_file():
        print(f"Loading Beta-Splatting point cloud: {point_cloud}")
        beta_model.load_ply(str(point_cloud))
    elif png_model.is_dir():
        print(f"Loading Beta-Splatting PNG model: {png_model}")
        beta_model.load_png(str(png_model))
    else:
        raise FileNotFoundError(
            f"No Beta-Splatting model found for iteration {iteration}"
        )

    render_root = (
        Path(dataset.model_path) / "test" / f"ours_{iteration}"
    )
    renders = render_root / "renders"
    gt = render_root / "gt"
    renders.mkdir(parents=True, exist_ok=True)
    gt.mkdir(parents=True, exist_ok=True)

    cameras = scene.getTestCameras()
    if not cameras:
        raise ValueError(
            "Beta-Splatting has no held-out test cameras. Train with evaluation "
            "split enabled."
        )
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            rendering = torch.clamp(
                beta_model.render(camera)["render"], 0.0, 1.0
            )
            target = torch.clamp(camera.original_image[:3], 0.0, 1.0)
            name = f"{index:05d}.png"
            torchvision.utils.save_image(rendering, renders / name)
            torchvision.utils.save_image(target, gt / name)
            print(f"[{index + 1}/{len(cameras)}] {name}", flush=True)
    print(f"Saved Beta-Splatting render pairs: {render_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
