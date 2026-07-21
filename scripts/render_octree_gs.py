from __future__ import annotations

import argparse
import inspect
import json
import textwrap
import time
from pathlib import Path


def octree_scalar_ape_source(source: str) -> str:
    return source.replace("ape_code[0]", "ape_code")


def install_octree_ape_compatibility() -> None:
    import gaussian_renderer

    original = gaussian_renderer.generate_neural_gaussians
    source = textwrap.dedent(inspect.getsource(original))
    patched_source = octree_scalar_ape_source(source)
    if patched_source == source:
        return
    exec(
        compile(
            patched_source,
            inspect.getsourcefile(original) or "<Octree-GS>",
            "exec",
        ),
        gaussian_renderer.__dict__,
    )
    print("Applied Octree-GS scalar appearance-code compatibility patch.")


def main() -> int:
    import numpy as np
    import torch
    import torchvision
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel, prefilter_voxel, render
    from scene import Scene
    from tqdm import tqdm
    from utils.general_utils import safe_state

    parser = argparse.ArgumentParser(description="Unified Octree-GS renderer")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--ape", type=int, default=10)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    args = get_combined_args(parser)
    args.eval = True
    install_octree_ape_compatibility()
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)

    safe_state(args.quiet)
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.fork,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
            dataset.add_level,
            dataset.visible_threshold,
            dataset.dist2level,
            dataset.base_layer,
            dataset.progressive,
            dataset.extend,
        )
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=args.iteration,
            shuffle=False,
            resolution_scales=dataset.resolution_scales,
        )
        gaussians.eval()
        background = torch.tensor(
            [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )

        def render_set(name, views):
            root = Path(dataset.model_path) / name / f"ours_{scene.loaded_iter}"
            renders = root / "renders"
            gt = root / "gt"
            renders.mkdir(parents=True, exist_ok=True)
            gt.mkdir(parents=True, exist_ok=True)
            times = []
            counts = {}
            for index, view in enumerate(tqdm(views, desc="Rendering progress")):
                torch.cuda.synchronize()
                started = time.time()
                gaussians.set_anchor_mask(
                    view.camera_center,
                    scene.loaded_iter,
                    view.resolution_scale,
                )
                visible = prefilter_voxel(
                    view, gaussians, pipeline, background
                )
                package = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    visible_mask=visible,
                    ape_code=args.ape,
                )
                torch.cuda.synchronize()
                times.append(time.time() - started)
                image = torch.clamp(package["render"], 0.0, 1.0)
                target = torch.clamp(view.original_image[:3], 0.0, 1.0)
                filename = f"{index:05d}.png"
                torchvision.utils.save_image(image, renders / filename)
                torchvision.utils.save_image(target, gt / filename)
                counts[filename] = int(package["visibility_filter"].sum().item())
            (root / "per_view_count.json").write_text(
                json.dumps(counts, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if len(times) > 5:
                print(f"Test FPS: {1.0 / np.mean(times[5:]):.5f}")

        if not args.skip_train:
            render_set("train", scene.getTrainCameras())
        if not args.skip_test:
            render_set("test", scene.getTestCameras())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
