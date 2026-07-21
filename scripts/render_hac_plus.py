from __future__ import annotations

import argparse


def main() -> int:
    import train as upstream
    from arguments import ModelParams, PipelineParams, get_combined_args
    from utils.general_utils import safe_state

    parser = argparse.ArgumentParser(description="Unified HAC++ renderer")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log2", type=int, default=None)
    parser.add_argument("--log2_2D", type=int, default=None)
    parser.add_argument("--n_features", type=int, default=None)
    args = get_combined_args(parser)
    for name, value in {
        "log2": 13,
        "log2_2D": 15,
        "n_features": 4,
    }.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    args.eval = True

    # The compressed codec path is not needed to verify trained rendering and
    # caused a very large allocation in the upstream post-training renderer.
    upstream.run_codec = False

    original_calc_interp_feat = upstream.GaussianModel.calc_interp_feat

    def calc_interp_feat_with_loaded_bounds(model, points):
        # HAC++ computes these bounds during training but does not serialize
        # them in point_cloud.ply. Reconstruct the same bounds after loading.
        if model._anchor.numel() and not bool(
            model.x_bound_min.abs().mean().detach().cpu().item()
        ):
            model.update_anchor_bound()
        return original_calc_interp_feat(model, points)

    upstream.GaussianModel.calc_interp_feat = calc_interp_feat_with_loaded_bounds
    safe_state(args.quiet)
    logger = upstream.get_logger(args.model_path)
    upstream.render_sets(
        args,
        model_params.extract(args),
        args.iteration,
        pipeline_params.extract(args),
        logger=logger,
    )
    print(f"Saved HAC++ render pairs for iteration {args.iteration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
