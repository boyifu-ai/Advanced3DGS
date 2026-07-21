from __future__ import annotations

import argparse
import inspect
import textwrap


_CONTEXTGS_DEVICE_BUG = "torch.arange(anchor.shape[0])[to_code]"
_CONTEXTGS_DEVICE_FIX = (
    "torch.arange(anchor.shape[0], device=to_code.device)[to_code]"
)


def contextgs_device_safe_source(source: str) -> str:
    if _CONTEXTGS_DEVICE_FIX in source:
        return source
    if _CONTEXTGS_DEVICE_BUG not in source:
        raise RuntimeError(
            "ContextGS multi_scale_generating changed; the verified device "
            "compatibility patch can no longer be applied safely."
        )
    return source.replace(_CONTEXTGS_DEVICE_BUG, _CONTEXTGS_DEVICE_FIX)


def install_contextgs_device_patch() -> None:
    import gaussian_renderer
    import scene.gaussian_model as gaussian_model

    original = gaussian_model.multi_scale_generating
    source = textwrap.dedent(inspect.getsource(original))
    patched_source = contextgs_device_safe_source(source)
    if patched_source == source:
        return
    exec(
        compile(
            patched_source,
            inspect.getsourcefile(original) or "<ContextGS>",
            "exec",
        ),
        gaussian_model.__dict__,
    )
    gaussian_renderer.multi_scale_generating = gaussian_model.multi_scale_generating
    print("Applied ContextGS CPU/CUDA index compatibility patch.")


def main() -> int:
    import test as upstream
    from arguments import ModelParams, PipelineParams, get_combined_args
    from utils.general_utils import safe_state

    parser = argparse.ArgumentParser(description="Unified ContextGS renderer")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--level_num", type=int, default=None)
    parser.add_argument("--level_scale", type=int, default=None)
    parser.add_argument("--log2", type=int, default=None)
    parser.add_argument("--log2_2D", type=int, default=None)
    parser.add_argument("--n_features", type=int, default=None)
    parser.add_argument("--disable_hyper", action="store_true", default=None)
    args = get_combined_args(parser)
    defaults = {
        "level_num": 3,
        "level_scale": 10,
        "log2": 13,
        "log2_2D": 15,
        "n_features": 4,
        "disable_hyper": False,
    }
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    args.eval = True

    # Rendering does not require entropy estimation or bitstream generation.
    # The public estimator also mixes CPU indices with a CUDA mask.
    upstream.run_codec = False
    upstream.GaussianModel.estimate_final_bits = (
        lambda _model: "Unified 3DGS: codec size estimation skipped for rendering"
    )
    install_contextgs_device_patch()

    safe_state(args.quiet)
    logger = upstream.get_logger(args.model_path)
    upstream.render_sets(
        args,
        model_params.extract(args),
        args.iteration,
        pipeline_params.extract(args),
        logger=logger,
    )
    print(f"Saved ContextGS render pairs for iteration {args.iteration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
