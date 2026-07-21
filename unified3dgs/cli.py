from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

from unified3dgs.config.loader import load_config, parse_scalar
from unified3dgs.methods.base import MethodRunConfig
from unified3dgs.methods.registry import available_methods, get_adapter


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_overrides(items: Iterable[str]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key is empty in: {item}")
        overrides[key] = parse_scalar(raw_value.strip())
    return overrides


def build_parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Unified 3DGS {action} entry point.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=available_methods(),
        help="Registered method name.",
    )
    parser.add_argument("--config", required=True, help="Method YAML config.")
    parser.add_argument("--data", required=True, help="Dataset path.")
    parser.add_argument("--output", required=True, help="Output/model path.")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint path used by adapters that support it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing third-party code.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value. Can be passed multiple times.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to the method command.",
    )
    return parser


def main(action: str) -> int:
    parser = build_parser(action)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    method_from_config = config.get("method")
    if method_from_config and method_from_config != args.method:
        parser.error(
            f"--method {args.method!r} does not match config method "
            f"{method_from_config!r}"
        )

    config.update(_parse_overrides(args.overrides))
    config["dataset_path"] = args.data
    config["output_path"] = args.output
    if args.checkpoint_path:
        config["checkpoint_path"] = args.checkpoint_path

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    run_config = MethodRunConfig(
        method=args.method,
        action=action,
        dataset_path=Path(args.data).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        config_path=config_path,
        project_root=_project_root(),
        values=config,
        dry_run=args.dry_run,
        extra_args=extra_args,
    )

    adapter = get_adapter(args.method)
    if action == "train":
        return adapter.train(run_config)
    if action == "render":
        return adapter.render(run_config)
    if action == "evaluate":
        return adapter.evaluate(run_config)
    raise ValueError(f"Unsupported action: {action}")
