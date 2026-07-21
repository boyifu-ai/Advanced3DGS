from __future__ import annotations

import os
from typing import List

from unified3dgs.methods.base import BaseMethodAdapter, MethodRunConfig
from unified3dgs.utils.network import available_tcp_port, forwarded_has_flag


class Vanilla3DGSAdapter(BaseMethodAdapter):
    method_name = "vanilla_3dgs"
    third_party_repo = "third_party/gaussian-splatting"
    train_entry = "train.py"
    render_entry = "render.py"
    eval_entry = "metrics.py"

    def build_action_args(self, action: str, config: MethodRunConfig) -> List[str]:
        args = super().build_action_args(action, config)
        if action != "train" or forwarded_has_flag(config.extra_args, "--port"):
            return args

        configured = config.values.get("port")
        port = (
            int(configured)
            if configured not in (None, "")
            else available_tcp_port(
                f"{self.method_name}:{config.output_path}:{os.getpid()}"
            )
        )
        args.extend(["--port", str(port)])
        return args
