from __future__ import annotations

from unified3dgs.methods.base import BaseMethodAdapter


class TwoDGSAdapter(BaseMethodAdapter):
    method_name = "2dgs"
    third_party_repo = "third_party/2d-gaussian-splatting"
    train_entry = "train.py"
    render_entry = "render.py"
    eval_entry = "metrics.py"

    def build_action_args(self, action, config):
        args = super().build_action_args(action, config)
        if action == "render":
            if config.values.get("skip_train_render") is True:
                args.append("--skip_train")
            if config.values.get("export_mesh") is False:
                args.append("--skip_mesh")
        return args
