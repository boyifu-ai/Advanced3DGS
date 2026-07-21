from __future__ import annotations

from unified3dgs.methods.base import BaseMethodAdapter


class ThreeDHGSAdapter(BaseMethodAdapter):
    method_name = "3dhgs"
    third_party_repo = "third_party/3DHGS"
    train_entry = "train.py"
    render_entry = "render.py"
    eval_entry = "metrics.py"

    def entry_for(self, action, config):
        if action == "evaluate":
            return str(
                (
                    config.project_root
                    / "third_party"
                    / "gaussian-splatting"
                    / "metrics.py"
                ).resolve()
            )
        return super().entry_for(action, config)
