from __future__ import annotations

from typing import Dict, List

from unified3dgs.methods.base import BaseMethodAdapter
from unified3dgs.methods.catalog import CatalogMethodAdapter
from unified3dgs.methods.student_splatting_scooping import StudentSplattingScoopingAdapter
from unified3dgs.methods.three_dgs_mcmc import ThreeDGSMCMCAdapter
from unified3dgs.methods.three_dhgs import ThreeDHGSAdapter
from unified3dgs.methods.twodgs import TwoDGSAdapter
from unified3dgs.methods.vanilla_3dgs import Vanilla3DGSAdapter
from unified3dgs.method_catalog import load_confirmed_catalog


_ADAPTERS: Dict[str, BaseMethodAdapter] = {
    "vanilla_3dgs": Vanilla3DGSAdapter(),
    "2dgs": TwoDGSAdapter(),
    "3dgs_mcmc": ThreeDGSMCMCAdapter(),
    "3dhgs": ThreeDHGSAdapter(),
    "sss": StudentSplattingScoopingAdapter(),
}

for _method in load_confirmed_catalog():
    _key = str(_method["key"])
    if _key not in _ADAPTERS:
        _ADAPTERS[_key] = CatalogMethodAdapter(_method)


def available_methods() -> List[str]:
    return sorted(_ADAPTERS)


def get_adapter(name: str) -> BaseMethodAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        supported = ", ".join(available_methods())
        raise ValueError(f"Unknown method {name!r}. Supported methods: {supported}") from exc
