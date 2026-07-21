from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.method_catalog import preflight_method, select_methods


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one method preflight in the active backend."
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    method, profile = select_methods([args.method])[0]
    result = preflight_method(
        method,
        profile,
        args.dataset.expanduser().resolve(),
    )
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
