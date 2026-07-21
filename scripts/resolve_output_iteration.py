from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified3dgs.utils.iterations import resolve_output_iteration


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the latest valid checkpoint iteration for one method output."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requested", type=int, default=None)
    args = parser.parse_args()

    print(resolve_output_iteration(args.output.expanduser().resolve(), args.requested))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
