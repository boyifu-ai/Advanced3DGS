from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Dict, List


def format_command(command: List[str]) -> str:
    return shlex.join(command)


def run_command(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    dry_run: bool = False,
) -> int:
    print("Working directory:", cwd)
    print("Command:", format_command(command))
    print("CUDA_HOME:", env.get("CUDA_HOME", ""))
    print("TORCH_CUDA_ARCH_LIST:", env.get("TORCH_CUDA_ARCH_LIST", ""))
    pythonpath_prefix = env.get("PYTHONPATH", "").split(";")[:3]
    if len(pythonpath_prefix) == 1:
        pythonpath_prefix = env.get("PYTHONPATH", "").split(":")[:3]
    print("PYTHONPATH prefix:", pythonpath_prefix)
    if dry_run:
        print("Dry run: command was not executed.")
        return 0
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    return int(completed.returncode)
