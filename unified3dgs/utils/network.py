from __future__ import annotations

import socket
import zlib
from typing import Sequence


def forwarded_has_flag(values: Sequence[object], flag: str) -> bool:
    for value in values:
        text = str(value)
        if text == flag or text.startswith(f"{flag}="):
            return True
    return False


def available_tcp_port(seed_text: str, host: str = "127.0.0.1") -> int:
    """Select a currently free, deterministic high port for an upstream GUI."""
    base = 20000 + (zlib.crc32(seed_text.encode("utf-8")) % 20000)
    for offset in range(1000):
        port = 20000 + ((base - 20000 + offset) % 20000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                handle.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError("could not find an available TCP port for upstream network_gui")
