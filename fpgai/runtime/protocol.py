from __future__ import annotations
import struct
import numpy as np


def pack_cmd(cmd: int) -> bytes:
    return struct.pack("<I", cmd & 0xFFFFFFFF)


def pack_f32_array(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x.tobytes(order="C")


def pack_cmd_and_payload(cmd: int, payload: bytes) -> bytes:
    return pack_cmd(cmd) + payload
