from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional
import struct


class ByteSink(Protocol):
    """
    Minimal abstraction: something you can write bytes into.
    Later:
      - PYNQ DMA writer
      - PCIe XDMA writer
      - TCP/IPC tunnel for virtualized multi-FPGA
    """
    def write(self, data: bytes) -> int: ...


@dataclass
class CommandStreamRuntime:
    """
    Runtime helper for command_stream kernels.
    Kernel expects:
      cmd=3 then weights payload
      cmd=2 then inputs payload
    """
    sink: ByteSink

    def send_cmd(self, cmd: int) -> None:
        self.sink.write(struct.pack("<I", cmd & 0xFFFFFFFF))

    def inject_weights(self, payload: bytes) -> None:
        self.send_cmd(3)
        self.sink.write(payload)

    def run_inference(self, input_payload: bytes) -> None:
        self.send_cmd(2)
        self.sink.write(input_payload)
