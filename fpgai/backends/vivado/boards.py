"""Board metadata used by the FPGAI Vivado bridge.

The first implementation targets PYNQ-Z2 because the bit/.hwh overlay flow is
simpler than Kria. KV260/KR260 entries are included as metadata only; their
block-design and deployment flows should be validated in later development stages.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class BoardInfo:
    name: str
    vendor: str
    family: str
    part: str
    board_part: Optional[str]
    default_clock_mhz: float
    ps_type: str
    overlay_style: str
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_BOARDS: Dict[str, BoardInfo] = {
    "pynq_z2": BoardInfo(
        name="pynq_z2",
        vendor="tul",
        family="zynq7000",
        part="xc7z020clg400-1",
        board_part="tul.com.tw:pynq-z2:part0:1.0",
        default_clock_mhz=100.0,
        ps_type="processing_system7",
        overlay_style="pynq_bit_hwh",
        notes="Primary Vivado bridge target: simple PYNQ overlay path.",
    ),
    "kv260": BoardInfo(
        name="kv260",
        vendor="amd",
        family="zynqmp",
        part="xck26-sfvc784-2LV-c",
        board_part="xilinx.com:kv260_som:part0:1.4",
        default_clock_mhz=100.0,
        ps_type="zynq_ultra_ps_e",
        overlay_style="kria_xsa_dtbo_later",
        notes="Metadata only at this stage; Kria deployment comes later.",
    ),
    "kr260": BoardInfo(
        name="kr260",
        vendor="amd",
        family="zynqmp",
        part="xck26-sfvc784-2LV-c",
        board_part="xilinx.com:kr260_som:part0:1.1",
        default_clock_mhz=100.0,
        ps_type="zynq_ultra_ps_e",
        overlay_style="kria_xsa_dtbo_later",
        notes="Metadata only at this stage; Kria deployment comes later.",
    ),
}


def get_board(name: str) -> BoardInfo:
    key = (name or "pynq_z2").strip().lower().replace("-", "_")
    if key not in _BOARDS:
        valid = ", ".join(sorted(_BOARDS))
        raise KeyError(f"Unknown Vivado board '{name}'. Valid boards: {valid}")
    return _BOARDS[key]


def board_names() -> list[str]:
    return sorted(_BOARDS)
