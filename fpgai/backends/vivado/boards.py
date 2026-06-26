"""Board metadata used by the FPGAI Vivado bridge.

The Vivado bridge supports board-aware Tcl generation for PYNQ-Z2, KV260,
and KR260. PYNQ-Z2 uses a Zynq-7000 Processing System block-design path;
KV260/KR260 use a Zynq UltraScale+ MPSoC block-design path.

This module is the canonical source for board/device capacity used by
board-fit reports, Vivado gating, runtime packaging metadata, and paper tables.

Resource limits are expressed using normalized FPGAI keys:
  lut, ff, bram_18k, uram, dsp, ddr_bytes

Notes:
  * BRAM is represented as BRAM_18K units because Vitis HLS commonly reports
    BRAM_18K.
  * DDR capacity is board-level memory capacity, not a guaranteed usable PL
    allocation. Runtime/Vivado board execution should still report actual
    allocation limits when known.
  * Clock fields are guide rails for reporting and gating; successful timing
    closure still requires real HLS/Vivado timing reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

    # Fabric resources.
    lut: Optional[int] = None
    ff: Optional[int] = None
    bram_18k: Optional[int] = None
    uram: Optional[int] = None
    dsp: Optional[int] = None

    # Board/system memory capacity.
    ddr_bytes: Optional[int] = None
    ddr_note: str = ""

    # Clock guide rails.
    safe_clock_mhz: Optional[float] = None
    max_clock_mhz: Optional[float] = None

    supports_bridge_generation: bool = True
    supports_hls_ip_export: bool = True
    supports_vivado_synth: bool = True
    supports_vivado_impl: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def resource_limits(self) -> Dict[str, int | float]:
        """Return available board capacity using normalized FPGAI keys."""
        out: Dict[str, int | float] = {}
        if self.lut is not None:
            out["lut"] = int(self.lut)
        if self.ff is not None:
            out["ff"] = int(self.ff)
        if self.bram_18k is not None:
            out["bram_18k"] = int(self.bram_18k)
            # Compatibility alias for older reporting code.
            out["bram"] = int(self.bram_18k)
        if self.uram is not None:
            out["uram"] = int(self.uram)
        if self.dsp is not None:
            out["dsp"] = int(self.dsp)
        if self.ddr_bytes is not None:
            out["ddr_bytes"] = int(self.ddr_bytes)
        if self.default_clock_mhz is not None:
            out["default_clock_mhz"] = float(self.default_clock_mhz)
        if self.safe_clock_mhz is not None:
            out["safe_clock_mhz"] = float(self.safe_clock_mhz)
        if self.max_clock_mhz is not None:
            out["max_clock_mhz"] = float(self.max_clock_mhz)
        return out


_GIB = 1024 ** 3

_BOARDS: Dict[str, BoardInfo] = {
    "pynq_z2": BoardInfo(
        name="pynq_z2",
        vendor="tul",
        family="zynq7000",
        part="xc7z020clg400-1",
        board_part="tul.com.tw:pynq-z2:part0:1.0",
        default_clock_mhz=100.0,
        safe_clock_mhz=100.0,
        max_clock_mhz=None,
        ps_type="processing_system7",
        overlay_style="pynq_bit_hwh",
        lut=53200,
        ff=106400,
        bram_18k=280,
        uram=0,
        dsp=220,
        ddr_bytes=512 * 1024 * 1024,
        ddr_note="PYNQ-Z2 board DDR capacity. Usable PL allocation depends on runtime/system reservation.",
        notes=(
            "Supported Vivado bridge target using Zynq-7000 PS7 and PYNQ "
            "bit/.hwh overlay path."
        ),
    ),
    "kv260": BoardInfo(
        name="kv260",
        vendor="amd",
        family="zynqmp",
        part="xck26-sfvc784-2LV-c",
        board_part="xilinx.com:kv260_som:part0:1.4",
        default_clock_mhz=100.0,
        safe_clock_mhz=100.0,
        max_clock_mhz=None,
        ps_type="zynq_ultra_ps_e",
        overlay_style="kria_xsa_dtbo",
        lut=117120,
        ff=234240,
        bram_18k=288,
        uram=64,
        dsp=1248,
        ddr_bytes=4 * _GIB,
        ddr_note="Kria K26/KV260 DDR capacity. Usable PL allocation depends on Linux/CMA/runtime configuration.",
        notes="Supported Vivado bridge target using ZynqMP PS and Kria XSA handoff path.",
    ),
    "kr260": BoardInfo(
        name="kr260",
        vendor="amd",
        family="zynqmp",
        part="xck26-sfvc784-2LV-c",
        board_part="xilinx.com:kr260_som:part0:1.1",
        default_clock_mhz=100.0,
        safe_clock_mhz=100.0,
        max_clock_mhz=None,
        ps_type="zynq_ultra_ps_e",
        overlay_style="kria_xsa_dtbo",
        lut=117120,
        ff=234240,
        bram_18k=288,
        uram=64,
        dsp=1248,
        ddr_bytes=4 * _GIB,
        ddr_note="Kria K26/KR260 DDR capacity. Usable PL allocation depends on Linux/CMA/runtime configuration.",
        notes="Supported Vivado bridge target using ZynqMP PS and Kria XSA handoff path.",
    ),
}


_PART_ALIASES: Dict[str, str] = {
    "xc7z020clg400-1": "pynq_z2",
    "xck26-sfvc784-2lv-c": "kv260",
    "xck26-sfvc784-2LV-c": "kv260",
}


def _normalize_board_key(name: str | None) -> str:
    return (name or "pynq_z2").strip().lower().replace("-", "_")


def get_board(name: str) -> BoardInfo:
    key = _normalize_board_key(name)
    if key not in _BOARDS:
        valid = ", ".join(sorted(_BOARDS))
        raise KeyError(f"Unknown Vivado board '{name}'. Valid boards: {valid}")
    return _BOARDS[key]


def get_board_by_part(part: str) -> BoardInfo:
    raw = str(part or "").strip()
    key = _PART_ALIASES.get(raw, _PART_ALIASES.get(raw.lower()))
    if not key:
        raise KeyError(f"Unknown FPGA part '{part}'")
    return get_board(key)


def board_resource_limits(name: str | None = None, part: str | None = None) -> Dict[str, int | float]:
    """Return capacity limits for a board name or FPGA part.

    Board name has priority because multiple carrier cards can share the same
    SOM/part but differ in deployment style.
    """
    if name:
        return get_board(name).resource_limits()
    if part:
        return get_board_by_part(part).resource_limits()
    return {}


def board_names() -> list[str]:
    return sorted(_BOARDS)
