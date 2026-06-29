from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re


@dataclass(frozen=True)
class MemorySemantics:
    mode: str
    has_weights_mem: bool
    has_weights_m_axi: bool
    has_runtime_payload: bool
    has_full_weight_arrays: bool
    has_const_weight_arrays: bool
    has_uram_weight_bind: bool
    has_bram_weight_bind: bool
    has_tile_weight_buffer: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "has_weights_mem": self.has_weights_mem,
            "has_weights_m_axi": self.has_weights_m_axi,
            "has_runtime_payload": self.has_runtime_payload,
            "has_full_weight_arrays": self.has_full_weight_arrays,
            "has_const_weight_arrays": self.has_const_weight_arrays,
            "has_uram_weight_bind": self.has_uram_weight_bind,
            "has_bram_weight_bind": self.has_bram_weight_bind,
            "has_tile_weight_buffer": self.has_tile_weight_buffer,
            "note": self.note,
        }


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _runtime_payload(run_dir: Path) -> bool:
    p = run_dir / "runtime_package/package_manifest.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
    except Exception:
        return False

    rw = data.get("runtime_weights", {})
    if not isinstance(rw, dict):
        return False

    return (
        bool(rw.get("required"))
        and bool(rw.get("present"))
        and int(rw.get("total_words") or 0) > 0
    )


def classify_generated_memory_semantics(run_dir: str | Path) -> MemorySemantics:
    run = Path(run_dir)
    cpp = _read(run / "hls/src/deeplearn.cpp")
    params = _read(run / "hls/src/fpgai_params.cpp")

    has_weights_mem = "weights_mem" in cpp
    has_weights_m_axi = "m_axi port=weights_mem" in cpp
    has_runtime_payload = _runtime_payload(run)

    # Current generated preload implementation uses these helpers.
    # Their presence means full preload/cache behavior, not scalable DDR tiling.
    has_preload_helper = (
        "fpgai_load_ddr_vector" in cpp
        or "fpgai_load_uram_vector" in cpp
        or "Runtime DDR weight load" in cpp
        or "Runtime-loaded URAM weight storage" in cpp
    )

    # Full local model W/B arrays in deeplearn.cpp.
    has_full_weight_arrays = bool(
        re.search(r"\bop\d+_wgt_t\s+W\d+\s*\[[^\]]+\]", cpp)
        or re.search(r"\bop\d+_bias_t\s+B\d+\s*\[[^\]]+\]", cpp)
        or has_preload_helper
    )

    # Embedded constant W/B arrays in fpgai_params.cpp.
    has_const_weight_arrays = bool(
        re.search(r"\bconst\s+op\d+_wgt_t\s+W\d+\s*\[[^\]]+\]", params)
        or re.search(r"\bconst\s+op\d+_bias_t\s+B\d+\s*\[[^\]]+\]", params)
    )

    has_uram_weight_bind = bool(
        re.search(r"#pragma\s+HLS\s+BIND_STORAGE\s+variable=W\d+\s+type=ram_\dp\s+impl=uram", cpp)
        or re.search(r"#pragma\s+HLS\s+BIND_STORAGE\s+variable=B\d+\s+type=ram_\dp\s+impl=uram", cpp)
        or "Runtime-loaded URAM weight storage" in cpp
    )

    has_bram_weight_bind = bool(
        re.search(r"#pragma\s+HLS\s+BIND_STORAGE\s+variable=W\d+\s+type=ram_\dp\s+impl=bram", cpp)
        or re.search(r"#pragma\s+HLS\s+BIND_STORAGE\s+variable=B\d+\s+type=ram_\dp\s+impl=bram", cpp)
    )

    # Strict detector for future real DDR-tiled implementation.
    # Generic names such as dense_out_in_tiled must not count.
    has_tile_weight_buffer = (
        False
        if has_preload_helper or not has_weights_mem
        else bool(
            re.search(r"\b(?:weight|weights|wgt|W)_tile\s*\[", cpp)
            or re.search(r"\btile_(?:weight|weights|wgt|W)\s*\[", cpp)
        )
    )

    if has_weights_mem and has_weights_m_axi and has_runtime_payload:
        if has_preload_helper and has_uram_weight_bind:
            mode = "uram_preload_full"
            note = (
                "Runtime weights are loaded through weights_mem into full local "
                "URAM-bound W/B arrays. This is URAM-cached/preloaded, not DDR-tiled."
            )
        elif has_preload_helper:
            mode = "ddr_preload_full"
            note = (
                "Runtime weights are loaded through weights_mem into full local W/B arrays. "
                "This is not scalable DDR-tiled execution."
            )
        elif has_tile_weight_buffer and not has_full_weight_arrays:
            mode = "ddr_tiled"
            note = (
                "External weights are accessed through weights_mem with explicit "
                "tile-sized weight buffers and no full local W/B arrays."
            )
        else:
            mode = "invalid_or_ambiguous"
            note = (
                "Runtime weights exist, but generated source does not match "
                "preload-full, URAM-preload-full, or tiled contract."
            )

    elif not has_weights_mem and not has_runtime_payload:
        if has_uram_weight_bind:
            mode = "uram_local"
            note = "Local URAM-bound weights without runtime weight payload."
        elif has_bram_weight_bind:
            mode = "bram_local"
            note = "Local BRAM-bound weights without runtime weight payload."
        elif has_const_weight_arrays:
            mode = "embedded_constants"
            note = "Weights are compiled as local constants."
        else:
            mode = "invalid_or_ambiguous"
            note = "No runtime weights and no recognizable embedded/local weight source."

    else:
        mode = "invalid_or_ambiguous"
        note = "Generated source and runtime package disagree about weight movement."

    return MemorySemantics(
        mode=mode,
        has_weights_mem=has_weights_mem,
        has_weights_m_axi=has_weights_m_axi,
        has_runtime_payload=has_runtime_payload,
        has_full_weight_arrays=has_full_weight_arrays,
        has_const_weight_arrays=has_const_weight_arrays,
        has_uram_weight_bind=has_uram_weight_bind,
        has_bram_weight_bind=has_bram_weight_bind,
        has_tile_weight_buffer=has_tile_weight_buffer,
        note=note,
    )
