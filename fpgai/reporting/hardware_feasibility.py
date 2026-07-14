#!/usr/bin/env python3
"""Classify FPGAI Vivado bridge artifacts as feasible / resource_fail / timing_fail / tool_fail.

Usage:
  python -m fpgai.reporting.hardware_feasibility experiments/<run_dir>

Inputs:
  <run_dir>/vivado_bridge_evidence/evidence.json
  <run_dir>/vivado_bridge_run_evidence.json, if present
Outputs:
  <run_dir>/hardware_feasibility/feasibility.csv
  <run_dir>/hardware_feasibility/feasibility.md
  <run_dir>/hardware_feasibility/feasibility.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fpgai.backends.vivado.boards import board_resource_limits

# Board resource limits are owned by fpgai.backends.vivado.boards.
# This fallback is used only when the board registry cannot resolve a legacy
# record. Values use normalized FPGAI keys; bram/bram_18k both mean BRAM_18K.
BOARD_LIMITS = {
    "pynq_z2": {
        "lut": 53200,
        "ff": 106400,
        "bram": 280,
        "bram_18k": 280,
        "uram": 0,
        "dsp": 220,
        "ddr_bytes": 512 * 1024 * 1024,
        "default_clock_mhz": 100.0,
        "safe_clock_mhz": 100.0,
    },
    "xc7z020clg400-1": {
        "lut": 53200,
        "ff": 106400,
        "bram": 280,
        "bram_18k": 280,
        "uram": 0,
        "dsp": 220,
        "ddr_bytes": 512 * 1024 * 1024,
        "default_clock_mhz": 100.0,
        "safe_clock_mhz": 100.0,
    },
}

FIELDS = [
    "design", "status", "reason", "board", "part", "bitstream_exists", "xsa_exists",
    "vivado_ok", "vivado_returncode", "power_w", "wns_ns", "lut", "lut_limit",
    "lut_util_pct", "ff", "ff_limit", "ff_util_pct", "bram", "bram_limit",
    "bram_util_pct", "dsp", "dsp_limit", "dsp_util_pct", "energy_j", "error_hint",
]


def _to_float(x: Any) -> Optional[float]:
    if x in (None, "", "None"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    f = _to_float(x)
    if f is None:
        return None
    return int(round(f))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _records_from_evidence(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "designs", "evidence"):
            if isinstance(data.get(key), list):
                return data[key]
        # maybe dict keyed by design
        if all(isinstance(v, dict) for v in data.values()):
            out = []
            for k, v in data.items():
                vv = dict(v)
                vv.setdefault("design", k)
                out.append(vv)
            return out
    raise SystemExit("Could not find artifact records in evidence.json")


def _run_map(run_path: Path) -> Dict[str, Dict[str, Any]]:
    p = run_path / "vivado_bridge_run_evidence.json"
    if not p.exists():
        return {}
    data = _load_json(p)
    records = data.get("records", data if isinstance(data, list) else [])
    out = {}
    for r in records:
        name = r.get("design") or r.get("design_name")
        if name:
            out[name] = r
    return out


def _read_tail(path: Optional[str], n: int = 40000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    txt = p.read_text(errors="ignore")
    return txt[-n:]


def _error_hint(run_rec: Dict[str, Any]) -> str:
    txt = "\n".join([
        str(run_rec.get("error", "")),
        _read_tail(run_rec.get("stdout_log")),
        _read_tail(run_rec.get("stderr_log")),
    ])
    if re.search(r"UTLZ-1|over-utilized|overutilized|requires more .* than are available", txt, re.I):
        return "Vivado DRC resource over-utilization"
    if re.search(r"place_design.*failed|Placer not run", txt, re.I):
        return "Vivado place_design failed"
    if re.search(r"route_design.*failed", txt, re.I):
        return "Vivado route_design failed"
    if re.search(r"timing", txt, re.I) and re.search(r"failed|violation", txt, re.I):
        return "Timing-related implementation failure"
    return ""


def _pct(value: Optional[int], limit: Optional[int]) -> Optional[float]:
    if value is None or not limit:
        return None
    return round(100.0 * value / limit, 2)


def _limits_for(board: str | None, part: str | None = None) -> Dict[str, Any]:
    """Resolve board limits from the canonical board registry."""
    try:
        limits = board_resource_limits(board, part=part)
    except Exception:
        limits = {}
    if limits:
        return dict(limits)
    return BOARD_LIMITS.get(str(board), BOARD_LIMITS.get(str(part), BOARD_LIMITS["pynq_z2"]))


def _first_present(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, ""):
            return mapping.get(key)
    return None


def classify_board_fit(
    resources: Dict[str, Any],
    board: str | None,
    part: str | None = None,
    near_limit_ratio: float = 0.80,
) -> Dict[str, Any]:
    """Classify whether a resource/memory/clock request fits a board.

    Status values:
      fits       : all known dimensions <= near_limit_ratio
      near_limit : at least one known dimension > near_limit_ratio, none > 1.0
      over_limit : any known dimension > 1.0
      unknown    : no usable board limits or no usable request/report values

    Dimensions currently supported:
      lut, ff, bram_18k, uram, dsp, ddr_bytes, target_clock_mhz
    """
    limits = _limits_for(board, part)
    normalized = {
        "lut": _to_int(_first_present(resources, "lut", "LUT", "luts", "LUTs")),
        "ff": _to_int(_first_present(resources, "ff", "FF", "ffs", "FFs")),
        "bram_18k": _to_int(
            _first_present(resources, "bram_18k", "BRAM_18K", "bram", "BRAM", "bram18", "BRAM18")
        ),
        "uram": _to_int(_first_present(resources, "uram", "URAM", "uram_blocks", "URAMs")),
        "dsp": _to_int(_first_present(resources, "dsp", "DSP", "dsps", "DSPs", "DSP48E")),
        "ddr_bytes": _to_int(
            _first_present(
                resources,
                "ddr_bytes",
                "external_memory_bytes",
                "required_ddr_bytes",
                "runtime_memory_bytes",
                "weight_axi_bytes",
            )
        ),
        "target_clock_mhz": _to_float(
            _first_present(resources, "target_clock_mhz", "clock_mhz", "requested_clock_mhz")
        ),
    }

    per_dimension: Dict[str, Dict[str, Any]] = {}
    any_value = False
    any_over = False
    any_near = False
    limiting_dimension = None
    max_ratio = -1.0

    for key, used in normalized.items():
        if key == "target_clock_mhz":
            limit = limits.get("safe_clock_mhz") or limits.get("max_clock_mhz") or limits.get("default_clock_mhz")
        else:
            limit = limits.get(key)
            if limit is None and key == "bram_18k":
                limit = limits.get("bram")

        ratio = None
        status = "unknown"

        if used is not None:
            any_value = True

        if used is not None and limit == 0 and key != "target_clock_mhz":
            ratio = None
            if used > 0:
                status = "over_limit"
                any_over = True
                if max_ratio < float("inf"):
                    max_ratio = float("inf")
                    limiting_dimension = key
            else:
                status = "fits"

        elif used is not None and limit not in (None, 0):
            ratio = float(used) / float(limit)
            if ratio > max_ratio:
                max_ratio = ratio
                limiting_dimension = key

            if key == "target_clock_mhz":
                # Clock guide rails are not the same as fabric capacity.
                # A request above safe/default is a warning until a hard max
                # clock or real Vivado timing result proves it impossible.
                max_clock = limits.get("max_clock_mhz")
                if max_clock not in (None, 0) and float(used) > float(max_clock):
                    status = "over_limit"
                    any_over = True
                elif ratio > 1.0:
                    status = "near_limit"
                    any_near = True
                else:
                    status = "fits"
            elif ratio > 1.0:
                status = "over_limit"
                any_over = True
            elif ratio > near_limit_ratio:
                status = "near_limit"
                any_near = True
            else:
                status = "fits"

        per_dimension[key] = {
            "used": used,
            "available": limit,
            "ratio": None if ratio is None else round(ratio, 6),
            "util_pct": None if ratio is None else round(ratio * 100.0, 2),
            "status": status,
        }

    if not limits or not any_value:
        status = "unknown"
    elif any_over:
        status = "over_limit"
    elif any_near:
        status = "near_limit"
    else:
        status = "fits"

    return {
        "board": board,
        "part": part,
        "status": status,
        "limiting_resource": limiting_dimension,
        "limiting_dimension": limiting_dimension,
        "near_limit_ratio": near_limit_ratio,
        "resources": per_dimension,
        "vivado_allowed": status != "over_limit",
    }


def _nested_resource_value(data: Dict[str, Any], *names: str) -> Any:
    """Find a resource value across common prediction/report layouts.

    FPGAI prediction artifacts may store values directly, under totals/top_level,
    or as fields such as predicted_lut/predicted_bram18. This helper searches
    recursively but returns the first exact/alias key match instead of guessing.
    """
    if not isinstance(data, dict):
        return None

    wanted = {str(n) for n in names}
    wanted_lower = {str(n).lower() for n in names}

    for key, value in data.items():
        key_s = str(key)
        if key_s in wanted or key_s.lower() in wanted_lower:
            if value not in (None, ""):
                return value

    # Search common nested sections first to keep deterministic behavior.
    for section_name in (
        "totals",
        "top_level",
        "summary",
        "resources",
        "area",
        "architecture",
        "architecture_model",
        "model",
        "analytical_model",
        "calibration",
    ):
        section = data.get(section_name)
        if isinstance(section, dict):
            val = _nested_resource_value(section, *names)
            if val not in (None, ""):
                return val

    # Then recursively inspect remaining dictionaries/lists.
    for value in data.values():
        if isinstance(value, dict):
            val = _nested_resource_value(value, *names)
            if val not in (None, ""):
                return val
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    val = _nested_resource_value(item, *names)
                    if val not in (None, ""):
                        return val

    return None


def extract_board_fit_resources(
    resource_data: Dict[str, Any],
    *,
    timing_data: Dict[str, Any] | None = None,
    target_clock_mhz: Any = None,
) -> Dict[str, Any]:
    """Normalize resource/timing fields for classify_board_fit.

    This accepts FPGAI prediction artifacts, HLS/Vivado parsed rows, or simple
    flat dictionaries. Missing dimensions remain absent/unknown; they are not
    faked as zero.
    """
    timing_data = timing_data or {}

    resources = {
        "lut": _nested_resource_value(
            resource_data,
            "lut",
            "LUT",
            "luts",
            "LUTs",
            "predicted_lut",
            "pred_lut",
            "estimated_lut",
            "total_lut",
            "predicted_lut_raw",
        ),
        "ff": _nested_resource_value(
            resource_data,
            "ff",
            "FF",
            "ffs",
            "FFs",
            "predicted_ff",
            "pred_ff",
            "estimated_ff",
            "total_ff",
            "predicted_ff_raw",
        ),
        "bram_18k": _nested_resource_value(
            resource_data,
            "bram_18k",
            "BRAM_18K",
            "bram18",
            "BRAM18",
            "bram",
            "BRAM",
            "predicted_bram18",
            "predicted_bram_18k",
            "pred_bram18",
            "estimated_bram18",
            "estimated_bram_18k",
            "total_bram18",
            "predicted_bram18_raw",
        ),
        "uram": _nested_resource_value(
            resource_data,
            "uram",
            "URAM",
            "uram_blocks",
            "URAMs",
            "predicted_uram",
            "estimated_uram",
            "total_uram",
        ),
        "dsp": _nested_resource_value(
            resource_data,
            "dsp",
            "DSP",
            "dsps",
            "DSPs",
            "DSP48E",
            "predicted_dsp",
            "pred_dsp",
            "estimated_dsp",
            "total_dsp",
            "predicted_dsp_raw",
        ),
        "ddr_bytes": _nested_resource_value(
            resource_data,
            "ddr_bytes",
            "external_memory_bytes",
            "required_ddr_bytes",
            "runtime_memory_bytes",
            "weight_axi_bytes",
            "total_external_bytes",
        ),
    }

    clock = target_clock_mhz
    if clock in (None, ""):
        clock = _nested_resource_value(timing_data, "target_clock_mhz", "clock_mhz", "requested_clock_mhz")
    if clock not in (None, ""):
        resources["target_clock_mhz"] = clock

    return {k: v for k, v in resources.items() if v not in (None, "")}



def _cfg_get_nested(raw: Dict[str, Any] | None, path: str, default: Any = None) -> Any:
    cur: Any = raw or {}
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _storage_values(raw_config: Dict[str, Any] | None) -> list[str]:
    raw = raw_config or {}
    values: list[str] = []
    for path in (
        'memory.weight_storage',
        'memory.activation_storage',
        'memory.gradient_storage',
        'memory.optimizer_state_storage',
        'memory.storage.weights',
        'memory.storage.activations',
        'memory.storage.gradients',
        'memory.storage.optimizer_state',
        'training.weight_storage',
        'training.gradient_storage',
        'training.optimizer_state_storage',
    ):
        val = _cfg_get_nested(raw, path, None)
        if val not in (None, ''):
            values.append(str(val).strip().lower().replace('-', '_'))
    return values


def _runtime_commands(raw_config: Dict[str, Any] | None) -> set[str]:
    raw = raw_config or {}
    seq = _cfg_get_nested(raw, 'runtime.sequence', [])
    out: set[str] = set()
    if not isinstance(seq, list):
        return out
    for entry in seq:
        if isinstance(entry, str):
            out.add(entry.strip().lower().replace('-', '_'))
        elif isinstance(entry, dict):
            for key in entry.keys():
                out.add(str(key).strip().lower().replace('-', '_'))
    return out


def _movement_requested(raw_config: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = raw_config or {}
    dm = raw.get('data_movement', {}) if isinstance(raw.get('data_movement', {}), dict) else {}
    runtime_commands = _runtime_commands(raw)
    requires_dma = False
    axi_stream_ports = 0
    m_axi_bundles = 0
    ddr_required = False
    reasons: list[str] = []

    physical_m_axi_roles: set[str] = set()

    def _role_from_prefix(prefix: str) -> str:
        parts = [part for part in prefix.replace('[', '.').replace(']', '').split('.') if part]
        if not parts:
            return 'data_movement'
        # data_movement.weights.import and data_movement.weights.export share the
        # same generated weights_mem bundle. Likewise, runtime sequence commands
        # should not add extra bundles when the explicit data_movement role is
        # already present. Count physical top-level memory roles, not YAML leaves.
        if parts[0] == 'data_movement' and len(parts) >= 2:
            return parts[1]
        return parts[-1]

    def _add_m_axi_role(role: str, reason: str) -> None:
        nonlocal ddr_required
        normalized = str(role or 'data_movement').strip().lower().replace('-', '_')
        physical_m_axi_roles.add(normalized)
        ddr_required = True
        reasons.append(reason)

    def visit(obj: Any, prefix: str = '') -> None:
        nonlocal requires_dma, axi_stream_ports, ddr_required
        if isinstance(obj, dict):
            iface = str(obj.get('interface', '') or '').strip().lower().replace('-', '_')
            transport = str(obj.get('transport', '') or '').strip().lower().replace('-', '_')
            storage = str(obj.get('storage', '') or '').strip().lower().replace('-', '_')
            mode = str(obj.get('mode', '') or '').strip().lower().replace('-', '_')
            if iface == 'axi_stream' or transport == 'dma':
                requires_dma = True
                axi_stream_ports += 1
                reasons.append(f'{prefix or "data_movement"}: requested AXI-stream/DMA movement')
            if iface == 'm_axi' or storage == 'ddr' or 'ddr' in mode:
                _add_m_axi_role(_role_from_prefix(prefix), f'{prefix or "data_movement"}: requested m_axi/DDR movement')
            for key, val in obj.items():
                visit(val, f'{prefix}.{key}' if prefix else str(key))
        elif isinstance(obj, list):
            for idx, val in enumerate(obj):
                visit(val, f'{prefix}[{idx}]')

    visit(dm, 'data_movement')

    weights_mode = str(_cfg_get_nested(raw, 'weights.mode', _cfg_get_nested(raw, 'data_movement.ps_pl.weights.mode', '')) or '').strip().lower().replace('-', '_')
    if weights_mode in {'import', 'import_export', 'tiled', 'tiled_mutable'}:
        _add_m_axi_role('weights', f'weights.mode={weights_mode} requires runtime/external weight movement')
    if 'export_gradients' in runtime_commands:
        _add_m_axi_role('gradients', 'runtime.sequence requests export_gradients')
    if 'export_optimizer_state' in runtime_commands:
        _add_m_axi_role('optimizer_state', 'runtime.sequence requests export_optimizer_state')
    if 'import_weights' in runtime_commands:
        _add_m_axi_role('weights', 'runtime.sequence requests import_weights')
    if 'export_weights' in runtime_commands:
        _add_m_axi_role('weights', 'runtime.sequence requests export_weights')

    m_axi_bundles = len(physical_m_axi_roles)

    # Count one DMA IP for a bidirectional AXI-stream pair. Keep this tied only
    # to explicit movement requests so embedded inference does not pay for DMA.
    dma_count = 1 if requires_dma else 0
    return {
        'dma_count': dma_count,
        'axi_stream_ports': axi_stream_ports,
        'm_axi_bundles': m_axi_bundles,
        'ddr_required': ddr_required,
        'reasons': reasons,
    }


def derive_board_fit_requirements(raw_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Derive compiler-side board requirements from resolved/user YAML.

    This intentionally counts only requested storage/movement/runtime paths. If
    the user did not request export/import/DMA, the requirement is absent so the
    board-fit report does not penalize unused hardware paths.
    """
    storage = _storage_values(raw_config)
    movement = _movement_requested(raw_config)
    resources: Dict[str, Any] = {}
    checks: list[Dict[str, Any]] = []
    if 'uram' in storage:
        resources['uram'] = max(_to_int(resources.get('uram')) or 0, 1)
        checks.append({
            'name': 'uram_storage_requested',
            'required': True,
            'reason': 'At least one selected storage class uses URAM.',
        })
    if any(v == 'ddr' for v in storage) or movement.get('ddr_required'):
        resources['ddr_bytes'] = max(_to_int(resources.get('ddr_bytes')) or 0, 1)
        checks.append({
            'name': 'ddr_required',
            'required': True,
            'reason': 'Selected storage or movement uses external DDR/m_axi.',
        })
    interface_requirements = {
        'dma_count': int(movement.get('dma_count') or 0),
        'axi_stream_ports': int(movement.get('axi_stream_ports') or 0),
        'm_axi_bundles': int(movement.get('m_axi_bundles') or 0),
        'ddr_required': bool(movement.get('ddr_required')),
        'reasons': list(movement.get('reasons') or []),
    }
    return {
        'resources': resources,
        'storage_values': storage,
        'interface_requirements': interface_requirements,
        'checks': checks,
    }


def _merge_requirement_resources(base: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (derived.get('resources') or {}).items():
        current = _to_int(out.get(key))
        incoming = _to_int(value)
        if incoming is None:
            continue
        if current is None:
            out[key] = incoming
        else:
            out[key] = max(current, incoming)
    return out


def _classify_interface_requirements(requirements: Dict[str, Any], board: str | None, part: str | None = None) -> Dict[str, Any]:
    limits = _limits_for(board, part)
    iface = requirements.get('interface_requirements') or {}
    checks: Dict[str, Dict[str, Any]] = {}
    failures: list[str] = []

    def one(name: str, used_key: str, limit_key: str) -> None:
        used = int(iface.get(used_key) or 0)
        limit = limits.get(limit_key)
        if used <= 0:
            status = 'not_required'
        elif limit is None:
            status = 'unknown'
        elif used > int(limit):
            status = 'over_limit'
            failures.append(name)
        else:
            status = 'fits'
        checks[name] = {'required': used, 'available': limit, 'status': status}

    one('axi_dma_count', 'dma_count', 'max_axi_dma')
    one('axi_stream_ports', 'axi_stream_ports', 'max_axi_stream_ports')
    one('m_axi_bundles', 'm_axi_bundles', 'max_m_axi_bundles')
    ddr_required = bool(iface.get('ddr_required'))
    supports_ddr = bool(limits.get('supports_ddr', True))
    if not ddr_required:
        ddr_status = 'not_required'
    elif not supports_ddr:
        ddr_status = 'over_limit'
        failures.append('ddr_interface')
    else:
        ddr_status = 'fits'
    checks['ddr_interface'] = {'required': ddr_required, 'available': supports_ddr, 'status': ddr_status}
    status = 'over_limit' if failures else 'fits'
    if any(v.get('status') == 'unknown' for v in checks.values()) and status == 'fits':
        status = 'unknown'
    return {'status': status, 'checks': checks, 'failures': failures, 'reasons': list(iface.get('reasons') or [])}

def _suggest_yaml_actions(fit: Dict[str, Any]) -> list[str]:
    limiting = str(fit.get("limiting_dimension") or fit.get("limiting_resource") or "")
    status = str(fit.get("status") or "unknown")

    if status == "fits":
        return [
            "This design is within the current board-fit guide rails.",
            "You can keep the current YAML settings for this board-fit stage.",
            "Still validate timing, power, and runtime on real Vivado/board artifacts before deployment claims.",
        ]

    if limiting == "dsp":
        return [
            "Reduce optimization.parallel.pe, optimization.parallel.simd, or optimization.parallel.unroll_factor.",
            "Use a resource-oriented policy such as DSP-Saver/resource_first if available.",
            "Reduce manual conv/dense unroll overrides.",
            "Try a lower precision_mode if accuracy remains acceptable.",
            "Use hardware.fit_policy: report_only only for design-space analysis, not deployment.",
        ]

    if limiting in {"lut", "ff"}:
        return [
            "Reduce parallelism/unroll factors in YAML.",
            "Prefer a resource-oriented policy instead of aggressive/latency-first settings.",
            "Reduce tiling buffer fanout or array partition factors.",
            "Check generated HLS comments/macros to see which manual override expanded logic.",
        ]

    if limiting == "bram_18k":
        return [
            "Reduce tile sizes that create local buffers.",
            "Change memory.weight_storage or activation storage strategy if it reduces on-chip buffers.",
            "Consider streaming weights/activations when supported by the design.",
            "Inspect precision_layout.json and memory_plan.json for buffer byte pressure.",
        ]

    if limiting == "uram":
        return [
            "Disable or reduce URAM-backed buffers with memory.use_uram: false when supported.",
            "Reduce large tile sizes and activation cache requirements.",
            "Prefer BRAM/DDR placement only when the generated memory plan confirms it is supported.",
        ]

    if limiting == "ddr_bytes":
        return [
            "Reduce model size, batch size, cached activations, or optimizer-state storage.",
            "Use lower precision for weights/activations/optimizer state if accuracy allows.",
            "Check runtime memory/CMA limits; board DDR capacity is not always fully available to PL/runtime.",
        ]

    if limiting == "target_clock_mhz":
        return [
            "The requested clock is above the board safe/default guide rail.",
            "This is allowed as an experiment, but it requires Vitis HLS/Vivado timing validation before deployment claims.",
            "Lower targets.platform.clocks[0].target_mhz for a safer first implementation, or keep the higher clock and require timing reports.",
            "Use pipeline settings to improve timing, but do not claim timing closure until Vivado reports pass.",
        ]

    return [
        "Inspect reports/resource_prediction.json, reports/timing_prediction.json, and ir/compile_plan.json.",
        "If the limiting dimension is unknown, add or fix the resource extractor for that artifact type.",
        "Do not treat unknown board-fit as deployable until real HLS/Vivado/board reports exist.",
    ]


def board_fit_markdown(payload: Dict[str, Any]) -> str:
    fit = payload.get("fit", {})
    resources = fit.get("resources", {}) if isinstance(fit, dict) else {}
    guidance = payload.get("suggested_yaml_actions", [])

    lines = [
        "# FPGAI board-fit report",
        "",
        f"- source: `{payload.get('source', '')}`",
        f"- board: `{payload.get('board', '')}`",
        f"- part: `{payload.get('part', '')}`",
        f"- status: `{fit.get('status', 'unknown')}`",
        f"- limiting_dimension: `{fit.get('limiting_dimension') or fit.get('limiting_resource') or ''}`",
        f"- vivado_allowed_by_fit: `{fit.get('vivado_allowed')}`",
        "",
        "## Resource / memory / clock dimensions",
        "",
        "| dimension | used | available | utilization | status |",
        "|---|---:|---:|---:|---|",
    ]

    for key in ("lut", "ff", "bram_18k", "uram", "dsp", "ddr_bytes", "target_clock_mhz"):
        row = resources.get(key, {}) if isinstance(resources, dict) else {}
        util = row.get("util_pct")
        util_s = "" if util is None else f"{util}%"
        lines.append(
            f"| {key} | {row.get('used', '')} | {row.get('available', '')} | {util_s} | {row.get('status', 'unknown')} |"
        )

    interface_fit = payload.get("interface_fit", {}) if isinstance(payload.get("interface_fit"), dict) else {}
    interface_checks = interface_fit.get("checks", {}) if isinstance(interface_fit.get("checks"), dict) else {}
    lines.extend([
        "",
        "## Interface requirements",
        "",
        "| check | required | available | status |",
        "|---|---:|---:|---|",
    ])
    for key in ("axi_dma_count", "axi_stream_ports", "m_axi_bundles", "ddr_interface"):
        row = interface_checks.get(key, {}) if isinstance(interface_checks, dict) else {}
        lines.append(f"| {key} | {row.get('required', '')} | {row.get('available', '')} | {row.get('status', 'unknown')} |")

    gating = payload.get("build_stage_gating", {}) if isinstance(payload.get("build_stage_gating"), dict) else {}
    lines.extend([
        "",
        "## Stage gating",
        "",
    ])
    for stage in ("vivado_project", "vivado_implementation", "bitstream"):
        row = gating.get(stage, {}) if isinstance(gating, dict) else {}
        lines.append(f"- {stage}: requested=`{row.get('requested')}` allowed=`{row.get('allowed')}`")

    lines.extend([
        "",
        "## Guidance",
        "",
    ])
    for item in guidance:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Truth boundary",
        "",
        "- This report classifies board fit from the available resource/memory/clock data.",
        "- Prediction-based board fit is not a replacement for Vitis HLS, Vivado implementation, timing, power, or real-board runtime validation.",
        "- Over-limit designs may still be useful for design-space analysis, but they should not be treated as deployable for the selected board.",
        "",
    ])

    return "\n".join(lines)


def emit_board_fit_report(
    reports_dir: Path,
    *,
    resource_data: Dict[str, Any],
    timing_data: Dict[str, Any] | None = None,
    board: str | None = None,
    part: str | None = None,
    target_clock_mhz: Any = None,
    source: str = "prediction",
    raw_config: Dict[str, Any] | None = None,
    build_stages: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Write reports/board_fit.json and reports/board_fit.md."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    normalized_resources = extract_board_fit_resources(
        resource_data,
        timing_data=timing_data,
        target_clock_mhz=target_clock_mhz,
    )
    derived_requirements = derive_board_fit_requirements(raw_config)
    normalized_resources = _merge_requirement_resources(normalized_resources, derived_requirements)
    fit = classify_board_fit(normalized_resources, board=board, part=part)
    interface_fit = _classify_interface_requirements(derived_requirements, board=board, part=part)
    if interface_fit.get("status") == "over_limit":
        fit = dict(fit)
        interface_failures = list(interface_fit.get("failures") or [])
        limiting_interface = str(interface_failures[0]) if interface_failures else "interfaces"
        fit["status"] = "over_limit"
        fit["limiting_resource"] = limiting_interface
        fit["limiting_dimension"] = limiting_interface
        fit["vivado_allowed"] = False
    stages = build_stages or {}
    vivado_implementation_allowed = bool(fit.get("vivado_allowed"))
    bitstream_allowed = bool(fit.get("vivado_allowed"))

    payload = {
        "format": "fpgai.board_fit.v1",
        "source": source,
        "board": board,
        "part": part,
        "status": fit.get("status"),
        "limiting_dimension": fit.get("limiting_dimension") or fit.get("limiting_resource"),
        "vivado_implementation_allowed": vivado_implementation_allowed,
        "bitstream_allowed": bitstream_allowed,
        "build_stage_gating": {
            "vivado_project": {"requested": bool(stages.get("vivado_project")), "allowed": True},
            "vivado_implementation": {"requested": bool(stages.get("vivado_implementation")), "allowed": vivado_implementation_allowed},
            "bitstream": {"requested": bool(stages.get("bitstream")), "allowed": bitstream_allowed},
        },
        "normalized_resources": normalized_resources,
        "derived_requirements": derived_requirements,
        "interface_fit": interface_fit,
        "fit": fit,
        "suggested_yaml_actions": _suggest_yaml_actions(fit),
        "truth_boundary": {
            "prediction_based": source == "prediction",
            "requires_hls_for_synthesis_truth": True,
            "requires_vivado_for_implementation_truth": True,
            "requires_board_runtime_for_deployment_truth": True,
            "unrequested_paths_not_counted": True,
        },
    }

    json_path = reports_dir / "board_fit.json"
    md_path = reports_dir / "board_fit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(board_fit_markdown(payload), encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "source": source,
        "board": board,
        "part": part,
        "status": fit.get("status"),
        "limiting_dimension": fit.get("limiting_dimension"),
        "vivado_allowed": fit.get("vivado_allowed"),
        "vivado_implementation_allowed": vivado_implementation_allowed,
        "bitstream_allowed": bitstream_allowed,
    }


def classify_record(rec: Dict[str, Any], run_rec: Dict[str, Any], default_board: str = "pynq_z2") -> Dict[str, Any]:
    design = rec.get("design") or rec.get("design_name") or rec.get("name") or "unknown"
    board = rec.get("board") or run_rec.get("board") or default_board
    part = rec.get("part") or run_rec.get("part") or ""
    limits = _limits_for(board, part)

    lut = _to_int(rec.get("lut") or rec.get("LUT"))
    ff = _to_int(rec.get("ff") or rec.get("FF"))
    bram = _to_int(rec.get("bram") or rec.get("BRAM"))
    dsp = _to_int(rec.get("dsp") or rec.get("DSP"))
    wns = _to_float(rec.get("wns_ns") or rec.get("WNS_ns") or rec.get("WNS"))
    power = _to_float(rec.get("total_power_w") or rec.get("power_w"))
    energy = _to_float(rec.get("estimated_energy_j") or rec.get("energy_j"))
    bit = bool(rec.get("bitstream_exists") or rec.get("bit") or run_rec.get("bitstream"))
    xsa = bool(rec.get("xsa_exists") or rec.get("xsa") or run_rec.get("xsa"))
    vivado_ok = run_rec.get("vivado_ok")
    vivado_returncode = run_rec.get("vivado_returncode")

    over = []
    for key, val in (("lut", lut), ("ff", ff), ("bram", bram), ("dsp", dsp)):
        lim = limits.get(key)
        if val is not None and lim is not None and val > lim:
            over.append(f"{key.upper()} {val}>{lim}")

    hint = _error_hint(run_rec)
    status = "unknown"
    reason = ""
    if bit and xsa and (wns is None or wns >= 0):
        status = "pass"
        reason = "bitstream/xsa generated and timing closed or no timing violation parsed"
    elif over or "resource" in hint.lower() or "over-util" in hint.lower():
        status = "resource_fail"
        reason = "; ".join(over) if over else hint
    elif wns is not None and wns < 0:
        status = "timing_fail"
        reason = f"negative WNS {wns} ns"
    elif not bit or not xsa or vivado_ok is False or vivado_returncode not in (None, 0, "0"):
        status = "tool_fail"
        reason = hint or "Vivado did not produce bitstream/xsa"

    return {
        "design": design,
        "status": status,
        "reason": reason,
        "board": board,
        "part": part,
        "bitstream_exists": bit,
        "xsa_exists": xsa,
        "vivado_ok": vivado_ok,
        "vivado_returncode": vivado_returncode,
        "power_w": power,
        "wns_ns": wns,
        "lut": lut,
        "lut_limit": limits.get("lut"),
        "lut_util_pct": _pct(lut, limits.get("lut")),
        "ff": ff,
        "ff_limit": limits.get("ff"),
        "ff_util_pct": _pct(ff, limits.get("ff")),
        "bram": bram,
        "bram_limit": limits.get("bram_18k", limits.get("bram")),
        "bram_util_pct": _pct(bram, limits.get("bram_18k", limits.get("bram"))),
        "dsp": dsp,
        "dsp_limit": limits.get("dsp"),
        "dsp_util_pct": _pct(dsp, limits.get("dsp")),
        "energy_j": energy,
        "error_hint": hint,
    }


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["design", "status", "reason", "bitstream_exists", "xsa_exists", "power_w", "wns_ns", "lut", "lut_util_pct", "ff", "bram", "dsp", "dsp_util_pct", "energy_j"]
    lines = ["# FPGAI hardware feasibility classification", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    run_path = Path(argv[1])
    ev_path = run_path / "vivado_bridge_evidence" / "evidence.json"
    if not ev_path.exists():
        raise SystemExit(f"Missing {ev_path}")
    records = _records_from_evidence(_load_json(ev_path))
    runs = _run_map(run_path)
    rows = []
    for rec in records:
        design = rec.get("design") or rec.get("design_name") or rec.get("name") or "unknown"
        rows.append(classify_record(rec, runs.get(design, {})))
    rows.sort(key=lambda r: r["design"])

    out = run_path / "hardware_feasibility"
    out.mkdir(parents=True, exist_ok=True)
    (out / "feasibility.json").write_text(json.dumps(rows, indent=2))
    with (out / "feasibility.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    write_md(out / "feasibility.md", rows)
    print((out / "feasibility.md").read_text())
    print(f"[OK] Wrote {out / 'feasibility.json'}")
    print(f"[OK] Wrote {out / 'feasibility.csv'}")
    print(f"[OK] Wrote {out / 'feasibility.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
