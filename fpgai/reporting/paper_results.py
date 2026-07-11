from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
ARTIFACT_KIND = "paper_master_results"

MASTER_RESULT_FIELDS: list[str] = [
    "schema_version",
    "design_id",
    "out_dir",
    "model",
    "model_path",
    "dataset",
    "board",
    "part",
    "mode",
    "precision",
    "memory_strategy",
    "data_movement",
    "parallel_factor",
    "pipeline_policy",
    "tiling",
    "training_optimizer",
    "training_loss",
    "build_cpp",
    "build_hls_project",
    "build_hls_synthesis",
    "build_vivado_project",
    "build_vivado_implementation",
    "build_bitstream",
    "runtime_package_status",
    "estimated_lut",
    "estimated_ff",
    "estimated_bram",
    "estimated_bram18",
    "estimated_dsp",
    "estimated_uram",
    "estimated_fmax_mhz",
    "estimated_clock_mhz",
    "estimated_latency_ms",
    "estimated_cycles",
    "estimated_throughput_fps",
    "estimated_parallel_macs",
    "estimated_memory_bytes",
    "estimated_weight_bytes",
    "estimated_activation_bytes",
    "estimated_gradient_bytes",
    "estimated_optimizer_state_bytes",
    "board_fit_status",
    "board_fit_limiting_dimension",
    "board_fit_vivado_allowed",
    "board_fit_bitstream_allowed",
    "generated_cpp_status",
    "movement_validation_status",
    "movement_validation_passed",
    "runtime_sequence_commands",
    "runtime_buffer_count",
    "hls_status",
    "hls_lut",
    "hls_ff",
    "hls_bram",
    "hls_dsp",
    "hls_latency_min",
    "hls_latency_max",
    "hls_ii",
    "hls_clock_period_ns",
    "vivado_project_status",
    "vivado_implementation_status",
    "vivado_lut",
    "vivado_ff",
    "vivado_bram",
    "vivado_dsp",
    "vivado_wns",
    "vivado_tns",
    "vivado_power_w",
    "bitstream_status",
    "deployment_package_status",
    "runtime_status",
    "runtime_board",
    "runtime_sample_count",
    "runtime_latency_ms_mean",
    "runtime_latency_ms_p50",
    "runtime_latency_ms_p95",
    "runtime_throughput",
    "runtime_accuracy",
    "runtime_loss_before",
    "runtime_loss_after",
    "runtime_training_step_ms",
    "support_status",
    "required_validation",
]

_NUMERIC_FIELDS = {
    "estimated_lut",
    "estimated_ff",
    "estimated_bram",
    "estimated_bram18",
    "estimated_dsp",
    "estimated_uram",
    "estimated_fmax_mhz",
    "estimated_clock_mhz",
    "estimated_latency_ms",
    "estimated_cycles",
    "estimated_throughput_fps",
    "estimated_parallel_macs",
    "estimated_memory_bytes",
    "estimated_weight_bytes",
    "estimated_activation_bytes",
    "estimated_gradient_bytes",
    "estimated_optimizer_state_bytes",
    "hls_lut",
    "hls_ff",
    "hls_bram",
    "hls_dsp",
    "hls_latency_min",
    "hls_latency_max",
    "hls_ii",
    "hls_clock_period_ns",
    "vivado_lut",
    "vivado_ff",
    "vivado_bram",
    "vivado_dsp",
    "vivado_wns",
    "vivado_tns",
    "vivado_power_w",
    "runtime_sample_count",
    "runtime_latency_ms_mean",
    "runtime_latency_ms_p50",
    "runtime_latency_ms_p95",
    "runtime_throughput",
    "runtime_accuracy",
    "runtime_loss_before",
    "runtime_loss_after",
    "runtime_training_step_ms",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _nested(data: Mapping[str, Any], path: Sequence[str], default: Any = "") -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(key, default)
    return cur


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _as_number(value: Any) -> float | int | str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in {"none", "nan", "n/a", "not_run", "missing"}:
            return ""
        num = float(text)
        return int(num) if num.is_integer() else num
    except Exception:
        return ""


def _find_number(data: Any, *candidate_keys: str) -> float | int | str:
    keys = {_norm_key(k) for k in candidate_keys}
    seen: set[int] = set()

    def walk(obj: Any) -> float | int | str:
        oid = id(obj)
        if oid in seen:
            return ""
        seen.add(oid)
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if _norm_key(str(k)) in keys:
                    num = _as_number(v)
                    if num != "":
                        return num
            for v in obj.values():
                num = walk(v)
                if num != "":
                    return num
        elif isinstance(obj, list):
            for v in obj:
                num = walk(v)
                if num != "":
                    return num
        return ""

    return walk(data)


def _status(data: Mapping[str, Any], default: str = "not_run") -> str:
    for key in ("status", "tool_status", "validation_status", "implementation_status", "bitstream_status"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    if data:
        return "present"
    return default


_NOT_AVAILABLE_STATUSES = {"", "not_run", "missing", "not_requested", "required_validation"}


def _has_result_status(value: Any) -> bool:
    return str(value or "").strip().lower() not in _NOT_AVAILABLE_STATUSES


def _needs_validation_status(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "not_run", "missing", "not_requested", "required_validation"}


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return ""


def _join_sequence(values: Any) -> str:
    if isinstance(values, list):
        out: list[str] = []
        for item in values:
            if isinstance(item, Mapping):
                out.append(str(_first_nonempty(item.get("command"), item.get("cmd"), item.get("mode"), item.get("name"))))
            else:
                out.append(str(item))
        return ";".join(x for x in out if x)
    return ""


def _load_reports(out_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "manifest": _read_json(out_dir / "manifest.json"),
        "model_profile": _read_json(out_dir / "reports" / "model_profile.json"),
        "resource_prediction": _read_json(out_dir / "reports" / "resource_prediction.json"),
        "timing_prediction": _read_json(out_dir / "reports" / "timing_prediction.json"),
        "board_fit": _read_json(out_dir / "reports" / "board_fit.json"),
        "hardware_knob_contract": _read_json(out_dir / "reports" / "hardware_knob_contract.json"),
        "generated_cpp_validation": _read_json(out_dir / "reports" / "generated_cpp_validation.json"),
        "generated_hls_explanation": _read_json(out_dir / "reports" / "generated_hls_explanation.json"),
        "movement_contract_validation": _read_json(out_dir / "reports" / "movement_contract_validation.json"),
        "runtime_sequence": _read_json(out_dir / "reports" / "runtime_sequence.json"),
        "buffer_plan": _read_json(out_dir / "runtime_package" / "buffer_plan.json"),
        "package_manifest": _read_json(out_dir / "runtime_package" / "package_manifest.json"),
        "training_plan": _read_json(out_dir / "training" / "training_plan.json"),
        "vivado_bd_validation": _read_json(out_dir / "reports" / "vivado_bd_validation.json"),
        "hls_synthesis_validation": _read_json(out_dir / "reports" / "hls_synthesis_validation.json"),
        "hls_synthesis_report": _read_json(out_dir / "reports" / "hls_synthesis_report.json"),
        "vivado_implementation_validation": _read_json(out_dir / "reports" / "vivado_implementation_validation.json"),
        "vivado_implementation_report": _read_json(out_dir / "reports" / "vivado_implementation_report.json"),
        "deployment_package_validation": _read_json(out_dir / "reports" / "deployment_package_validation.json"),
        "runtime_results": _read_json(out_dir / "reports" / "runtime_results.json"),
    }


def _knob_value(contract: Mapping[str, Any], *names: str) -> str:
    knobs = contract.get("knobs")
    if not isinstance(knobs, list):
        return ""
    wanted = {_norm_key(n) for n in names}
    for knob in knobs:
        if not isinstance(knob, Mapping):
            continue
        joined = " ".join(str(knob.get(k, "")) for k in ("name", "path", "config_path", "key", "feature"))
        if any(name in _norm_key(joined) for name in wanted):
            value = _first_nonempty(knob.get("value"), knob.get("resolved_value"), knob.get("effective_value"), knob.get("requested_value"))
            return str(value) if value != "" else ""
    return ""




def _find_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _parse_xml_numbers(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(path).getroot()
    except Exception:
        return {}

    values: dict[str, Any] = {}

    def clean_tag(tag: str) -> str:
        return tag.split('}', 1)[-1] if '}' in tag else tag

    def number(text: Any) -> float | int | str:
        return _as_number(text)

    tag_map = {
        'lut': 'hls_lut',
        'ff': 'hls_ff',
        'bram18k': 'hls_bram',
        'bram_18k': 'hls_bram',
        'dsp': 'hls_dsp',
        'bestcaselatency': 'hls_latency_min',
        'averagecaselatency': 'hls_latency_avg',
        'worstcaselatency': 'hls_latency_max',
        'intervalmin': 'hls_ii',
        'intervalmax': 'hls_ii_max',
        'estimatedclockperiod': 'hls_clock_period_ns',
        'targetclockperiod': 'hls_target_clock_period_ns',
    }
    for elem in root.iter():
        key = _norm_key(clean_tag(elem.tag))
        mapped = tag_map.get(key)
        if mapped and mapped not in values:
            num = number(elem.text)
            if num != '':
                values[mapped] = num
    return values


def _parse_hls_reports(root: Path) -> dict[str, Any]:
    report_json = _read_json(root / 'reports' / 'hls_synthesis_report.json')
    values: dict[str, Any] = {}
    xml_path = _find_existing([
        root / 'hls' / 'fpgai_hls_proj' / 'sol1' / 'syn' / 'report' / 'deeplearn_csynth.xml',
        root / 'hls' / 'fpgai_hls_proj' / 'sol1' / 'syn' / 'report' / 'csynth.xml',
    ])
    if xml_path is None:
        report_dir = root / 'hls' / 'fpgai_hls_proj' / 'sol1' / 'syn' / 'report'
        matches = sorted(report_dir.glob('*_csynth.xml')) if report_dir.exists() else []
        xml_path = matches[0] if matches else None
    if xml_path is not None:
        values.update(_parse_xml_numbers(xml_path))
        values['hls_report_source'] = str(xml_path)

    # Text fallback for simple report summaries.
    rpt_path = _find_existing([
        root / 'hls' / 'fpgai_hls_proj' / 'sol1' / 'syn' / 'report' / 'csynth.rpt',
        root / 'hls' / 'fpgai_hls_proj' / 'sol1' / 'syn' / 'report' / 'deeplearn_csynth.rpt',
    ])
    if rpt_path is not None and not values.get('hls_latency_max'):
        try:
            text = rpt_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            text = ''
        # Vitis report tables vary, so only use conservative key/value matches here.
        patterns = {
            'hls_latency_min': [r'Latency\s*\(cycles\).*?min\s*[:=]\s*([0-9.]+)', r'Best-caseLatency\D+([0-9.]+)'],
            'hls_latency_max': [r'Latency\s*\(cycles\).*?max\s*[:=]\s*([0-9.]+)', r'Worst-caseLatency\D+([0-9.]+)'],
            'hls_ii': [r'Interval\s*\(cycles\).*?min\s*[:=]\s*([0-9.]+)', r'Interval-min\D+([0-9.]+)'],
            'hls_clock_period_ns': [r'Estimated\s+Clock\s+Period\s*[:=]\s*([0-9.]+)'],
        }
        for field, pats in patterns.items():
            if values.get(field) not in (None, ''):
                continue
            for pat in pats:
                m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
                if m:
                    num = _as_number(m.group(1))
                    if num != '':
                        values[field] = num
                        break
        if values and 'hls_report_source' not in values:
            values['hls_report_source'] = str(rpt_path)

    # JSON values override only when they contain actual numeric data.
    for target, keys in {
        'hls_lut': ('lut', 'luts', 'LUT'),
        'hls_ff': ('ff', 'ffs', 'FF'),
        'hls_bram': ('bram', 'bram18', 'BRAM_18K'),
        'hls_dsp': ('dsp', 'dsps', 'DSP'),
        'hls_latency_min': ('latency_min', 'min_latency'),
        'hls_latency_max': ('latency_max', 'max_latency'),
        'hls_ii': ('ii', 'interval'),
        'hls_clock_period_ns': ('clock_period_ns', 'estimated_clock_period_ns'),
    }.items():
        num = _find_number(report_json, *keys)
        if num != '':
            values[target] = num
    if report_json and 'hls_status' not in values:
        values['hls_status'] = _status(report_json, 'present')
    elif values:
        values['hls_status'] = 'passed'
    return values


def _parse_vivado_utilization(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {}
    values: dict[str, Any] = {}

    def first_number_after(label_patterns: Sequence[str]) -> float | int | str:
        for pat in label_patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                num = _as_number(m.group(1))
                if num != '':
                    return num
        return ''

    values['vivado_lut'] = first_number_after([r'\|\s*CLB LUTs\*?\s*\|\s*([0-9,]+)', r'\|\s*Slice LUTs\s*\|\s*([0-9,]+)'])
    values['vivado_ff'] = first_number_after([r'\|\s*CLB Registers\s*\|\s*([0-9,]+)', r'\|\s*Slice Registers\s*\|\s*([0-9,]+)'])
    values['vivado_dsp'] = first_number_after([r'\|\s*DSPs\s*\|\s*([0-9,]+)', r'\|\s*DSP48E2\s*\|\s*([0-9,]+)'])
    values['vivado_bram'] = first_number_after([r'\|\s*Block RAM Tile\s*\|\s*([0-9,.]+)', r'\|\s*RAMB36/FIFO\*?\s*\|\s*([0-9,.]+)'])
    return {k: v for k, v in values.items() if v != ''}


def _parse_vivado_timing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {}
    values: dict[str, Any] = {}
    # Prefer implemented Timing Details Setup line.
    m = re.search(r'Setup\s*:\s*\d+\s+Failing Endpoints,\s*Worst Slack\s*([+-]?[0-9.]+)ns,\s*Total Violation\s*([+-]?[0-9.]+)ns', text, re.IGNORECASE)
    if m:
        values['vivado_wns'] = _as_number(m.group(1))
        values['vivado_tns'] = _as_number(m.group(2))
    else:
        # Fallback to Design Timing Summary table first numeric columns.
        m2 = re.search(r'WNS\(ns\).*?\n[-\s]*\n?\s*([+-]?[0-9.]+)\s+([+-]?[0-9.]+)', text, re.IGNORECASE | re.DOTALL)
        if m2:
            values['vivado_wns'] = _as_number(m2.group(1))
            values['vivado_tns'] = _as_number(m2.group(2))
    return values


def _parse_vivado_power(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {}
    m = re.search(r'\|\s*Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)', text, re.IGNORECASE)
    if not m:
        m = re.search(r'Total On-Chip Power[^0-9]*([0-9.]+)', text, re.IGNORECASE)
    return {'vivado_power_w': _as_number(m.group(1))} if m else {}


def _parse_vivado_reports(root: Path) -> dict[str, Any]:
    report_json = _read_json(root / 'reports' / 'vivado_implementation_report.json')
    bridge_reports = root / 'vivado_bridge' / 'reports'
    values: dict[str, Any] = {}
    values.update(_parse_vivado_utilization(bridge_reports / 'utilization_impl.rpt'))
    values.update(_parse_vivado_timing(bridge_reports / 'timing_impl.rpt'))
    values.update(_parse_vivado_power(bridge_reports / 'power_impl.rpt'))

    for target, keys in {
        'vivado_lut': ('lut', 'luts'),
        'vivado_ff': ('ff', 'ffs'),
        'vivado_bram': ('bram', 'bram18'),
        'vivado_dsp': ('dsp', 'dsps'),
        'vivado_wns': ('wns',),
        'vivado_tns': ('tns',),
        'vivado_power_w': ('power_w', 'total_power_w', 'total_on_chip_power_w'),
    }.items():
        num = _find_number(report_json, *keys)
        if num != '':
            values[target] = num

    manifest = _read_json(root / 'manifest.json')
    bridge = manifest.get('vivado_bridge') if isinstance(manifest.get('vivado_bridge'), Mapping) else {}
    if bridge.get('vivado_impl_requested') is True and not values.get('vivado_implementation_status'):
        values['vivado_implementation_status'] = 'passed' if values or bridge.get('ok') is True else 'requested'
    elif report_json and not values.get('vivado_implementation_status'):
        values['vivado_implementation_status'] = _status(report_json, 'present')
    return values


def build_master_result_row(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    reports = _load_reports(root)
    manifest = reports["manifest"]
    package = reports["package_manifest"]
    profile = reports["model_profile"]
    resource = reports["resource_prediction"]
    timing = reports["timing_prediction"]
    board_fit = reports["board_fit"]
    knob = reports["hardware_knob_contract"]
    hls = reports["hls_synthesis_validation"] or reports["hls_synthesis_report"]
    vivado = reports["vivado_implementation_validation"] or reports["vivado_implementation_report"]
    parsed_hls = _parse_hls_reports(root)
    parsed_vivado = _parse_vivado_reports(root)
    deployment = reports["deployment_package_validation"]
    runtime = reports["runtime_results"]
    training = reports["training_plan"]
    movement = reports["movement_contract_validation"]
    runtime_sequence = reports["runtime_sequence"]
    buffer_plan = reports["buffer_plan"]
    hls_explain = reports["generated_hls_explanation"]
    vivado_bd = reports["vivado_bd_validation"]

    build_stages = _first_nonempty(manifest.get("build_stages"), package.get("build_stages"), {})
    if not isinstance(build_stages, Mapping):
        build_stages = {}

    buffers = buffer_plan.get("buffers") if isinstance(buffer_plan.get("buffers"), list) else []
    memory_summary = hls_explain.get("memory_summary") if isinstance(hls_explain.get("memory_summary"), Mapping) else {}
    communication_summary = hls_explain.get("communication_summary") if isinstance(hls_explain.get("communication_summary"), Mapping) else {}

    row: dict[str, Any] = {field: "" for field in MASTER_RESULT_FIELDS}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "design_id": root.name,
            "out_dir": str(root),
            "model": _first_nonempty(profile.get("graph_name"), Path(str(profile.get("model_path") or manifest.get("model_path") or "")).stem),
            "model_path": _first_nonempty(profile.get("model_path"), manifest.get("model_path")),
            "dataset": _first_nonempty(manifest.get("dataset"), _nested(profile, ["dataset"])),
            "board": _first_nonempty(package.get("board"), board_fit.get("board"), manifest.get("board")),
            "part": _first_nonempty(board_fit.get("part"), package.get("part"), manifest.get("part")),
            "mode": _first_nonempty(package.get("pipeline_mode"), manifest.get("pipeline_mode"), profile.get("pipeline_mode")),
            "precision": _first_nonempty(_knob_value(knob, "precision"), _nested(hls_explain, ["decisions", "precision"])),
            "memory_strategy": _first_nonempty(
                _knob_value(knob, "memory", "storage"),
                memory_summary.get("weight_storage"),
                memory_summary.get("storage"),
                training.get("weight_storage"),
            ),
            "data_movement": _first_nonempty(
                runtime_sequence.get("memory_semantics_mode"),
                communication_summary.get("interface"),
                communication_summary.get("data_movement"),
                training.get("movement_policy"),
            ),
            "parallel_factor": _first_nonempty(_knob_value(knob, "parallel"), timing.get("predicted_parallel_macs")),
            "pipeline_policy": _first_nonempty(_knob_value(knob, "pipeline"), _nested(timing, ["architecture_schedule_model", "pipeline_policy"])),
            "tiling": _first_nonempty(_knob_value(knob, "tiling", "tile"), memory_summary.get("tiling")),
            "training_optimizer": training.get("optimizer_type", ""),
            "training_loss": training.get("loss_type", ""),
            "build_cpp": bool(build_stages.get("cpp", False)),
            "build_hls_project": bool(build_stages.get("hls_project", False)),
            "build_hls_synthesis": bool(build_stages.get("hls_synthesis", False)),
            "build_vivado_project": bool(build_stages.get("vivado_project", False)),
            "build_vivado_implementation": bool(build_stages.get("vivado_implementation", False)),
            "build_bitstream": bool(build_stages.get("bitstream", False)),
            "runtime_package_status": _status(package, "missing"),
            "estimated_lut": _find_number(resource.get("totals", resource), "lut", "luts", "estimated_lut"),
            "estimated_ff": _find_number(resource.get("totals", resource), "ff", "ffs", "flipflop", "estimated_ff"),
            "estimated_bram": _find_number(resource.get("totals", resource), "bram", "brams", "estimated_bram"),
            "estimated_bram18": _find_number(resource.get("totals", resource), "bram18", "bram_18k", "estimated_bram18"),
            "estimated_dsp": _find_number(resource.get("totals", resource), "dsp", "dsps", "estimated_dsp"),
            "estimated_uram": _find_number(resource.get("totals", resource), "uram", "urams", "estimated_uram"),
            "estimated_fmax_mhz": _find_number(timing, "estimated_fmax_mhz", "predicted_fmax_mhz", "fmax_mhz"),
            "estimated_clock_mhz": _as_number(timing.get("clock_mhz")),
            "estimated_latency_ms": _as_number(timing.get("predicted_latency_ms")),
            "estimated_cycles": _as_number(timing.get("predicted_cycles")),
            "estimated_throughput_fps": _as_number(timing.get("predicted_throughput_fps")),
            "estimated_parallel_macs": _as_number(timing.get("predicted_parallel_macs")),
            "estimated_memory_bytes": _find_number(resource, "memory_bytes", "estimated_memory_bytes", "total_memory_bytes"),
            "estimated_weight_bytes": _first_nonempty(profile.get("parameter_bytes"), _find_number(resource, "weight_bytes", "parameter_bytes")),
            "estimated_activation_bytes": _find_number(resource, "activation_bytes", "estimated_activation_bytes"),
            "estimated_gradient_bytes": _find_number(resource, "gradient_bytes", "estimated_gradient_bytes"),
            "estimated_optimizer_state_bytes": _find_number(resource, "optimizer_state_bytes", "estimated_optimizer_state_bytes"),
            "board_fit_status": board_fit.get("status", ""),
            "board_fit_limiting_dimension": board_fit.get("limiting_dimension", ""),
            "board_fit_vivado_allowed": board_fit.get("vivado_implementation_allowed", ""),
            "board_fit_bitstream_allowed": board_fit.get("bitstream_allowed", ""),
            "generated_cpp_status": _status(reports["generated_cpp_validation"], "missing"),
            "movement_validation_status": _status(movement, "missing"),
            "movement_validation_passed": movement.get("passed", ""),
            "runtime_sequence_commands": _join_sequence(runtime_sequence.get("sequence")),
            "runtime_buffer_count": len(buffers),
            "hls_status": _first_nonempty(parsed_hls.get("hls_status"), _status(hls, "not_run")),
            "hls_lut": _first_nonempty(parsed_hls.get("hls_lut"), _find_number(hls, "lut", "luts")),
            "hls_ff": _first_nonempty(parsed_hls.get("hls_ff"), _find_number(hls, "ff", "ffs")),
            "hls_bram": _first_nonempty(parsed_hls.get("hls_bram"), _find_number(hls, "bram", "bram18")),
            "hls_dsp": _first_nonempty(parsed_hls.get("hls_dsp"), _find_number(hls, "dsp", "dsps")),
            "hls_latency_min": _first_nonempty(parsed_hls.get("hls_latency_min"), _find_number(hls, "latency_min", "min_latency", "latencymin")),
            "hls_latency_max": _first_nonempty(parsed_hls.get("hls_latency_max"), _find_number(hls, "latency_max", "max_latency", "latencymax")),
            "hls_ii": _first_nonempty(parsed_hls.get("hls_ii"), _find_number(hls, "ii", "interval", "initiation_interval")),
            "hls_clock_period_ns": _first_nonempty(parsed_hls.get("hls_clock_period_ns"), _find_number(hls, "clock_period_ns", "target_clock_period_ns")),
            "vivado_project_status": _status(vivado_bd, "not_requested"),
            "vivado_implementation_status": _first_nonempty(parsed_vivado.get("vivado_implementation_status"), _status(vivado, "not_run")),
            "vivado_lut": _first_nonempty(parsed_vivado.get("vivado_lut"), _find_number(vivado, "lut", "luts")),
            "vivado_ff": _first_nonempty(parsed_vivado.get("vivado_ff"), _find_number(vivado, "ff", "ffs")),
            "vivado_bram": _first_nonempty(parsed_vivado.get("vivado_bram"), _find_number(vivado, "bram", "bram18")),
            "vivado_dsp": _first_nonempty(parsed_vivado.get("vivado_dsp"), _find_number(vivado, "dsp", "dsps")),
            "vivado_wns": _first_nonempty(parsed_vivado.get("vivado_wns"), _find_number(vivado, "wns")),
            "vivado_tns": _first_nonempty(parsed_vivado.get("vivado_tns"), _find_number(vivado, "tns")),
            "vivado_power_w": _first_nonempty(parsed_vivado.get("vivado_power_w"), _find_number(vivado, "power_w", "total_power_w", "total_on_chip_power_w")),
            "bitstream_status": _first_nonempty(vivado.get("bitstream_status"), deployment.get("bitstream_status"), "not_run"),
            "deployment_package_status": _status(deployment, "not_run"),
            "runtime_status": _status(runtime, "not_run"),
            "runtime_board": _first_nonempty(runtime.get("board"), runtime.get("runtime_board")),
            "runtime_sample_count": _find_number(runtime, "sample_count", "samples"),
            "runtime_latency_ms_mean": _find_number(runtime, "latency_ms_mean", "mean_latency_ms"),
            "runtime_latency_ms_p50": _find_number(runtime, "latency_ms_p50", "p50_latency_ms"),
            "runtime_latency_ms_p95": _find_number(runtime, "latency_ms_p95", "p95_latency_ms"),
            "runtime_throughput": _find_number(runtime, "throughput", "throughput_fps", "samples_per_sec"),
            "runtime_accuracy": _find_number(runtime, "accuracy", "runtime_accuracy"),
            "runtime_loss_before": _find_number(runtime, "loss_before"),
            "runtime_loss_after": _find_number(runtime, "loss_after"),
            "runtime_training_step_ms": _find_number(runtime, "training_step_ms", "runtime_training_step_ms"),
        }
    )

    required: list[str] = []
    if _needs_validation_status(row["hls_status"]):
        required.append("hls_synthesis")
    if _needs_validation_status(row["vivado_implementation_status"]):
        required.append("vivado_implementation")
    if _needs_validation_status(row["runtime_status"]):
        required.append("board_runtime")
    row["required_validation"] = ";".join(required)
    row["support_status"] = _support_status(row)
    return row


def _support_status(row: Mapping[str, Any]) -> str:
    if _has_result_status(row.get("runtime_status")):
        return "runtime_result_available"
    if _has_result_status(row.get("vivado_implementation_status")):
        return "vivado_implementation_available"
    if _has_result_status(row.get("hls_status")):
        return "hls_synthesis_available"
    if row.get("generated_cpp_status") == "passed" and row.get("movement_validation_status") == "passed":
        return "static_validation_passed"
    if row.get("generated_cpp_status") == "passed":
        return "artifact_generated"
    return "incomplete"


def build_master_results(out_dirs: Iterable[str | Path]) -> dict[str, Any]:
    rows = [build_master_result_row(path) for path in out_dirs]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "passed",
        "summary": {
            "rows": len(rows),
            "with_hls_synthesis_result": sum(1 for r in rows if _has_result_status(r.get("hls_status"))),
            "with_vivado_implementation_result": sum(1 for r in rows if _has_result_status(r.get("vivado_implementation_status"))),
            "with_vivado_project_result": sum(1 for r in rows if _has_result_status(r.get("vivado_project_status"))),
            "with_runtime_result": sum(1 for r in rows if _has_result_status(r.get("runtime_status"))),
            "with_static_validation": sum(
                1
                for r in rows
                if r.get("generated_cpp_status") == "passed"
                and r.get("movement_validation_status") == "passed"
            ),
        },
        "schema_fields": MASTER_RESULT_FIELDS,
        "rows": rows,
    }


def write_master_results(
    out_dirs: Iterable[str | Path],
    *,
    output_json: str | Path,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    schema_json: str | Path | None = None,
    schema_md: str | Path | None = None,
) -> dict[str, Any]:
    results = build_master_results(out_dirs)
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if output_csv:
        _write_csv(Path(output_csv), results["rows"])
    if output_md:
        _write_md(Path(output_md), results["rows"])
    if schema_json:
        write_schema_json(schema_json)
    if schema_md:
        write_schema_md(schema_md)
    return results


def write_schema_json(path: str | Path) -> Path:
    schema = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "paper_master_result_schema",
        "fields": [
            {"name": field, "type": "number_or_blank" if field in _NUMERIC_FIELDS else "string_or_bool_or_blank"}
            for field in MASTER_RESULT_FIELDS
        ],
        "status_values": {
            "support_status": [
                "incomplete",
                "artifact_generated",
                "static_validation_passed",
                "hls_synthesis_available",
                "vivado_implementation_available",
                "runtime_result_available",
            ]
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return out


def write_schema_md(path: str | Path) -> Path:
    lines = ["# FPGAI Paper Master Result Schema", "", "| Field | Type |", "|---|---|"]
    for field in MASTER_RESULT_FIELDS:
        typ = "number or blank" if field in _NUMERIC_FIELDS else "string/bool or blank"
        lines.append(f"| `{field}` | {typ} |")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MASTER_RESULT_FIELDS})


def _write_md(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview_fields = [
        "design_id",
        "model",
        "board",
        "mode",
        "estimated_lut",
        "estimated_dsp",
        "estimated_latency_ms",
        "hls_status",
        "vivado_implementation_status",
        "runtime_status",
        "support_status",
    ]
    lines = ["# FPGAI Paper Master Results", "", f"Rows: {len(rows)}", "", "| " + " | ".join(preview_fields) + " |", "|" + "|".join(["---"] * len(preview_fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in preview_fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FPGAI paper master results from existing compile reports.")
    parser.add_argument("out_dirs", nargs="*", help="FPGAI compile output directories.")
    parser.add_argument("--output-json", default="paper_results/master_results.json")
    parser.add_argument("--output-csv", default="paper_results/master_results.csv")
    parser.add_argument("--output-md", default="paper_results/master_results.md")
    parser.add_argument("--schema-json", default="paper_results/schema/master_result_schema.json")
    parser.add_argument("--schema-md", default="paper_results/schema/master_result_schema.md")
    args = parser.parse_args(argv)
    if not args.out_dirs:
        parser.error("at least one compile output directory is required")
    results = write_master_results(
        args.out_dirs,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_md=args.output_md,
        schema_json=args.schema_json,
        schema_md=args.schema_md,
    )
    print(json.dumps({"artifact_kind": ARTIFACT_KIND, "status": results["status"], "summary": results["summary"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
