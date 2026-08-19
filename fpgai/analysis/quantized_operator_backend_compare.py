from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(baseline: Any, candidate: Any) -> dict[str, float | None]:
    base = _number(baseline)
    cand = _number(candidate)
    if base is None or cand is None:
        return {"absolute": None, "percent": None}
    absolute = cand - base
    percent = None if base == 0 else (absolute / base) * 100.0
    return {"absolute": absolute, "percent": percent}


def _max_buffer_depth(physical: Mapping[str, Any]) -> int | None:
    depths: list[int] = []
    for edge in physical.get("edges", ()) or ():
        if not isinstance(edge, Mapping):
            continue
        value = edge.get("elastic_buffer_depth_words")
        try:
            if value is not None:
                depths.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(depths) if depths else None


def _partition_add_contract(numeric: Mapping[str, Any]) -> dict[str, Any] | None:
    partition = numeric.get("partition")
    if not isinstance(partition, Mapping):
        return None
    add = partition.get("add")
    if not isinstance(add, Mapping):
        return None
    lowering = add.get("lowering") if isinstance(add.get("lowering"), Mapping) else {}
    left_q = add.get("left_quantization") if isinstance(add.get("left_quantization"), Mapping) else {}
    right_q = add.get("right_quantization") if isinstance(add.get("right_quantization"), Mapping) else {}
    output_q = add.get("output_quantization") if isinstance(add.get("output_quantization"), Mapping) else {}
    return {
        "left_scale": left_q.get("scale"),
        "left_zero_point": left_q.get("zero_point"),
        "right_scale": right_q.get("scale"),
        "right_zero_point": right_q.get("zero_point"),
        "output_scale": output_q.get("scale"),
        "output_zero_point": output_q.get("zero_point"),
        "left_multiplier": lowering.get("left_multiplier"),
        "left_shift": lowering.get("left_shift"),
        "right_multiplier": lowering.get("right_multiplier"),
        "right_shift": lowering.get("right_shift"),
        "rounding_mode": lowering.get("rounding_mode"),
        "saturation_mode": lowering.get("saturation_mode"),
        "qmin": lowering.get("qmin"),
        "qmax": lowering.get("qmax"),
    }


def _variant(
    *,
    name: str,
    backend: str,
    operator: str,
    characterization: Mapping[str, Any],
    tool_result: Mapping[str, Any],
    physical: Mapping[str, Any],
    numeric: Mapping[str, Any],
    add_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resources = characterization.get("resources") if isinstance(characterization.get("resources"), Mapping) else {}
    clock = characterization.get("clock") if isinstance(characterization.get("clock"), Mapping) else {}
    power = characterization.get("power") if isinstance(characterization.get("power"), Mapping) else {}
    numeric_pass = bool(tool_result.get("mixed_language_simulation_passed"))
    sim_metrics = tool_result.get("simulation_metrics") if isinstance(tool_result.get("simulation_metrics"), Mapping) else {}
    return {
        "name": name,
        "backend": backend,
        "operator": operator,
        "precision": "int8 activations / int8 weights / int32 accumulators",
        "quantization": dict(add_contract or {}),
        "buffering": {
            "max_elastic_buffer_depth_words": _max_buffer_depth(physical),
            "reported_edges": [
                {
                    "tensor": edge.get("tensor"),
                    "elastic_buffer_depth_words": edge.get("elastic_buffer_depth_words"),
                }
                for edge in physical.get("edges", ()) or ()
                if isinstance(edge, Mapping) and edge.get("elastic_buffer_depth_words") is not None
            ],
        },
        "latency": {
            "first_output_latency_cycles": sim_metrics.get("first_output_latency_cycles"),
            "packet_completion_latency_cycles": sim_metrics.get("packet_completion_latency_cycles"),
            "post_input_drain_cycles": sim_metrics.get("post_input_drain_cycles"),
            "input_accept_span_cycles": sim_metrics.get("input_accept_span_cycles"),
            "output_accept_span_cycles": sim_metrics.get("output_accept_span_cycles"),
            "mean_output_interbeat_cycles": sim_metrics.get("mean_output_interbeat_cycles"),
            "measurement_status": (
                "cycle_accurate_behavioral_xsim"
                if sim_metrics.get("first_output_latency_cycles") is not None
                else "not_measured"
            ),
            "initiation_interval": sim_metrics.get("initiation_interval"),
            "initiation_interval_status": sim_metrics.get("initiation_interval_status", "not_measured"),
        },
        "resources": {
            "lut": resources.get("lut"),
            "ff": resources.get("ff"),
            "dsp": resources.get("dsp"),
            "bram": resources.get("bram"),
            "uram": resources.get("uram"),
        },
        "clock": {
            "target_mhz": clock.get("target_mhz"),
            "target_period_ns": clock.get("target_period_ns"),
            "wns_ns": clock.get("wns_ns"),
            "tns_ns": clock.get("tns_ns"),
            "timing_met": clock.get("timing_met"),
        },
        "power": {
            "total_power_w": power.get("total_power_w"),
            "dynamic_power_w": power.get("dynamic_power_w"),
            "static_power_w": power.get("static_power_w"),
        },
        "numeric_exactness": {
            "exact_integer_xsim": numeric_pass,
            "comparison": physical.get("numeric_validation", {}).get("comparison")
            if isinstance(physical.get("numeric_validation"), Mapping)
            else None,
        },
        "validation_level": tool_result.get("validation_level"),
        "implementation_status": tool_result.get("status"),
        "artifacts": characterization.get("artifacts"),
        "source_numeric_schema": numeric.get("schema"),
    }


def build_quantized_add_backend_comparison(
    *,
    hls_characterization_path: str | Path,
    hls_tool_result_path: str | Path,
    hls_physical_path: str | Path,
    hls_numeric_path: str | Path,
    vhdl_characterization_path: str | Path,
    vhdl_tool_result_path: str | Path,
    vhdl_physical_path: str | Path,
    vhdl_numeric_path: str | Path,
) -> dict[str, Any]:
    hls_char = _load_json(hls_characterization_path)
    hls_tool = _load_json(hls_tool_result_path)
    hls_physical = _load_json(hls_physical_path)
    hls_numeric = _load_json(hls_numeric_path)
    vhdl_char = _load_json(vhdl_characterization_path)
    vhdl_tool = _load_json(vhdl_tool_result_path)
    vhdl_physical = _load_json(vhdl_physical_path)
    vhdl_numeric = _load_json(vhdl_numeric_path)

    add_contract = _partition_add_contract(vhdl_numeric)
    hls_variant = _variant(
        name="hls_residual_add",
        backend="vitis_hls",
        operator="Add",
        characterization=hls_char,
        tool_result=hls_tool,
        physical=hls_physical,
        numeric=hls_numeric,
        add_contract=add_contract,
    )
    vhdl_variant = _variant(
        name="vhdl_residual_add",
        backend="vhdl",
        operator="Add",
        characterization=vhdl_char,
        tool_result=vhdl_tool,
        physical=vhdl_physical,
        numeric=vhdl_numeric,
        add_contract=add_contract,
    )

    resource_delta = {
        key: _delta(hls_variant["resources"].get(key), vhdl_variant["resources"].get(key))
        for key in ("lut", "ff", "dsp", "bram", "uram")
    }
    power_delta = {
        key: _delta(hls_variant["power"].get(key), vhdl_variant["power"].get(key))
        for key in ("total_power_w", "dynamic_power_w", "static_power_w")
    }
    latency_delta = {
        key: _delta(hls_variant["latency"].get(key), vhdl_variant["latency"].get(key))
        for key in (
            "first_output_latency_cycles",
            "packet_completion_latency_cycles",
            "post_input_drain_cycles",
            "output_accept_span_cycles",
            "mean_output_interbeat_cycles",
        )
    }
    latency_available = (
        hls_variant["latency"].get("first_output_latency_cycles") is not None
        and vhdl_variant["latency"].get("first_output_latency_cycles") is not None
    )

    return {
        "schema": "fpgai.quantized-add-backend-comparison/v1",
        "status": "passed" if hls_tool.get("status") == "passed" and vhdl_tool.get("status") == "passed" else "incomplete",
        "scope": "whole_quantized_residual_cnn_with_add_and_terminal_relu_backend_partition_choice",
        "comparison_unit": "whole_design_routed_implementation",
        "operator_under_study": "quantized residual Add within the current Add+terminal-ReLU VHDL partition",
        "experimental_design": {
            "isolated_add_backend_effect": False,
            "changed_between_variants": [
                "residual Add backend",
                "terminal ReLU backend",
                "residual skip buffering required by the VHDL partition",
            ],
            "next_control_required": "VHDL Add with HLS terminal ReLU, plus the complementary HLS Add/VHDL ReLU control, for an isolated 2x2 backend-effect analysis",
        },
        "variants": {
            "hls": hls_variant,
            "vhdl": vhdl_variant,
        },
        "delta_vhdl_minus_hls": {
            "resources": resource_delta,
            "power": power_delta,
            "latency": latency_delta,
            "wns_ns": _delta(hls_variant["clock"].get("wns_ns"), vhdl_variant["clock"].get("wns_ns")),
            "max_elastic_buffer_depth_words": _delta(
                hls_variant["buffering"].get("max_elastic_buffer_depth_words"),
                vhdl_variant["buffering"].get("max_elastic_buffer_depth_words"),
            ),
        },
        "comparability": {
            "same_model": True,
            "same_quantized_input_output_contract": True,
            "same_terminal_relu_backend": False,
            "isolated_add_backend_effect": False,
            "same_target_clock_required": hls_variant["clock"].get("target_mhz") == vhdl_variant["clock"].get("target_mhz"),
            "latency_comparison_available": latency_available,
            "latency_measurement_scope": "cycle-accurate behavioral XSim at the whole-DAG external ready/valid interface",
            "notes": [
                "Resource, timing, and power values are whole-design routed measurements; they are not isolated operator-only costs.",
                "The current VHDL variant changes both residual Add and terminal ReLU backends, so these deltas must not be published as the isolated cost of VHDL Add.",
                "The VHDL residual path includes explicit skip-branch elastic buffering required to avoid residual backpressure deadlock.",
                "WNS is used only to report whether the requested clock is met; this comparison does not claim measured Fmax.",
                "Latency is measured in behavioral XSim cycles from accepted external input/output handshakes for one packet; it is a whole-DAG latency measurement, not isolated Add latency.",
                "True initiation interval is not claimed from the single-packet testbench; mean output inter-beat spacing is reported separately as a throughput proxy.",
            ],
        },
        "usage": {"platform_scope": "research", "production_path": "morfics"},
    }


def write_quantized_add_backend_comparison(payload: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "quantized_add_backend_comparison.json"
    md_path = root / "quantized_add_backend_comparison.md"
    json_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    variants = payload["variants"]
    hls = variants["hls"]
    vhdl = variants["vhdl"]
    delta = payload["delta_vhdl_minus_hls"]
    lines = [
        "# Quantized residual partition backend comparison",
        "",
        "Whole-design routed comparison of the same PTQ residual CNN. In the current VHDL variant both residual Add and terminal ReLU are partitioned to VHDL, so this artifact is not an isolated Add-only causal comparison.",
        "",
        "| Metric | HLS Add | VHDL Add | VHDL − HLS |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("lut", "LUT"), ("ff", "FF"), ("dsp", "DSP"), ("bram", "BRAM"), ("uram", "URAM")):
        d = delta["resources"][key]["absolute"]
        lines.append(f"| {label} | {hls['resources'].get(key)} | {vhdl['resources'].get(key)} | {d} |")
    lines.extend([
        f"| First-output latency (cycles) | {hls['latency'].get('first_output_latency_cycles')} | {vhdl['latency'].get('first_output_latency_cycles')} | {delta['latency']['first_output_latency_cycles']['absolute']} |",
        f"| Packet-completion latency (cycles) | {hls['latency'].get('packet_completion_latency_cycles')} | {vhdl['latency'].get('packet_completion_latency_cycles')} | {delta['latency']['packet_completion_latency_cycles']['absolute']} |",
        f"| Mean output inter-beat spacing (cycles) | {hls['latency'].get('mean_output_interbeat_cycles')} | {vhdl['latency'].get('mean_output_interbeat_cycles')} | {delta['latency']['mean_output_interbeat_cycles']['absolute']} |",
        f"| WNS (ns) | {hls['clock'].get('wns_ns')} | {vhdl['clock'].get('wns_ns')} | {delta['wns_ns']['absolute']} |",
        f"| Total power (W) | {hls['power'].get('total_power_w')} | {vhdl['power'].get('total_power_w')} | {delta['power']['total_power_w']['absolute']} |",
        f"| Dynamic power (W) | {hls['power'].get('dynamic_power_w')} | {vhdl['power'].get('dynamic_power_w')} | {delta['power']['dynamic_power_w']['absolute']} |",
        f"| Static power (W) | {hls['power'].get('static_power_w')} | {vhdl['power'].get('static_power_w')} | {delta['power']['static_power_w']['absolute']} |",
        f"| Max elastic buffer depth (words) | {hls['buffering'].get('max_elastic_buffer_depth_words')} | {vhdl['buffering'].get('max_elastic_buffer_depth_words')} | {delta['max_elastic_buffer_depth_words']['absolute']} |",
        "",
        "## Validation",
        "",
        f"- HLS variant: `{hls.get('validation_level')}`, exact integer XSim: `{hls['numeric_exactness'].get('exact_integer_xsim')}`",
        f"- VHDL variant: `{vhdl.get('validation_level')}`, exact integer XSim: `{vhdl['numeric_exactness'].get('exact_integer_xsim')}`",
        "",
        "## Interpretation limits",
        "",
    ])
    lines.extend(f"- {note}" for note in payload["comparability"]["notes"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


__all__ = ["build_quantized_add_backend_comparison", "write_quantized_add_backend_comparison"]
