from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from fpgai.analysis.architecture_schedule_model import (
    estimate_architecture_schedules,
)


FLOAT_BYTES = 4


def _get(
    data: Mapping[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]

    return current


def _positive_float(
    value: Any,
    default: float,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result) or result <= 0:
        return default

    return result


def _nonnegative_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result) or result < 0:
        return default

    return result


def _ceil_div(
    value: int,
    divisor: int,
) -> int:
    if value <= 0:
        return 0

    divisor = max(1, divisor)
    return (value + divisor - 1) // divisor


def _dimensions(
    layer: Mapping[str, Any],
) -> Mapping[str, Any]:
    architecture = layer.get("architecture", {})

    if isinstance(architecture, Mapping):
        dimensions = architecture.get(
            "dimensions",
            layer.get("dimensions", {}),
        )
    else:
        dimensions = layer.get("dimensions", {})

    if isinstance(dimensions, Mapping):
        return dimensions

    return {}


def _input_elements(
    layer: Mapping[str, Any],
) -> int:
    dimensions = _dimensions(layer)
    value = int(
        dimensions.get("input_elements", 0) or 0
    )

    if value > 0:
        return value

    return _ceil_div(
        int(
            layer.get(
                "activation_bytes_in",
                0,
            )
            or 0
        ),
        FLOAT_BYTES,
    )


def _output_elements(
    layer: Mapping[str, Any],
) -> int:
    dimensions = _dimensions(layer)
    value = int(
        dimensions.get("output_elements", 0) or 0
    )

    if value > 0:
        return value

    return _ceil_div(
        int(
            layer.get(
                "activation_bytes_out",
                0,
            )
            or 0
        ),
        FLOAT_BYTES,
    )


def _clock_mhz(
    resource_estimate: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> float:
    architecture = resource_estimate.get(
        "architecture",
        {},
    )

    if isinstance(architecture, Mapping):
        value = architecture.get("clock_mhz")

        if value is not None:
            return _positive_float(
                value,
                200.0,
            )

    clocks = _get(
        raw_cfg,
        "targets.platform.clocks",
        [],
    )

    if isinstance(clocks, Sequence) and clocks:
        first_clock = clocks[0]

        if isinstance(first_clock, Mapping):
            return _positive_float(
                first_clock.get("target_mhz"),
                200.0,
            )

    return 200.0


def _execution_mode(
    resource_estimate: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> str:
    architecture = resource_estimate.get(
        "architecture",
        {},
    )

    if isinstance(architecture, Mapping):
        configured = architecture.get(
            "execution_mode"
        )

        if configured:
            normalized = str(
                configured
            ).strip().lower()

            if normalized in {
                "dataflow",
                "streaming",
            }:
                return "dataflow"

            return "sequential"

    configured = _get(
        raw_cfg,
        "hls.execution_mode",
        _get(
            raw_cfg,
            "optimization.execution_mode",
            "sequential",
        ),
    )

    if str(configured).strip().lower() in {
        "dataflow",
        "streaming",
    }:
        return "dataflow"

    return "sequential"


def _input_output_words(
    layers: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    if not layers:
        return 0, 0

    first_layer = layers[0]
    last_layer = layers[-1]

    input_words = _ceil_div(
        int(
            first_layer.get(
                "activation_bytes_in",
                0,
            )
            or 0
        ),
        FLOAT_BYTES,
    )
    output_words = _ceil_div(
        int(
            last_layer.get(
                "activation_bytes_out",
                0,
            )
            or 0
        ),
        FLOAT_BYTES,
    )

    if input_words <= 0:
        input_words = _input_elements(
            first_layer
        )

    if output_words <= 0:
        output_words = _output_elements(
            last_layer
        )

    return input_words, output_words


def _interface_configuration(
    raw_cfg: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "input_words_per_cycle": _positive_float(
            _get(
                raw_cfg,
                "analysis.design_space.estimator."
                "input_words_per_cycle",
                1.0,
            ),
            1.0,
        ),
        "output_words_per_cycle": _positive_float(
            _get(
                raw_cfg,
                "analysis.design_space.estimator."
                "output_words_per_cycle",
                1.0,
            ),
            1.0,
        ),
        "input_startup_cycles": _nonnegative_float(
            _get(
                raw_cfg,
                "analysis.design_space.estimator."
                "input_startup_cycles",
                3.0,
            ),
            3.0,
        ),
        "output_startup_cycles": _nonnegative_float(
            _get(
                raw_cfg,
                "analysis.design_space.estimator."
                "output_startup_cycles",
                3.0,
            ),
            3.0,
        ),
    }


def _control_cycles(
    layer_count: int,
    execution_mode: str,
    raw_cfg: Mapping[str, Any],
) -> float:
    entry_cycles = _nonnegative_float(
        _get(
            raw_cfg,
            "analysis.design_space.estimator."
            "kernel_entry_cycles",
            4.0,
        ),
        4.0,
    )
    call_cycles_per_layer = _nonnegative_float(
        _get(
            raw_cfg,
            "analysis.design_space.estimator."
            "call_cycles_per_layer",
            2.0,
        ),
        2.0,
    )
    exit_cycles = _nonnegative_float(
        _get(
            raw_cfg,
            "analysis.design_space.estimator."
            "kernel_exit_cycles",
            2.0,
        ),
        2.0,
    )

    if execution_mode == "dataflow":
        dataflow_start_cycles = _nonnegative_float(
            _get(
                raw_cfg,
                "analysis.design_space.estimator."
                "dataflow_start_cycles",
                2.0,
            ),
            2.0,
        )

        return (
            entry_cycles
            + dataflow_start_cycles
            + exit_cycles
        )

    return (
        entry_cycles
        + layer_count * call_cycles_per_layer
        + exit_cycles
    )


def _explicit_calibration(
    raw_cfg: Mapping[str, Any],
) -> dict[str, float | bool]:
    configured = _get(
        raw_cfg,
        "analysis.design_space.calibration.performance",
        {},
    )

    if not isinstance(configured, Mapping) or not configured:
        return {
            "enabled": False,
            "fixed_cycles": 0.0,
            "cycle_scale": 1.0,
            "input_words_per_cycle": 1.0,
            "output_words_per_cycle": 1.0,
        }

    explicit_enabled = configured.get("enabled")
    enabled = (
        bool(explicit_enabled)
        if explicit_enabled is not None
        else True
    )

    return {
        "enabled": enabled,
        "fixed_cycles": _nonnegative_float(
            configured.get(
                "fixed_cycles",
                0.0,
            )
        ),
        "cycle_scale": _positive_float(
            configured.get(
                "cycle_scale",
                1.0,
            ),
            1.0,
        ),
        "input_words_per_cycle": _positive_float(
            configured.get(
                "input_words_per_cycle",
                1.0,
            ),
            1.0,
        ),
        "output_words_per_cycle": _positive_float(
            configured.get(
                "output_words_per_cycle",
                1.0,
            ),
            1.0,
        ),
    }


def _schedule_body_cycles(
    schedule: Mapping[str, Any],
) -> float:
    predicted = float(
        schedule.get(
            "predicted_cycles",
            0.0,
        )
        or 0.0
    )
    fill = float(
        schedule.get(
            "pipeline_fill_cycles",
            0.0,
        )
        or 0.0
    )

    return max(
        0.0,
        predicted - fill,
    )


def _aggregate_compute_cycles(
    schedules: Sequence[Mapping[str, Any]],
    execution_mode: str,
) -> tuple[float, float, float]:
    if not schedules:
        return 0.0, 0.0, 0.0

    pipeline_fill_cycles = sum(
        float(
            schedule.get(
                "pipeline_fill_cycles",
                0.0,
            )
            or 0.0
        )
        for schedule in schedules
    )

    sequential_cycles = sum(
        float(
            schedule.get(
                "predicted_cycles",
                0.0,
            )
            or 0.0
        )
        for schedule in schedules
    )

    if execution_mode != "dataflow":
        return (
            sequential_cycles,
            pipeline_fill_cycles,
            sequential_cycles,
        )

    # For a streamed pipeline, all stages fill once. Steady-state
    # throughput is then limited by the slowest stage.
    slowest_stage_cycles = max(
        _schedule_body_cycles(schedule)
        for schedule in schedules
    )
    dataflow_cycles = (
        pipeline_fill_cycles
        + slowest_stage_cycles
    )

    return (
        dataflow_cycles,
        pipeline_fill_cycles,
        sequential_cycles,
    )


def estimate_performance(
    *,
    resource_estimate: dict[str, Any],
    raw_cfg: dict[str, Any],
) -> dict[str, Any]:
    layers = resource_estimate.get(
        "layers",
        [],
    )

    if not isinstance(layers, list):
        layers = []

    schedules = estimate_architecture_schedules(
        layers,
        raw_cfg,
    )

    layer_cycle_rows: list[dict[str, Any]] = []

    for layer, schedule in zip(
        layers,
        schedules,
    ):
        layer_cycle_rows.append(
            {
                "layer_index": layer.get(
                    "layer_index"
                ),
                "layer_name": layer.get(
                    "layer_name"
                ),
                "op_type": layer.get(
                    "op_type"
                ),
                "input_elements": _input_elements(
                    layer
                ),
                "output_elements": _output_elements(
                    layer
                ),
                "pe": int(
                    layer.get("pe", 1) or 1
                ),
                "simd": int(
                    layer.get("simd", 1) or 1
                ),
                **schedule,
            }
        )

    execution_mode = _execution_mode(
        resource_estimate,
        raw_cfg,
    )

    (
        compute_cycles,
        pipeline_fill_cycles,
        sequential_compute_cycles,
    ) = _aggregate_compute_cycles(
        layer_cycle_rows,
        execution_mode,
    )

    input_words, output_words = (
        _input_output_words(layers)
    )
    interface = _interface_configuration(
        raw_cfg
    )
    calibration = _explicit_calibration(
        raw_cfg
    )

    if calibration["enabled"]:
        input_words_per_cycle = float(
            calibration[
                "input_words_per_cycle"
            ]
        )
        output_words_per_cycle = float(
            calibration[
                "output_words_per_cycle"
            ]
        )
    else:
        input_words_per_cycle = float(
            interface[
                "input_words_per_cycle"
            ]
        )
        output_words_per_cycle = float(
            interface[
                "output_words_per_cycle"
            ]
        )

    input_transfer_cycles = (
        interface["input_startup_cycles"]
        + input_words
        / input_words_per_cycle
    )
    output_transfer_cycles = (
        interface["output_startup_cycles"]
        + output_words
        / output_words_per_cycle
    )

    if execution_mode == "dataflow":
        # Input and output movement overlap with the streamed compute
        # pipeline. The slowest path determines steady-state duration.
        transfer_cycles = max(
            input_transfer_cycles,
            output_transfer_cycles,
        )
        overlapped_core_cycles = max(
            compute_cycles,
            transfer_cycles,
        )
    else:
        transfer_cycles = (
            input_transfer_cycles
            + output_transfer_cycles
        )
        overlapped_core_cycles = (
            compute_cycles
            + transfer_cycles
        )

    control_cycles = _control_cycles(
        len(layers),
        execution_mode,
        raw_cfg,
    )

    analytical_cycles = (
        overlapped_core_cycles
        + control_cycles
    )

    fixed_cycles = (
        float(
            calibration["fixed_cycles"]
        )
        if calibration["enabled"]
        else 0.0
    )
    unscaled_cycles = (
        analytical_cycles
        + fixed_cycles
    )
    cycle_scale = (
        float(
            calibration["cycle_scale"]
        )
        if calibration["enabled"]
        else 1.0
    )
    total_cycles = (
        unscaled_cycles
        * cycle_scale
    )

    clock_mhz = _clock_mhz(
        resource_estimate,
        raw_cfg,
    )
    latency_ms = (
        total_cycles
        / (clock_mhz * 1_000.0)
    )
    throughput_fps = (
        1_000.0 / latency_ms
        if latency_ms > 0.0
        else 0.0
    )

    cpu_baseline_ms = _positive_float(
        _get(
            raw_cfg,
            "analysis.design_space.performance."
            "baseline_cpu_latency_ms",
            1.0,
        ),
        1.0,
    )
    speedup_vs_cpu = (
        cpu_baseline_ms / latency_ms
        if latency_ms > 0.0
        else 0.0
    )

    totals = resource_estimate.get(
        "totals",
        {},
    )
    total_macs = (
        int(
            totals.get(
                "total_macs",
                0,
            )
            or 0
        )
        if isinstance(totals, Mapping)
        else 0
    )
    effective_parallel_macs = (
        total_macs / compute_cycles
        if compute_cycles > 0.0
        else 0.0
    )

    return {
        "performance_model": (
            "operator_schedule_calibrated_v1"
        ),
        "analytical_performance_model": (
            "operator_execution_schedule_v2"
        ),
        "architecture_schedule_model": (
            "policy_aware_hls_schedule_v1"
        ),
        "estimation_mode": (
            "analytical_with_explicit_calibration"
            if calibration["enabled"]
            else "analytical"
        ),
        "execution_mode": execution_mode,
        "clock_mhz": float(clock_mhz),
        "predicted_parallel_macs": float(
            effective_parallel_macs
        ),
        "predicted_compute_cycles": float(
            compute_cycles
        ),
        "predicted_sequential_compute_cycles": float(
            sequential_compute_cycles
        ),
        "predicted_pipeline_fill_cycles": float(
            pipeline_fill_cycles
        ),
        "predicted_input_transfer_cycles": float(
            input_transfer_cycles
        ),
        "predicted_output_transfer_cycles": float(
            output_transfer_cycles
        ),
        "predicted_transfer_cycles": float(
            transfer_cycles
        ),
        "predicted_control_cycles": float(
            control_cycles
        ),
        "predicted_analytical_cycles": float(
            analytical_cycles
        ),
        "predicted_fixed_cycles": float(
            fixed_cycles
        ),
        "predicted_unscaled_cycles": float(
            unscaled_cycles
        ),
        "predicted_cycles": float(
            total_cycles
        ),
        "predicted_latency_ms": float(
            latency_ms
        ),
        "predicted_throughput_fps": float(
            throughput_fps
        ),
        "predicted_speedup_vs_cpu": float(
            speedup_vs_cpu
        ),
        "input_words": int(input_words),
        "output_words": int(output_words),
        "interface": interface,
        "calibration": calibration,
        "layer_cycles": layer_cycle_rows,
    }