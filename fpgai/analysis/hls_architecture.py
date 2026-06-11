from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from fpgai.config.access import get_path
from fpgai.numerics.precision_policy import (
    canonical_op_type,
    resolve_precision_for_op,
)


_get = get_path


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, default)


def _positive_float(value: Any, default: float = 200.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result) or result <= 0:
        return default

    return result


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0

    divisor = max(1, divisor)
    return (value + divisor - 1) // divisor


def _numel(shape: Sequence[int]) -> int:
    if not shape:
        return 0

    result = 1

    for value in shape:
        result *= int(value)

    return result


def _shape(
    descriptor: Any,
    attribute: str,
    index: int = 0,
) -> tuple[int, ...]:
    shapes = getattr(descriptor, attribute, []) or []

    if not isinstance(shapes, (list, tuple)):
        return ()

    if index < 0 or index >= len(shapes):
        return ()

    try:
        result = tuple(int(value) for value in shapes[index])
    except (TypeError, ValueError):
        return ()

    if len(result) > 1 and result[0] == 1:
        return result[1:]

    return result


def _plan_to_dict(plan: Any) -> dict[str, Any]:
    if isinstance(plan, Mapping):
        return dict(plan)

    if hasattr(plan, "to_dict"):
        value = plan.to_dict()

        if isinstance(value, Mapping):
            return dict(value)

    return {}


def _plan_map(compile_plan: Any) -> dict[str, dict[str, Any]]:
    if compile_plan is None:
        return {}

    if isinstance(compile_plan, Mapping):
        plans = compile_plan.get("layer_plans", [])
    else:
        plans = getattr(compile_plan, "layer_plans", [])

    result: dict[str, dict[str, Any]] = {}

    if not isinstance(plans, (list, tuple)):
        return result

    for plan in plans:
        value = _plan_to_dict(plan)
        node_name = str(value.get("node_name", ""))

        if node_name:
            result[node_name] = value

    return result


def _compile_plan_notes(compile_plan: Any) -> dict[str, Any]:
    if compile_plan is None:
        return {}

    if isinstance(compile_plan, Mapping):
        notes = compile_plan.get("notes", {})
    else:
        notes = getattr(compile_plan, "notes", {})

    if isinstance(notes, Mapping):
        return dict(notes)

    return {}


def _compile_plan_clock(compile_plan: Any) -> float | None:
    if compile_plan is None:
        return None

    if isinstance(compile_plan, Mapping):
        value = compile_plan.get("clock_mhz")
    else:
        value = getattr(compile_plan, "clock_mhz", None)

    if value is None:
        return None

    return _positive_float(value)


def _configured_clock(raw_cfg: Mapping[str, Any]) -> float:
    clocks = _get(raw_cfg, "targets.platform.clocks", [])

    if isinstance(clocks, (list, tuple)) and clocks:
        first_clock = clocks[0]

        if isinstance(first_clock, Mapping):
            return _positive_float(
                first_clock.get("target_mhz"),
                200.0,
            )

    return 200.0


@dataclass(frozen=True)
class LayerArchitecture:
    name: str
    op_type: str
    dimensions: dict[str, int]
    pipeline_scope: str
    pipeline_ii: int
    unroll: dict[str, int]
    reduction_iterations: int
    pipeline_overlap: int
    explicit_lanes: int
    effective_lanes: int
    arithmetic: dict[str, Any]
    memory: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HLSArchitecture:
    policy: str
    execution_mode: str
    clock_mhz: float
    layers: list[LayerArchitecture]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dense_dimensions(descriptor: Any) -> dict[str, int]:
    input_shape = _shape(descriptor, "input_shapes")
    output_shape = _shape(descriptor, "output_shapes")
    weight_shape = _shape(descriptor, "input_shapes", 1)
    attrs = dict(getattr(descriptor, "attrs", {}) or {})

    input_features = int(attrs.get("in_features", 0) or 0)
    output_features = int(attrs.get("out_features", 0) or 0)

    if input_features <= 0:
        input_features = _numel(input_shape)

    if output_features <= 0:
        output_features = _numel(output_shape)

    if len(weight_shape) == 2:
        if output_features <= 0:
            output_features = int(weight_shape[0])

        if input_features <= 0:
            input_features = int(weight_shape[1])

    return {
        "input_features": input_features,
        "output_features": output_features,
        "input_elements": input_features,
        "output_elements": output_features,
        "weight_elements": input_features * output_features,
        "bias_elements": output_features,
        "macs": input_features * output_features,
    }


def _conv_dimensions(descriptor: Any) -> dict[str, int]:
    input_shape = _shape(descriptor, "input_shapes")
    output_shape = _shape(descriptor, "output_shapes")
    weight_shape = _shape(descriptor, "input_shapes", 1)
    attrs = dict(getattr(descriptor, "attrs", {}) or {})

    input_channels = 0
    input_height = 0
    input_width = 0
    output_channels = 0
    output_height = 0
    output_width = 0

    if len(input_shape) == 3:
        input_channels, input_height, input_width = input_shape

    if len(output_shape) == 3:
        output_channels, output_height, output_width = output_shape

    kernel_height = 0
    kernel_width = 0

    if len(weight_shape) == 4:
        weight_output_channels = int(weight_shape[0])
        weight_input_channels = int(weight_shape[1])
        kernel_height = int(weight_shape[2])
        kernel_width = int(weight_shape[3])

        if output_channels <= 0:
            output_channels = weight_output_channels

        if input_channels <= 0:
            input_channels = weight_input_channels

    kernel_shape = attrs.get("kernel_shape", [])

    if (
        (kernel_height <= 0 or kernel_width <= 0)
        and isinstance(kernel_shape, (list, tuple))
        and len(kernel_shape) >= 2
    ):
        kernel_height = int(kernel_shape[0])
        kernel_width = int(kernel_shape[1])

    kernel_height = max(1, kernel_height)
    kernel_width = max(1, kernel_width)

    groups = _positive_int(attrs.get("group", 1))
    channels_per_group = _ceil_div(
        input_channels,
        groups,
    )
    kernel_elements = kernel_height * kernel_width
    input_elements = (
        input_channels
        * input_height
        * input_width
    )
    output_elements = (
        output_channels
        * output_height
        * output_width
    )
    weight_elements = (
        output_channels
        * channels_per_group
        * kernel_elements
    )

    return {
        "input_channels": input_channels,
        "input_height": input_height,
        "input_width": input_width,
        "output_channels": output_channels,
        "output_height": output_height,
        "output_width": output_width,
        "kernel_height": kernel_height,
        "kernel_width": kernel_width,
        "kernel_elements": kernel_elements,
        "groups": groups,
        "channels_per_group": channels_per_group,
        "input_elements": input_elements,
        "output_elements": output_elements,
        "weight_elements": weight_elements,
        "bias_elements": output_channels,
        "macs": (
            output_elements
            * channels_per_group
            * kernel_elements
        ),
    }


def _generic_dimensions(
    descriptor: Any,
) -> dict[str, int]:
    input_shape = _shape(descriptor, "input_shapes")
    output_shape = _shape(descriptor, "output_shapes")
    attrs = dict(getattr(descriptor, "attrs", {}) or {})

    kernel_shape = attrs.get("kernel_shape", [1, 1])
    kernel_height = 1
    kernel_width = 1

    if (
        isinstance(kernel_shape, (list, tuple))
        and len(kernel_shape) >= 2
    ):
        kernel_height = _positive_int(kernel_shape[0])
        kernel_width = _positive_int(kernel_shape[1])

    return {
        "input_elements": _numel(input_shape),
        "output_elements": _numel(output_shape),
        "kernel_height": kernel_height,
        "kernel_width": kernel_width,
        "kernel_elements": kernel_height * kernel_width,
        "weight_elements": 0,
        "bias_elements": 0,
        "macs": int(
            getattr(descriptor, "macs", 0) or 0
        ),
    }


def _dimensions(
    descriptor: Any,
    op_type: str,
) -> dict[str, int]:
    if op_type == "Dense":
        return _dense_dimensions(descriptor)

    if op_type == "Conv":
        return _conv_dimensions(descriptor)

    return _generic_dimensions(descriptor)


def _precision_bits(
    raw_cfg: dict[str, Any],
    descriptor: Any,
    index: int,
    name: str,
    op_type: str,
) -> dict[str, int]:
    op = SimpleNamespace(
        name=name,
        op_type=op_type,
        attrs=dict(
            getattr(descriptor, "attrs", {}) or {}
        ),
    )

    precision = resolve_precision_for_op(
        raw_cfg,
        op,
        index,
    ).specs

    return {
        role: int(spec["total_bits"])
        for role, spec in precision.items()
    }


def _plan_unroll(
    plan: Mapping[str, Any],
    key: str,
    fallback: Any,
) -> int:
    unroll = plan.get("unroll", {})

    if isinstance(unroll, Mapping) and key in unroll:
        return _positive_int(
            unroll[key],
            _positive_int(fallback),
        )

    return _positive_int(fallback)


def _pipeline_ii(
    plan: Mapping[str, Any],
    default_ii: int,
) -> int:
    value = plan.get("pipeline_ii")

    if value is None:
        return default_ii

    return _positive_int(value, default_ii)


def build_hls_architecture(
    descriptors: Sequence[Any],
    raw_cfg: Mapping[str, Any] | None,
    compile_plan: Any = None,
) -> HLSArchitecture:
    config = dict(raw_cfg or {})
    plans = _plan_map(compile_plan)
    plan_notes = _compile_plan_notes(compile_plan)

    global_pe = _positive_int(
        plan_notes.get(
            "parallel_pe",
            _get(
                config,
                "optimization.parallel.pe",
                1,
            ),
        )
    )
    global_simd = _positive_int(
        plan_notes.get(
            "parallel_simd",
            _get(
                config,
                "optimization.parallel.simd",
                1,
            ),
        )
    )
    global_unroll = _positive_int(
        plan_notes.get(
            "parallel_unroll_factor",
            _get(
                config,
                "optimization.parallel.unroll_factor",
                1,
            ),
        )
    )
    partition_factor = _positive_int(
        plan_notes.get(
            "parallel_partition_factor",
            _get(
                config,
                "optimization.parallel.partition_factor",
                1,
            ),
        )
    )
    default_pipeline_ii = _positive_int(
        _get(
            config,
            "hls.pipeline_ii",
            1,
        )
    )

    layers: list[LayerArchitecture] = []

    for index, descriptor in enumerate(descriptors):
        name = str(
            getattr(
                descriptor,
                "node_name",
                f"layer_{index}",
            )
        )
        op_type = canonical_op_type(
            str(
                getattr(
                    descriptor,
                    "op_type",
                    "Unknown",
                )
            )
        )
        plan = plans.get(name, {})
        dimensions = _dimensions(
            descriptor,
            op_type,
        )
        bits = _precision_bits(
            config,
            descriptor,
            index,
            name,
            op_type,
        )
        pipeline_ii = _pipeline_ii(
            plan,
            default_pipeline_ii,
        )

        unroll: dict[str, int]
        pipeline_scope: str
        reduction_iterations: int
        pipeline_overlap: int
        explicit_lanes: int
        input_banks: int
        output_banks: int
        weight_banks: int

        if op_type == "Conv":
            output_unroll = _plan_unroll(
                plan,
                "oc",
                _get(
                    config,
                    "hls.conv.oc_unroll",
                    global_pe,
                ),
            )
            input_unroll = _plan_unroll(
                plan,
                "ic",
                _get(
                    config,
                    "hls.conv.ic_unroll",
                    global_simd,
                ),
            )
            unroll = {
                "oc": output_unroll,
                "ic": input_unroll,
            }

            reduction_iterations = (
                _ceil_div(
                    dimensions["channels_per_group"],
                    input_unroll,
                )
                * dimensions["kernel_elements"]
            )

            pipeline_scope = "output_column"
            pipeline_overlap = max(
                1,
                _ceil_div(
                    reduction_iterations,
                    pipeline_ii,
                ),
            )
            explicit_lanes = (
                output_unroll
                * input_unroll
            )

            input_banks = max(
                partition_factor,
                input_unroll,
            )
            output_banks = max(
                partition_factor,
                output_unroll,
            )
            weight_banks = max(
                partition_factor,
                explicit_lanes,
            )

        elif op_type == "Dense":
            output_unroll = _plan_unroll(
                plan,
                "out",
                _get(
                    config,
                    "hls.dense.out_unroll",
                    global_pe,
                ),
            )
            input_unroll = _plan_unroll(
                plan,
                "in",
                _get(
                    config,
                    "hls.dense.in_unroll",
                    global_simd,
                ),
            )
            unroll = {
                "out": output_unroll,
                "in": input_unroll,
            }

            reduction_iterations = _ceil_div(
                dimensions["input_features"],
                input_unroll,
            )

            pipeline_scope = "input_base"
            pipeline_overlap = 1
            explicit_lanes = (
                output_unroll
                * input_unroll
            )

            input_banks = max(
                partition_factor,
                input_unroll,
            )
            output_banks = max(
                partition_factor,
                output_unroll,
            )
            weight_banks = max(
                partition_factor,
                explicit_lanes,
            )

        elif op_type in {"MaxPool", "AvgPool"}:
            unroll = {}

            reduction_iterations = max(
                1,
                dimensions["kernel_elements"],
            )
            pipeline_scope = "output_element"
            pipeline_overlap = max(
                1,
                _ceil_div(
                    reduction_iterations,
                    pipeline_ii,
                ),
            )
            explicit_lanes = 1

            input_banks = partition_factor
            output_banks = partition_factor
            weight_banks = 1

        else:
            element_unroll = _positive_int(
                _get(
                    config,
                    "hls.activation.unroll",
                    global_unroll,
                )
            )
            unroll = {
                "element": element_unroll,
            }

            reduction_iterations = (
                dimensions["output_elements"]
                if op_type == "Softmax"
                else 1
            )
            pipeline_scope = (
                "three_pass_element"
                if op_type == "Softmax"
                else "element"
            )
            pipeline_overlap = 1
            explicit_lanes = element_unroll

            input_banks = max(
                partition_factor,
                element_unroll,
            )
            output_banks = max(
                partition_factor,
                element_unroll,
            )
            weight_banks = 1

        effective_lanes = (
            explicit_lanes
            * pipeline_overlap
        )

        has_multiplier = op_type in {
            "Conv",
            "Dense",
            "LeakyRelu",
        }

        effective_multiplier_units = (
            effective_lanes
            if has_multiplier
            else 0
        )

        layers.append(
            LayerArchitecture(
                name=name,
                op_type=op_type,
                dimensions=dimensions,
                pipeline_scope=pipeline_scope,
                pipeline_ii=pipeline_ii,
                unroll=unroll,
                reduction_iterations=(
                    reduction_iterations
                ),
                pipeline_overlap=pipeline_overlap,
                explicit_lanes=explicit_lanes,
                effective_lanes=effective_lanes,
                arithmetic={
                    "activation_bits": bits["activation"],
                    "weight_bits": bits["weight"],
                    "bias_bits": bits["bias"],
                    "accumulator_bits": bits["accum"],
                    "output_bits": bits["activation"],
                    "multiply_left_bits": (
                        bits["accum"]
                        if has_multiplier
                        else 0
                    ),
                    "multiply_right_bits": (
                        bits["accum"]
                        if has_multiplier
                        else 0
                    ),
                    "effective_multiplier_units": (
                        effective_multiplier_units
                    ),
                },
                memory={
                    "partition_factor": (
                        partition_factor
                    ),
                    "input_banks": input_banks,
                    "output_banks": output_banks,
                    "weight_banks": weight_banks,
                    "weight_mode": plan.get(
                        "weight_mode",
                        "embedded",
                    ),
                    "activation_mode": plan.get(
                        "activation_mode",
                        "buffer",
                    ),
                    "buffering": plan.get(
                        "buffering",
                        "single",
                    ),
                },
            )
        )

    planned_clock = _compile_plan_clock(
        compile_plan
    )

    return HLSArchitecture(
        policy=str(
            plan_notes.get(
                "parallel_policy",
                _get(
                    config,
                    "optimization.parallel_policy",
                    "Balanced",
                ),
            )
        ),
        execution_mode=str(
            _get(
                config,
                "hls.execution_mode",
                "sequential",
            )
        ),
        clock_mhz=(
            planned_clock
            if planned_clock is not None
            else _configured_clock(config)
        ),
        layers=layers,
    )
