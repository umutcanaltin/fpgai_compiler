from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from fpgai.analysis.architecture_resource_model import (
    estimate_architecture_resources,
)
from fpgai.analysis.hls_architecture import (
    HLSArchitecture,
    build_hls_architecture,
)


BRAM18_BITS = 18_432


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
    default: float = 1.0,
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


def _banked_bram18(
    elements: int,
    bits_per_element: int,
    banks: int = 1,
) -> int:
    if elements <= 0 or bits_per_element <= 0:
        return 0

    banks = min(
        max(1, banks),
        elements,
    )
    elements_per_bank = _ceil_div(
        elements,
        banks,
    )
    bits_per_bank = (
        elements_per_bank
        * bits_per_element
    )

    return banks * _ceil_div(
        bits_per_bank,
        BRAM18_BITS,
    )


def _explicit_calibration(
    raw_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    configured = _get(
        raw_cfg,
        "analysis.design_space.calibration.resources",
        {},
    )

    if not isinstance(configured, Mapping) or not configured:
        return {
            "enabled": False,
            "fixed_lut": 0.0,
            "fixed_ff": 0.0,
            "fixed_dsp": 0.0,
            "fixed_bram18": 0.0,
            "lut_scale": 1.0,
            "ff_scale": 1.0,
            "dsp_scale": 1.0,
            "bram18_scale": 1.0,
        }

    explicit_enabled = configured.get("enabled")

    enabled = (
        bool(explicit_enabled)
        if explicit_enabled is not None
        else True
    )

    return {
        "enabled": enabled,
        "fixed_lut": _nonnegative_float(
            configured.get("fixed_lut", 0.0)
        ),
        "fixed_ff": _nonnegative_float(
            configured.get("fixed_ff", 0.0)
        ),
        "fixed_dsp": _nonnegative_float(
            configured.get("fixed_dsp", 0.0)
        ),
        "fixed_bram18": _nonnegative_float(
            configured.get("fixed_bram18", 0.0)
        ),
        "lut_scale": _positive_float(
            configured.get("lut_scale", 1.0)
        ),
        "ff_scale": _positive_float(
            configured.get("ff_scale", 1.0)
        ),
        "dsp_scale": _positive_float(
            configured.get("dsp_scale", 1.0)
        ),
        "bram18_scale": _positive_float(
            configured.get("bram18_scale", 1.0)
        ),
    }


def _apply_calibration(
    *,
    raw_value: int,
    fixed_value: float,
    scale: float,
    enabled: bool,
) -> int:
    if not enabled:
        return max(0, int(raw_value))

    return max(
        0,
        int(
            round(
                (raw_value + fixed_value)
                * scale
            )
        ),
    )


def _descriptor_bytes(
    descriptor: Any,
) -> dict[str, int]:
    return {
        "parameter_bytes": int(
            getattr(
                descriptor,
                "param_bytes",
                0,
            )
            or 0
        ),
        "activation_bytes_in": int(
            getattr(
                descriptor,
                "activation_bytes_in",
                0,
            )
            or 0
        ),
        "activation_bytes_out": int(
            getattr(
                descriptor,
                "activation_bytes_out",
                0,
            )
            or 0
        ),
    }


def _enrich_layer_result(
    result: dict[str, Any],
    descriptor: Any,
    index: int,
) -> dict[str, Any]:
    enriched = dict(result)
    byte_counts = _descriptor_bytes(
        descriptor
    )

    dimensions = dict(
        enriched.get("dimensions", {})
    )

    descriptor_macs = int(
        getattr(descriptor, "macs", 0) or 0
    )
    dimension_macs = int(
        dimensions.get("macs", 0) or 0
    )
    macs = (
        dimension_macs
        if dimension_macs > 0
        else descriptor_macs
    )

    activation_bits = int(
        enriched.get("activation_bits", 16)
    )
    weight_bits = int(
        enriched.get("weight_bits", 16)
    )
    accumulator_bits = int(
        enriched.get("accumulator_bits", 24)
    )

    enriched.update(
        {
            "layer_index": index,
            "layer_name": str(
                getattr(
                    descriptor,
                    "node_name",
                    f"layer_{index}",
                )
            ),
            "activation_bits": activation_bits,
            "weight_bits": weight_bits,
            "accumulator_bits": (
                accumulator_bits
            ),
            "act_bits": activation_bits,
            "wgt_bits": weight_bits,
            "acc_bits": accumulator_bits,
            "macs": macs,
            "parameter_bytes": (
                byte_counts["parameter_bytes"]
            ),
            "param_bytes": (
                byte_counts["parameter_bytes"]
            ),
            "activation_bytes_in": (
                byte_counts["activation_bytes_in"]
            ),
            "activation_bytes_out": (
                byte_counts["activation_bytes_out"]
            ),
            "attrs": dict(
                getattr(
                    descriptor,
                    "attrs",
                    {},
                )
                or {}
            ),
            "dimensions": dimensions,
        }
    )

    return enriched


def _top_level_resources(
    layers: Sequence[Mapping[str, Any]],
    architecture: HLSArchitecture,
    raw_cfg: Mapping[str, Any],
) -> dict[str, int]:
    layer_count = len(layers)
    stream_edges = max(
        0,
        layer_count + 1,
    )

    input_elements = 0
    output_elements = 0
    activation_bits = 16

    if layers:
        first_layer = layers[0]
        last_layer = layers[-1]

        first_dimensions = first_layer.get(
            "dimensions",
            {},
        )
        last_dimensions = last_layer.get(
            "dimensions",
            {},
        )

        if isinstance(
            first_dimensions,
            Mapping,
        ):
            input_elements = int(
                first_dimensions.get(
                    "input_elements",
                    0,
                )
                or 0
            )

        if isinstance(
            last_dimensions,
            Mapping,
        ):
            output_elements = int(
                last_dimensions.get(
                    "output_elements",
                    0,
                )
                or 0
            )

        activation_bits = int(
            first_layer.get(
                "activation_bits",
                16,
            )
            or 16
        )

    dataflow_enabled = (
        str(
            architecture.execution_mode
        ).strip().lower()
        == "dataflow"
    )

    interface_lut = (
        260
        + 45 * stream_edges
    )
    interface_ff = (
        340
        + 64 * stream_edges
    )

    dispatch_lut = (
        45 * layer_count
        if dataflow_enabled
        else 70 * layer_count
    )
    dispatch_ff = (
        60 * layer_count
        if dataflow_enabled
        else 90 * layer_count
    )

    conversion_lut = (
        _ceil_div(
            input_elements,
            64,
        )
        + _ceil_div(
            output_elements,
            64,
        )
    )
    conversion_ff = (
        _ceil_div(
            input_elements
            * activation_bits,
            128,
        )
        + _ceil_div(
            output_elements
            * activation_bits,
            128,
        )
    )

    fifo_lut = 0
    fifo_ff = 0
    fifo_bram = 0

    if dataflow_enabled:
        fifo_depth = max(
            2,
            int(
                _get(
                    raw_cfg,
                    "hls.dataflow.fifo_depth",
                    2,
                )
            ),
        )
        fifo_lut = (
            stream_edges
            * (
                20
                + _ceil_div(
                    activation_bits,
                    4,
                )
            )
        )
        fifo_ff = (
            stream_edges
            * (
                24
                + activation_bits
            )
        )

        if fifo_depth * activation_bits > 1024:
            fifo_bram = (
                stream_edges
                * _banked_bram18(
                    fifo_depth,
                    activation_bits,
                )
            )

    io_bram = 0

    if bool(
        _get(
            raw_cfg,
            "analysis.design_space.estimator."
            "buffer_top_io",
            False,
        )
    ):
        io_bram = (
            _banked_bram18(
                input_elements,
                activation_bits,
            )
            + _banked_bram18(
                output_elements,
                activation_bits,
            )
        )

    return {
        "predicted_lut": (
            interface_lut
            + dispatch_lut
            + conversion_lut
            + fifo_lut
        ),
        "predicted_ff": (
            interface_ff
            + dispatch_ff
            + conversion_ff
            + fifo_ff
        ),
        "predicted_dsp": 0,
        "predicted_bram18": (
            io_bram
            + fifo_bram
        ),
        "interface_lut": interface_lut,
        "interface_ff": interface_ff,
        "dispatch_lut": dispatch_lut,
        "dispatch_ff": dispatch_ff,
        "conversion_lut": conversion_lut,
        "conversion_ff": conversion_ff,
        "fifo_lut": fifo_lut,
        "fifo_ff": fifo_ff,
        "fifo_bram18": fifo_bram,
        "io_bram18": io_bram,
        "dataflow_enabled": int(
            dataflow_enabled
        ),
    }


def estimate_resources_from_descriptors(
    descriptors: Sequence[Any],
    raw_cfg: Mapping[str, Any],
    compile_plan: Any = None,
) -> dict[str, Any]:
    config = dict(raw_cfg or {})

    architecture = build_hls_architecture(
        descriptors,
        config,
        compile_plan,
    )

    architecture_layers = (
        estimate_architecture_resources(
            architecture,
            config,
        )
    )

    layers = [
        _enrich_layer_result(
            layer_result,
            descriptor,
            index,
        )
        for index, (
            layer_result,
            descriptor,
        ) in enumerate(
            zip(
                architecture_layers,
                descriptors,
            )
        )
    ]

    top_level = _top_level_resources(
        layers,
        architecture,
        config,
    )
    calibration = _explicit_calibration(
        config
    )

    raw_lut = (
        sum(
            int(
                layer["predicted_lut_raw"]
            )
            for layer in layers
        )
        + top_level["predicted_lut"]
    )
    raw_ff = (
        sum(
            int(
                layer["predicted_ff_raw"]
            )
            for layer in layers
        )
        + top_level["predicted_ff"]
    )
    raw_dsp = (
        sum(
            int(
                layer["predicted_dsp_raw"]
            )
            for layer in layers
        )
        + top_level["predicted_dsp"]
    )
    raw_bram18 = (
        sum(
            int(
                layer["predicted_bram18_raw"]
            )
            for layer in layers
        )
        + top_level["predicted_bram18"]
    )

    predicted_lut = _apply_calibration(
        raw_value=raw_lut,
        fixed_value=calibration["fixed_lut"],
        scale=calibration["lut_scale"],
        enabled=calibration["enabled"],
    )
    predicted_ff = _apply_calibration(
        raw_value=raw_ff,
        fixed_value=calibration["fixed_ff"],
        scale=calibration["ff_scale"],
        enabled=calibration["enabled"],
    )
    predicted_dsp = _apply_calibration(
        raw_value=raw_dsp,
        fixed_value=calibration["fixed_dsp"],
        scale=calibration["dsp_scale"],
        enabled=calibration["enabled"],
    )
    predicted_bram18 = _apply_calibration(
        raw_value=raw_bram18,
        fixed_value=calibration[
            "fixed_bram18"
        ],
        scale=calibration["bram18_scale"],
        enabled=calibration["enabled"],
    )

    totals = {
        "predicted_lut": predicted_lut,
        "predicted_ff": predicted_ff,
        "predicted_dsp": predicted_dsp,
        "predicted_bram18": predicted_bram18,
        "predicted_lut_raw": int(
            raw_lut
        ),
        "predicted_ff_raw": int(
            raw_ff
        ),
        "predicted_dsp_raw": int(
            raw_dsp
        ),
        "predicted_bram18_raw": int(
            raw_bram18
        ),
        "total_macs": int(
            sum(
                int(
                    layer.get(
                        "macs",
                        0,
                    )
                    or 0
                )
                for layer in layers
            )
        ),
        "total_multiplier_lanes": int(
            sum(
                int(
                    layer.get(
                        "multiplier_lanes",
                        0,
                    )
                    or 0
                )
                for layer in layers
                if layer.get("op_type")
                in {"Dense", "Conv"}
            )
        ),
    }

    return {
        # Retained for compatibility with existing report readers.
        "model": "operator_aware_calibrated_v1",
        "analytical_model": (
            "operator_structural_v2"
        ),
        "architecture_model": (
            "policy_aware_hls_architecture_v1"
        ),
        "estimation_mode": (
            "analytical_with_explicit_calibration"
            if calibration["enabled"]
            else "analytical"
        ),
        "calibration": calibration,
        "architecture": (
            architecture.to_dict()
        ),
        "top_level": top_level,
        "totals": totals,
        "worst_lut_layer": (
            max(
                layers,
                key=lambda layer: int(
                    layer["predicted_lut"]
                ),
            )
            if layers
            else None
        ),
        "worst_dsp_layer": (
            max(
                layers,
                key=lambda layer: int(
                    layer["predicted_dsp"]
                ),
            )
            if layers
            else None
        ),
        "worst_bram_layer": (
            max(
                layers,
                key=lambda layer: int(
                    layer["predicted_bram18"]
                ),
            )
            if layers
            else None
        ),
        "layers": layers,
    }