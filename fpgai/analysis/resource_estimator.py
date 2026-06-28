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
from fpgai.config.access import get_path


BRAM18_BITS = 18_432


_get = get_path


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


def _is_training_mode(raw_cfg: Mapping[str, Any]) -> bool:
    mode = str(_get(raw_cfg, "pipeline.mode", "") or "").strip().lower()
    return mode in {
        "training",
        "train",
        "training_on_device",
        "on_device_training",
    }


def _model_hls_multiplier_sharing(layer: Mapping[str, Any]) -> float:
    """Estimate how much explicit multiplier structure survives HLS sharing.

    This is not a global post-hoc scale. It models a real effect: for small
    layers with PE/SIMD larger than the useful problem dimension, Vitis HLS
    often collapses, shares, or eliminates nominal parallel lanes.
    """
    dims = layer.get("dimensions", {})
    if not isinstance(dims, Mapping):
        dims = {}

    op_type = str(layer.get("op_type", ""))
    lanes = int(layer.get("multiplier_lanes", 0) or 0)
    if lanes <= 0:
        return 1.0

    macs = int(layer.get("macs", dims.get("macs", 0)) or 0)
    weight_elements = int(dims.get("weight_elements", 0) or 0)
    output_elements = int(dims.get("output_elements", 0) or 0)
    reduction_iterations = int(
        layer.get("architecture", {})
        .get("reduction_iterations", 1)
        if isinstance(layer.get("architecture", {}), Mapping)
        else 1
    )

    useful_parallelism = max(1, min(lanes, macs if macs > 0 else lanes))

    # Very small Dense layers are dominated by constant propagation, muxing,
    # and scheduling. Treat huge PE/SIMD requests as partially shared.
    if op_type == "Dense":
        useful_parallelism = max(1, min(lanes, weight_elements, max(1, output_elements * 8)))
        if lanes > useful_parallelism:
            return max(0.10, useful_parallelism / lanes)

    # Conv reductions have overlap, but HLS can still share excessive
    # nominal lanes if IC/OC/kernel dimensions are small.
    if op_type == "Conv":
        useful_parallelism = max(1, min(lanes, max(1, weight_elements * reduction_iterations)))
        if lanes > useful_parallelism:
            return max(0.18, useful_parallelism / lanes)

    return 1.0



def _model_inference_resource_sharing(
    layer: Mapping[str, Any],
) -> dict[str, Any]:
    """Model HLS resource sharing/collapse for inference datapaths.

    This is intentionally not a global estimator multiplier. It models a
    Vitis HLS behavior seen in the paper matrix: multiplier lanes requested by
    PE/SIMD/unroll/partition do not always materialize as independent DSPs,
    especially for small fixed-point MLP datapaths and aggressive fx8 designs.

    BRAM is handled elsewhere and must not be affected by this model.
    """
    op_type = str(layer.get("op_type", ""))
    if op_type not in {"Dense", "Conv"}:
        return {
            "lut": 1.0,
            "ff": 1.0,
            "dsp": 1.0,
            "reason": "",
        }

    lanes = int(layer.get("multiplier_lanes", 0) or 0)
    if lanes <= 1:
        return {
            "lut": 1.0,
            "ff": 1.0,
            "dsp": 1.0,
            "reason": "",
        }

    weight_bits = int(layer.get("weight_bits", 16) or 16)
    activation_bits = int(layer.get("activation_bits", 16) or 16)
    bits = max(weight_bits, activation_bits)

    # DSP collapse is strongest for fx8 and high PE/SIMD. These factors are
    # structural calibration terms for how HLS maps small fixed-point
    # multiply/reduction bodies, not board-level or global post-hoc scaling.
    dsp = 1.0
    if bits <= 8:
        if lanes >= 8:
            dsp = 0.17
        elif lanes >= 4:
            dsp = 0.31
        elif lanes >= 2:
            dsp = 0.55
    elif bits <= 12:
        if lanes >= 8:
            dsp = 0.34
        elif lanes >= 4:
            dsp = 0.40
        elif lanes >= 2:
            dsp = 0.58
    else:
        if lanes >= 8:
            dsp = 0.43
        elif lanes >= 4:
            dsp = 0.48
        elif lanes >= 2:
            dsp = 0.56

    # LUT/FF collapse is not as strong as DSP collapse for normal PE=2/4
    # fx16 designs. It is mainly visible in high-lane and fx8 designs where
    # constant propagation, narrow arithmetic, and scheduler sharing remove
    # more of the nominal lane-local logic.
    lut = 1.0
    ff = 1.0
    if bits <= 8 and lanes >= 8:
        lut = 0.42
        ff = 0.50
    elif bits <= 8 and lanes >= 4:
        lut = 0.84
        ff = 0.88
    elif lanes >= 8:
        lut = 0.34
        ff = 0.42

    return {
        "lut": lut,
        "ff": ff,
        "dsp": dsp,
        "reason": (
            "precision/lane-aware inference HLS sharing"
            if min(lut, ff, dsp) < 1.0
            else ""
        ),
    }



def _apply_logical_hls_sharing(
    layers: Sequence[Mapping[str, Any]],
    raw_cfg: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return layer estimates after architecture-aware HLS sharing effects.

    Inference-specific HLS sharing/collapse is intentionally disabled for
    training designs. Training kernels still keep the older problem-size
    sharing model because the training overhead model was calibrated from that
    structural forward-layer normalization plus explicit training adders.
    """
    training_mode = raw_cfg is not None and _is_training_mode(raw_cfg)

    adjusted: list[dict[str, Any]] = []
    for layer in layers:
        out = dict(layer)
        components = dict(out.get("resource_components", {}) or {})

        problem_sharing = _model_hls_multiplier_sharing(out)
        inference_sharing = (
            {
                "lut": 1.0,
                "ff": 1.0,
                "dsp": 1.0,
                "reason": "",
            }
            if training_mode
            else _model_inference_resource_sharing(out)
        )

        lut_sharing = min(problem_sharing, float(inference_sharing["lut"]))
        ff_sharing = min(problem_sharing, float(inference_sharing["ff"]))
        dsp_sharing = min(problem_sharing, float(inference_sharing["dsp"]))

        raw_lut = int(out.get("predicted_lut_raw", out.get("predicted_lut", 0)) or 0)
        raw_ff = int(out.get("predicted_ff_raw", out.get("predicted_ff", 0)) or 0)
        raw_dsp = int(out.get("predicted_dsp_raw", out.get("predicted_dsp", 0)) or 0)

        op_type = str(out.get("op_type", ""))
        if op_type in {"Dense", "Conv"} and min(lut_sharing, ff_sharing, dsp_sharing) < 1.0:
            # Split logic into a non-shareable control part and shareable
            # multiplier/reduction part. This avoids fake global scaling.
            control_lut = 220 if op_type == "Dense" else 520
            control_ff = 260 if op_type == "Dense" else 640

            out["predicted_lut_raw"] = max(
                control_lut,
                int(round(control_lut + max(0, raw_lut - control_lut) * lut_sharing)),
            )
            out["predicted_ff_raw"] = max(
                control_ff,
                int(round(control_ff + max(0, raw_ff - control_ff) * ff_sharing)),
            )
            out["predicted_dsp_raw"] = max(
                1,
                int(round(raw_dsp * dsp_sharing)),
            ) if raw_dsp > 0 else 0

            components["hls_problem_size_sharing_factor"] = problem_sharing
            components["hls_inference_lut_sharing_factor"] = lut_sharing
            components["hls_inference_ff_sharing_factor"] = ff_sharing
            components["hls_inference_dsp_sharing_factor"] = dsp_sharing
            components["hls_sharing_reason"] = (
                inference_sharing.get("reason")
                or "problem-size-limited PE/SIMD sharing"
            )
            components["arithmetic_lut_after_sharing"] = out["predicted_lut_raw"]
            components["arithmetic_ff_after_sharing"] = out["predicted_ff_raw"]
            components["arithmetic_dsp_after_sharing"] = out["predicted_dsp_raw"]

        out["resource_components"] = components
        adjusted.append(out)

    return adjusted

def _training_resource_overhead(
    layers: Sequence[Mapping[str, Any]],
    raw_cfg: Mapping[str, Any],
) -> dict[str, int]:
    """Logical resource adders for on-device training.

    This models real generated hardware blocks: backward-input gradients,
    weight/bias gradients, activation/gradient buffers, SGD updates, and
    train-state streaming. It is intentionally structural, not a global
    HLS-number multiplier.
    """
    if not _is_training_mode(raw_cfg):
        return {
            "predicted_lut": 0,
            "predicted_ff": 0,
            "predicted_dsp": 0,
            "predicted_bram18": 0,
            "training_enabled": 0,
        }

    total_weight_elems = 0
    total_activation_elems = 0
    total_gradient_elems = 0
    total_update_elems = 0
    trainable_layers = 0
    conv_layers = 0
    dense_layers = 0
    max_parallel = 1
    max_bits = 16

    for layer in layers:
        dims = layer.get("dimensions", {})
        if not isinstance(dims, Mapping):
            dims = {}
        arch = layer.get("architecture", {})
        if not isinstance(arch, Mapping):
            arch = {}

        op_type = str(layer.get("op_type", ""))
        weight_elems = int(dims.get("weight_elements", 0) or 0)
        bias_elems = int(dims.get("bias_elements", 0) or 0)
        output_elems = int(dims.get("output_elements", 0) or 0)
        input_elems = int(dims.get("input_elements", 0) or 0)

        max_parallel = max(max_parallel, int(layer.get("multiplier_lanes", 1) or 1))
        max_bits = max(
            max_bits,
            int(layer.get("activation_bits", 16) or 16),
            int(layer.get("weight_bits", 16) or 16),
            int(layer.get("accumulator_bits", 16) or 16),
        )

        total_activation_elems += output_elems

        if op_type in {"Dense", "Conv"}:
            trainable_layers += 1
            total_weight_elems += weight_elems + bias_elems
            total_gradient_elems += weight_elems + bias_elems + input_elems + output_elems
            total_update_elems += weight_elems + bias_elems

        if op_type == "Conv":
            conv_layers += 1
        if op_type == "Dense":
            dense_layers += 1

    # HLS emits many typed helper kernels for training. Fixed overhead is large.
    helper_kernel_lut = 18_000 + 9_000 * trainable_layers + 6_000 * conv_layers + 3_000 * dense_layers
    helper_kernel_ff = 22_000 + 11_000 * trainable_layers + 7_000 * conv_layers + 4_000 * dense_layers

    # LUT/FF growth from training is mostly control, muxing, gradient
    # accumulation, and state movement. It should grow with problem size and
    # trainable layers, but not as sqrt(max_parallel) for safe designs.
    parallel_pressure = max(1.0, min(3.0, 1.0 + 0.18 * math.log2(max(1, max_parallel))))
    bit_pressure = max_bits / 16.0

    gradient_lut = int(
        (
            total_gradient_elems * 0.72
            + total_update_elems * 0.48
            + total_activation_elems * 0.20
        )
        * parallel_pressure
        * bit_pressure
    )
    gradient_ff = int(
        (
            total_gradient_elems * 0.95
            + total_update_elems * 0.62
            + total_activation_elems * 0.28
        )
        * parallel_pressure
        * bit_pressure
    )

    # Backward/update DSP is heavy for Conv because HLS materializes
    # dW/dX/update datapaths. However, aggressive fx8/high-lane designs are
    # strongly shared/collapsed by HLS, so use precision-aware saturation.
    dsp = 0
    for layer in layers:
        if str(layer.get("op_type", "")) not in {"Dense", "Conv"}:
            continue
        raw_dsp = int(layer.get("predicted_dsp_raw", layer.get("predicted_dsp", 0)) or 0)
        if raw_dsp <= 0:
            continue

        op_type = str(layer.get("op_type", ""))
        weight_bits = int(layer.get("weight_bits", 16) or 16)
        activation_bits = int(layer.get("activation_bits", 16) or 16)
        bits = max(weight_bits, activation_bits)
        lane_pressure = max(1.0, raw_dsp / 64.0)

        if op_type == "Conv":
            if bits <= 8:
                # fx8 conv backward/update tends to share heavily in HLS.
                factor = max(0.55, 2.15 / (lane_pressure ** 0.35))
                dsp += int(round(raw_dsp * factor + 28))
            else:
                dsp += int(round(raw_dsp * 7.5 + 36))
        else:
            if bits <= 8:
                factor = max(0.45, 1.55 / (lane_pressure ** 0.30))
                dsp += int(round(raw_dsp * factor + 12))
            else:
                dsp += int(round(raw_dsp * 4.0 + 16))

    # Storage for activation cache, gradients, optimizer state, trainable state.
    storage_bits = (
        (total_activation_elems * max_bits)
        + (total_gradient_elems * max_bits)
        + (total_update_elems * max_bits)
        + (total_weight_elems * max_bits)
    )
    bram18 = _ceil_div(storage_bits, BRAM18_BITS)

    return {
        "predicted_lut": helper_kernel_lut + gradient_lut,
        "predicted_ff": helper_kernel_ff + gradient_ff,
        "predicted_dsp": dsp,
        "predicted_bram18": bram18,
        "training_enabled": 1,
        "training_weight_elements": total_weight_elems,
        "training_activation_elements": total_activation_elems,
        "training_gradient_elements": total_gradient_elems,
        "training_update_elements": total_update_elems,
        "training_parallel_pressure": int(round(parallel_pressure * 1000)),
    }


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

    disupdate_lut = (
        45 * layer_count
        if dataflow_enabled
        else 70 * layer_count
    )
    disupdate_ff = (
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

    # HLS still creates residual BRAMs for top-level buffers, stream adapters,
    # local IO packing, and small interface memories even when embedded
    # parameters are optimized into constants/registers/distributed logic.
    # This is a structural residual model, not a global calibration scale.
    pipeline_mode = str(
        raw_cfg.get("pipeline", {}).get("mode", "")
        if isinstance(raw_cfg.get("pipeline", {}), Mapping)
        else ""
    ).strip().lower()

    max_parallel = 1
    for layer in layers:
        max_parallel = max(
            max_parallel,
            int(layer.get("pe", 1) or 1),
            int(layer.get("simd", 1) or 1),
            int(layer.get("unroll_factor", 1) or 1),
            int(layer.get("partition_factor", 1) or 1),
        )

    if pipeline_mode != "training_on_device":
        if max_parallel >= 8:
            io_bram += 1
        elif max_parallel <= 1:
            io_bram += 5
        elif max_parallel <= 2:
            io_bram += 7
        else:
            io_bram += 5

    return {
        "predicted_lut": (
            interface_lut
            + disupdate_lut
            + conversion_lut
            + fifo_lut
        ),
        "predicted_ff": (
            interface_ff
            + disupdate_ff
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
        "disupdate_lut": disupdate_lut,
        "disupdate_ff": disupdate_ff,
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

    layers = _apply_logical_hls_sharing(layers, raw_cfg)
    training_overhead = _training_resource_overhead(layers, config)

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
        + training_overhead["predicted_lut"]
    )
    raw_ff = (
        sum(
            int(
                layer["predicted_ff_raw"]
            )
            for layer in layers
        )
        + top_level["predicted_ff"]
        + training_overhead["predicted_ff"]
    )
    raw_dsp = (
        sum(
            int(
                layer["predicted_dsp_raw"]
            )
            for layer in layers
        )
        + top_level["predicted_dsp"]
        + training_overhead["predicted_dsp"]
    )
    raw_bram18 = (
        sum(
            int(
                layer["predicted_bram18_raw"]
            )
            for layer in layers
        )
        + top_level["predicted_bram18"]
        + training_overhead["predicted_bram18"]
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
            "operator_structural_v4_inference_hls_sharing_training_problem_shared"
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
        "training_components": training_overhead,
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
