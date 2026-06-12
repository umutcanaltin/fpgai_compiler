from __future__ import annotations

from typing import Any, Mapping


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, dict):
            return result

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _nested_get(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, Mapping):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def emit_layer_architecture_comments(compile_plan: Any) -> str:
    """Emit readable HLS comments for each effective layer architecture."""
    plan = _to_dict(compile_plan)
    layers = getattr(compile_plan, "layer_plans", None)

    if layers is None:
        layers = plan.get("layer_plans", [])

    lines = [
        "// FPGAI effective per-layer architecture",
    ]

    if not layers:
        lines.append("//   no compile plan supplied")
        return "\n".join(lines) + "\n"

    for index, layer in enumerate(layers):
        layer_dict = _to_dict(layer)

        name = str(layer_dict.get("name", f"layer_{index}"))
        op_type = str(layer_dict.get("op_type", "unknown"))

        architecture = layer_dict.get("architecture", {})
        if not isinstance(architecture, Mapping):
            architecture = {}

        ii = _nested_get(
            architecture,
            "pipeline",
            "ii",
            default=layer_dict.get("pipeline_ii", "?"),
        )
        pe = _nested_get(
            architecture,
            "parallelism",
            "pe",
            default=_nested_get(
                architecture,
                "parallelism",
                "output_unroll",
                default=layer_dict.get("pe", "?"),
            ),
        )
        simd = _nested_get(
            architecture,
            "parallelism",
            "simd",
            default=_nested_get(
                architecture,
                "parallelism",
                "input_unroll",
                default=layer_dict.get("simd", "?"),
            ),
        )
        input_partition = _nested_get(
            architecture,
            "partitioning",
            "input",
            default=layer_dict.get("partition_factor", "?"),
        )
        output_partition = _nested_get(
            architecture,
            "partitioning",
            "output",
            default=layer_dict.get("partition_factor", "?"),
        )
        weight_partition = _nested_get(
            architecture,
            "partitioning",
            "weight",
            default=layer_dict.get("partition_factor", "?"),
        )
        tile = architecture.get("tiling", layer_dict.get("tile", {}))
        memory = architecture.get("memory", layer_dict.get("memory", {}))
        signature = layer_dict.get(
            "architecture_signature",
            layer_dict.get("signature", "?"),
        )

        lines.append(
            "//   "
            f"{index}: {name} "
            f"({op_type}) "
            f"ii={ii} "
            f"pe={pe} "
            f"simd={simd} "
            f"part_in={input_partition} "
            f"part_out={output_partition} "
            f"part_w={weight_partition} "
            f"tile={tile} "
            f"memory={memory} "
            f"sig={signature}"
        )

    return "\n".join(lines) + "\n"
