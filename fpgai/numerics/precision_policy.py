from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

from fpgai.config.access import get_path

PRECISION_ROLES = ("activation", "weight", "bias", "accum")

DEFAULT_PRECISION: Dict[str, Dict[str, Any]] = {
    "activation": {
        "type": "ap_fixed",
        "total_bits": 16,
        "int_bits": 6,
    },
    "weight": {
        "type": "ap_fixed",
        "total_bits": 16,
        "int_bits": 6,
    },
    "bias": {
        "type": "ap_fixed",
        "total_bits": 24,
        "int_bits": 10,
    },
    "accum": {
        "type": "ap_fixed",
        "total_bits": 24,
        "int_bits": 10,
    },
}

OP_TYPE_ALIASES = {
    "gemm": "Dense",
    "matmul": "Dense",
}


_cfg_get = get_path


def canonical_op_type(op_type: str) -> str:
    value = str(op_type).strip()
    lower = value.lower()

    if lower in {
        "linear",
        "torchlinear",
        "torch_linear",
    }:
        return "Dense"

    if lower == "dense":
        return "Dense"

    return OP_TYPE_ALIASES.get(
        lower,
        value,
    )


def normalize_precision_spec(
    spec: Mapping[str, Any] | None,
    *,
    fallback: Mapping[str, Any],
    path: str,
) -> Dict[str, Any]:
    source = dict(fallback)

    if spec is not None:
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"{path} must be a mapping"
            )

        source.update(spec)

    precision_type = str(
        source.get(
            "type",
            "ap_fixed",
        )
    )

    if precision_type != "ap_fixed":
        raise ValueError(
            f"{path}.type must be 'ap_fixed', "
            f"got {precision_type!r}"
        )

    try:
        raw_total_bits = source["total_bits"]
        raw_int_bits = source["int_bits"]
    except KeyError as exc:
        raise ValueError(
            f"{path} requires integer total_bits "
            "and int_bits"
        ) from exc

    if (
        type(raw_total_bits) is not int
        or type(raw_int_bits) is not int
    ):
        raise ValueError(
            f"{path} requires integer total_bits "
            "and int_bits"
        )

    total_bits = raw_total_bits
    int_bits = raw_int_bits

    if total_bits <= 0:
        raise ValueError(
            f"{path}.total_bits must be greater "
            "than zero"
        )

    if int_bits <= 0:
        raise ValueError(
            f"{path}.int_bits must be greater "
            "than zero"
        )

    if int_bits > total_bits:
        raise ValueError(
            f"{path}.int_bits ({int_bits}) cannot "
            f"exceed total_bits ({total_bits})"
        )

    return {
        "type": "ap_fixed",
        "total_bits": total_bits,
        "int_bits": int_bits,
    }


def _ceil_div_int(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("division denominator must be positive")
    return (int(a) + int(b) - 1) // int(b)


def precision_role_bits(policy: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    """Return total bit width for activation/weight/bias/accumulator roles."""
    out: Dict[str, int] = {}
    for role in PRECISION_ROLES:
        spec = policy.get(role, DEFAULT_PRECISION[role])
        out[role] = int(spec.get("total_bits", DEFAULT_PRECISION[role]["total_bits"]))
    return out


def values_per_word(value_bits: int, word_bits: int) -> int:
    """How many precision values fit in one transport/storage word."""
    value_bits = int(value_bits)
    word_bits = int(word_bits)
    if value_bits <= 0:
        raise ValueError("value_bits must be positive")
    if word_bits <= 0:
        raise ValueError("word_bits must be positive")
    return max(1, word_bits // value_bits)


def packed_word_count(element_count: int, value_bits: int, word_bits: int) -> int:
    """Number of packed words required to store/transfer element_count values."""
    return _ceil_div_int(int(element_count), values_per_word(value_bits, word_bits))


def packed_byte_count(element_count: int, value_bits: int, word_bits: int) -> int:
    """Byte count after packing values into word_bits words."""
    words = packed_word_count(element_count, value_bits, word_bits)
    return words * _ceil_div_int(word_bits, 8)


def raw_bit_count(element_count: int, value_bits: int) -> int:
    return int(element_count) * int(value_bits)


def raw_byte_count(element_count: int, value_bits: int) -> int:
    return _ceil_div_int(raw_bit_count(element_count, value_bits), 8)


def build_precision_layout(
    raw_cfg: Mapping[str, Any],
    *,
    input_elements: int = 0,
    output_elements: int = 0,
    weight_elements: int = 0,
    bias_elements: int = 0,
    activation_buffer_elements: int = 0,
    axis_word_bits: int | None = None,
    axi_word_bits: int | None = None,
) -> Dict[str, Any]:
    """Build one shared precision/storage/communication layout.

    This is the central truth used by codegen, runtime packing, estimators,
    and reports. Precision must affect not only compute types, but also
    activation/weight/bias storage and AXIS/AXI transfer sizes.
    """
    policy = default_precision_policy(raw_cfg)
    bits = precision_role_bits(policy)

    axis_word_bits = int(
        axis_word_bits
        if axis_word_bits is not None
        else _cfg_get(raw_cfg, "data_movement.ps_pl.axis_word_bits", 32)
    )
    axi_word_bits = int(
        axi_word_bits
        if axi_word_bits is not None
        else _cfg_get(raw_cfg, "data_movement.ps_pl.axi_word_bits", 128)
    )

    precision_mode = str(
        _cfg_get(
            raw_cfg,
            "numerics.precision_mode",
            _cfg_get(raw_cfg, "analysis.precision_sweep.selected_candidate", "custom"),
        )
    )

    act_bits = bits["activation"]
    weight_bits = bits["weight"]
    bias_bits = bits["bias"]
    accum_bits = bits["accum"]

    layout: Dict[str, Any] = {
        "precision_mode": precision_mode,
        "roles": {
            "activation": dict(policy["activation"]),
            "weight": dict(policy["weight"]),
            "bias": dict(policy["bias"]),
            "accum": dict(policy["accum"]),
        },
        "bits": {
            "activation": act_bits,
            "weight": weight_bits,
            "bias": bias_bits,
            "accum": accum_bits,
        },
        "word_bits": {
            "axis": axis_word_bits,
            "axi": axi_word_bits,
        },
        "pack_factors": {
            "activation_per_axis_word": values_per_word(act_bits, axis_word_bits),
            "output_per_axis_word": values_per_word(act_bits, axis_word_bits),
            "weight_per_axi_word": values_per_word(weight_bits, axi_word_bits),
            "bias_per_axi_word": values_per_word(bias_bits, axi_word_bits),
            "activation_per_axi_word": values_per_word(act_bits, axi_word_bits),
        },
        "element_counts": {
            "input": int(input_elements),
            "output": int(output_elements),
            "weight": int(weight_elements),
            "bias": int(bias_elements),
            "activation_buffer": int(activation_buffer_elements),
        },
    }

    layout["raw_bits"] = {
        "input": raw_bit_count(input_elements, act_bits),
        "output": raw_bit_count(output_elements, act_bits),
        "weight": raw_bit_count(weight_elements, weight_bits),
        "bias": raw_bit_count(bias_elements, bias_bits),
        "activation_buffer": raw_bit_count(activation_buffer_elements, act_bits),
    }

    layout["raw_bytes"] = {
        "input": raw_byte_count(input_elements, act_bits),
        "output": raw_byte_count(output_elements, act_bits),
        "weight": raw_byte_count(weight_elements, weight_bits),
        "bias": raw_byte_count(bias_elements, bias_bits),
        "activation_buffer": raw_byte_count(activation_buffer_elements, act_bits),
    }

    layout["packed_transfer_bytes"] = {
        "input_axis": packed_byte_count(input_elements, act_bits, axis_word_bits),
        "output_axis": packed_byte_count(output_elements, act_bits, axis_word_bits),
        "weight_axi": packed_byte_count(weight_elements, weight_bits, axi_word_bits),
        "bias_axi": packed_byte_count(bias_elements, bias_bits, axi_word_bits),
        "activation_axi": packed_byte_count(activation_buffer_elements, act_bits, axi_word_bits),
    }

    layout["packed_word_counts"] = {
        "input_axis": packed_word_count(input_elements, act_bits, axis_word_bits),
        "output_axis": packed_word_count(output_elements, act_bits, axis_word_bits),
        "weight_axi": packed_word_count(weight_elements, weight_bits, axi_word_bits),
        "bias_axi": packed_word_count(bias_elements, bias_bits, axi_word_bits),
        "activation_axi": packed_word_count(activation_buffer_elements, act_bits, axi_word_bits),
    }

    return layout


def precision_layout_markdown(layout: Mapping[str, Any]) -> str:
    bits = layout.get("bits", {})
    pack = layout.get("pack_factors", {})
    elems = layout.get("element_counts", {})
    raw_bytes = layout.get("raw_bytes", {})
    packed_bytes = layout.get("packed_transfer_bytes", {})

    lines = [
        "# Precision layout",
        "",
        f"- Precision mode: `{layout.get('precision_mode')}`",
        f"- Activation bits: `{bits.get('activation')}`",
        f"- Weight bits: `{bits.get('weight')}`",
        f"- Bias bits: `{bits.get('bias')}`",
        f"- Accumulator bits: `{bits.get('accum')}`",
        "",
        "## Pack factors",
        "",
        f"- Activation values per AXIS word: `{pack.get('activation_per_axis_word')}`",
        f"- Output values per AXIS word: `{pack.get('output_per_axis_word')}`",
        f"- Weight values per AXI word: `{pack.get('weight_per_axi_word')}`",
        f"- Bias values per AXI word: `{pack.get('bias_per_axi_word')}`",
        f"- Activation values per AXI word: `{pack.get('activation_per_axi_word')}`",
        "",
        "## Element counts",
        "",
        f"- Input elements: `{elems.get('input')}`",
        f"- Output elements: `{elems.get('output')}`",
        f"- Weight elements: `{elems.get('weight')}`",
        f"- Bias elements: `{elems.get('bias')}`",
        f"- Activation buffer elements: `{elems.get('activation_buffer')}`",
        "",
        "## Raw byte counts",
        "",
        f"- Input raw bytes: `{raw_bytes.get('input')}`",
        f"- Output raw bytes: `{raw_bytes.get('output')}`",
        f"- Weight raw bytes: `{raw_bytes.get('weight')}`",
        f"- Bias raw bytes: `{raw_bytes.get('bias')}`",
        f"- Activation-buffer raw bytes: `{raw_bytes.get('activation_buffer')}`",
        "",
        "## Packed transfer byte counts",
        "",
        f"- Input AXIS bytes: `{packed_bytes.get('input_axis')}`",
        f"- Output AXIS bytes: `{packed_bytes.get('output_axis')}`",
        f"- Weight AXI bytes: `{packed_bytes.get('weight_axi')}`",
        f"- Bias AXI bytes: `{packed_bytes.get('bias_axi')}`",
        f"- Activation AXI bytes: `{packed_bytes.get('activation_axi')}`",
        "",
    ]
    return "\n".join(lines)



def default_precision_policy(
    raw_cfg: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for role in PRECISION_ROLES:
        result[role] = normalize_precision_spec(
            _cfg_get(
                raw_cfg,
                f"numerics.defaults.{role}",
            ),
            fallback=DEFAULT_PRECISION[role],
            path=f"numerics.defaults.{role}",
        )

    return result


def _candidate_names(
    op: Any,
    aliases: Iterable[str],
) -> set[str]:
    names = {
        str(value)
        for value in aliases
        if str(value)
    }

    name = getattr(
        op,
        "name",
        "",
    )

    if name:
        names.add(str(name))

    attrs = getattr(
        op,
        "attrs",
        {},
    ) or {}

    if isinstance(attrs, Mapping):
        for key in (
            "src_name",
            "source_name",
            "onnx_name",
        ):
            value = attrs.get(key)

            if value:
                names.add(str(value))

    return names


def rule_matches(
    rule_match: Mapping[str, Any],
    op: Any,
    index: int,
    *,
    name_aliases: Sequence[str] = (),
) -> bool:
    if (
        not isinstance(rule_match, Mapping)
        or not rule_match
    ):
        return False

    if "name" in rule_match:
        names = _candidate_names(
            op,
            name_aliases,
        )

        if str(rule_match["name"]) not in names:
            return False

    if "op_type" in rule_match:
        expected = canonical_op_type(
            str(rule_match["op_type"])
        )
        actual = canonical_op_type(
            str(
                getattr(
                    op,
                    "op_type",
                    "",
                )
            )
        )

        if expected != actual:
            return False

    if "index" in rule_match:
        try:
            expected_index = int(
                rule_match["index"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "numerics.layers[].match.index "
                "must be an integer"
            ) from exc

        if expected_index != index:
            return False

    return True


@dataclass(frozen=True)
class ResolvedPrecision:
    specs: Dict[str, Dict[str, Any]]
    matched_rule_indices: tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specs": deepcopy(self.specs),
            "matched_rule_indices": list(
                self.matched_rule_indices
            ),
        }


def resolve_precision_for_op(
    raw_cfg: Mapping[str, Any],
    op: Any,
    index: int,
    *,
    name_aliases: Sequence[str] = (),
) -> ResolvedPrecision:
    specs = default_precision_policy(raw_cfg)

    rules = _cfg_get(
        raw_cfg,
        "numerics.layers",
        [],
    ) or []

    if not isinstance(rules, list):
        raise ValueError(
            "numerics.layers must be a list"
        )

    matched: list[int] = []

    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise ValueError(
                f"numerics.layers[{rule_index}] "
                "must be a mapping"
            )

        match = rule.get(
            "match",
            {},
        )

        if not rule_matches(
            match,
            op,
            index,
            name_aliases=name_aliases,
        ):
            continue

        matched.append(rule_index)

        for role in PRECISION_ROLES:
            if role not in rule:
                continue

            specs[role] = normalize_precision_spec(
                rule[role],
                fallback=specs[role],
                path=(
                    f"numerics.layers"
                    f"[{rule_index}].{role}"
                ),
            )

    return ResolvedPrecision(
        specs=specs,
        matched_rule_indices=tuple(matched),
    )


def sweep_layer_override_mode(
    sweep_cfg: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str:
    mode = str(
        candidate.get(
            "layer_overrides",
            sweep_cfg.get(
                "layer_overrides",
                "clear",
            ),
        )
    ).lower()

    if mode not in {
        "clear",
        "preserve",
    }:
        raise ValueError(
            "precision sweep layer_overrides must "
            "be 'clear' or 'preserve'"
        )

    return mode
