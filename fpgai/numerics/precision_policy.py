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
