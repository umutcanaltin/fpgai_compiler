from __future__ import annotations

from types import SimpleNamespace

import pytest

from fpgai.engine.layerwise_precision import (
    resolve_layerwise_precision,
)
from fpgai.numerics.precision_policy import (
    canonical_op_type,
    default_precision_policy,
    normalize_precision_spec,
    resolve_precision_for_op,
    sweep_layer_override_mode,
)


def _spec(
    total_bits: int,
    int_bits: int,
) -> dict[str, object]:
    return {
        "type": "ap_fixed",
        "total_bits": total_bits,
        "int_bits": int_bits,
    }


def _op(
    op_type: str,
    name: str,
    *,
    src_name: str | None = None,
) -> SimpleNamespace:
    attrs = {}

    if src_name is not None:
        attrs["src_name"] = src_name

    return SimpleNamespace(
        op_type=op_type,
        name=name,
        attrs=attrs,
    )


def test_canonical_op_type_maps_dense_aliases() -> None:
    assert canonical_op_type("Dense") == "Dense"
    assert canonical_op_type("dense") == "Dense"
    assert canonical_op_type("Gemm") == "Dense"
    assert canonical_op_type("MatMul") == "Dense"
    assert canonical_op_type("Linear") == "Dense"
    assert canonical_op_type("torch_linear") == "Dense"


def test_canonical_op_type_preserves_unknown_operator() -> None:
    assert canonical_op_type("Conv") == "Conv"
    assert canonical_op_type("CustomOp") == "CustomOp"


def test_default_precision_policy_uses_defaults() -> None:
    result = default_precision_policy({})

    assert result == {
        "activation": _spec(16, 6),
        "weight": _spec(16, 6),
        "bias": _spec(24, 10),
        "accum": _spec(24, 10),
    }


def test_default_precision_policy_reads_yaml_values() -> None:
    raw = {
        "numerics": {
            "defaults": {
                "activation": _spec(12, 5),
                "weight": _spec(10, 4),
                "bias": _spec(18, 7),
                "accum": _spec(22, 9),
            }
        }
    }

    result = default_precision_policy(raw)

    assert result["activation"] == _spec(12, 5)
    assert result["weight"] == _spec(10, 4)
    assert result["bias"] == _spec(18, 7)
    assert result["accum"] == _spec(22, 9)


def test_normalize_precision_spec_uses_fallback_fields() -> None:
    result = normalize_precision_spec(
        {
            "total_bits": 10,
        },
        fallback=_spec(16, 6),
        path="test.precision",
    )

    assert result == _spec(10, 6)


def test_dense_rule_matches_onnx_gemm_alias() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "op_type": "Dense",
                    },
                    "weight": _spec(9, 3),
                }
            ]
        }
    }

    result = resolve_precision_for_op(
        raw,
        _op(
            "Gemm",
            "classifier",
        ),
        0,
    )

    assert result.specs["weight"] == _spec(9, 3)
    assert result.matched_rule_indices == (0,)


def test_dense_rule_matches_onnx_matmul_alias() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "op_type": "Dense",
                    },
                    "activation": _spec(11, 4),
                }
            ]
        }
    }

    result = resolve_precision_for_op(
        raw,
        _op(
            "MatMul",
            "matmul_node",
        ),
        0,
    )

    assert result.specs["activation"] == _spec(11, 4)
    assert result.matched_rule_indices == (0,)


def test_name_rule_matches_stable_name() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "name": "dense0",
                    },
                    "accum": _spec(22, 9),
                }
            ]
        }
    }

    result = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense0",
        ),
        0,
    )

    assert result.specs["accum"] == _spec(22, 9)
    assert result.matched_rule_indices == (0,)


def test_name_rule_matches_original_source_name() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "name": "onnx_classifier",
                    },
                    "activation": _spec(10, 4),
                }
            ]
        }
    }

    result = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense0",
            src_name="onnx_classifier",
        ),
        0,
    )

    assert result.specs["activation"] == _spec(10, 4)
    assert result.matched_rule_indices == (0,)


def test_name_rule_matches_explicit_alias() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "name": "dense0",
                    },
                    "weight": _spec(8, 3),
                }
            ]
        }
    }

    result = resolve_precision_for_op(
        raw,
        _op(
            "Gemm",
            "original_name",
        ),
        0,
        name_aliases=[
            "dense0",
        ],
    )

    assert result.specs["weight"] == _spec(8, 3)
    assert result.matched_rule_indices == (0,)


def test_index_rule_matches_only_requested_index() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "index": 1,
                    },
                    "bias": _spec(17, 7),
                }
            ]
        }
    }

    first = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense0",
        ),
        0,
    )
    second = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense1",
        ),
        1,
    )

    assert first.specs["bias"] == _spec(24, 10)
    assert first.matched_rule_indices == ()

    assert second.specs["bias"] == _spec(17, 7)
    assert second.matched_rule_indices == (0,)


def test_combined_match_requires_all_fields() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "op_type": "Dense",
                        "name": "dense1",
                        "index": 1,
                    },
                    "weight": _spec(7, 3),
                }
            ]
        }
    }

    wrong_name = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense0",
        ),
        1,
    )
    wrong_index = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense1",
        ),
        0,
    )
    matching = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense1",
        ),
        1,
    )

    assert wrong_name.matched_rule_indices == ()
    assert wrong_index.matched_rule_indices == ()
    assert matching.matched_rule_indices == (0,)
    assert matching.specs["weight"] == _spec(7, 3)


def test_later_matching_rule_overrides_only_its_roles() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "op_type": "Dense",
                    },
                    "activation": _spec(12, 5),
                    "weight": _spec(11, 4),
                },
                {
                    "match": {
                        "name": "dense0",
                    },
                    "weight": _spec(8, 3),
                },
            ]
        }
    }

    result = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense0",
        ),
        0,
    )

    assert result.specs["activation"] == _spec(12, 5)
    assert result.specs["weight"] == _spec(8, 3)
    assert result.specs["bias"] == _spec(24, 10)
    assert result.specs["accum"] == _spec(24, 10)
    assert result.matched_rule_indices == (0, 1)


def test_empty_match_does_not_apply_to_every_layer() -> None:
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {},
                    "weight": _spec(8, 3),
                }
            ]
        }
    }

    result = resolve_precision_for_op(
        raw,
        _op(
            "Dense",
            "dense0",
        ),
        0,
    )

    assert result.specs["weight"] == _spec(16, 6)
    assert result.matched_rule_indices == ()


def test_invalid_precision_range_is_rejected() -> None:
    raw = {
        "numerics": {
            "defaults": {
                "activation": _spec(8, 9),
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        resolve_precision_for_op(
            raw,
            _op(
                "Relu",
                "relu0",
            ),
            0,
        )


def test_zero_total_bits_is_rejected() -> None:
    raw = {
        "numerics": {
            "defaults": {
                "activation": _spec(0, 0),
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        resolve_precision_for_op(
            raw,
            _op(
                "Relu",
                "relu0",
            ),
            0,
        )


def test_boolean_bit_width_is_rejected() -> None:
    raw = {
        "numerics": {
            "defaults": {
                "activation": {
                    "type": "ap_fixed",
                    "total_bits": True,
                    "int_bits": 4,
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="requires integer",
    ):
        resolve_precision_for_op(
            raw,
            _op(
                "Relu",
                "relu0",
            ),
            0,
        )


def test_non_integer_bit_width_is_rejected() -> None:
    raw = {
        "numerics": {
            "defaults": {
                "activation": {
                    "type": "ap_fixed",
                    "total_bits": 16.0,
                    "int_bits": 6,
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="requires integer",
    ):
        resolve_precision_for_op(
            raw,
            _op(
                "Relu",
                "relu0",
            ),
            0,
        )


def test_unsupported_precision_type_is_rejected() -> None:
    raw = {
        "numerics": {
            "defaults": {
                "activation": {
                    "type": "float32",
                    "total_bits": 32,
                    "int_bits": 8,
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="must be 'ap_fixed'",
    ):
        resolve_precision_for_op(
            raw,
            _op(
                "Relu",
                "relu0",
            ),
            0,
        )


def test_layerwise_resolver_updates_graph_metadata() -> None:
    operation = _op(
        "Dense",
        "dense0",
    )
    graph = SimpleNamespace(
        ops=[
            operation,
        ]
    )
    raw = {
        "numerics": {
            "layers": [
                {
                    "match": {
                        "op_type": "Dense",
                    },
                    "weight": _spec(9, 3),
                }
            ]
        }
    }

    resolve_layerwise_precision(
        graph,
        raw,
    )

    assert operation.attrs["precision"]["weight"] == _spec(
        9,
        3,
    )
    assert operation.attrs["precision_tag"] == "op0"
    assert operation.attrs["precision_rule_indices"] == [
        0,
    ]


def test_sweep_clears_layer_overrides_by_default() -> None:
    assert (
        sweep_layer_override_mode(
            {},
            {},
        )
        == "clear"
    )


def test_sweep_uses_global_preserve_mode() -> None:
    assert (
        sweep_layer_override_mode(
            {
                "layer_overrides": "preserve",
            },
            {},
        )
        == "preserve"
    )


def test_sweep_candidate_can_override_global_mode() -> None:
    assert (
        sweep_layer_override_mode(
            {
                "layer_overrides": "clear",
            },
            {
                "layer_overrides": "preserve",
            },
        )
        == "preserve"
    )


def test_invalid_sweep_override_mode_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be 'clear' or 'preserve'",
    ):
        sweep_layer_override_mode(
            {
                "layer_overrides": "sometimes",
            },
            {},
        )