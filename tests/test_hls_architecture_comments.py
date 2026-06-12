from __future__ import annotations

from pathlib import Path

from fpgai.backends.hls.emit.architecture_comments import (
    emit_layer_architecture_comments,
)


def test_emit_layer_architecture_comments_from_dict_plan() -> None:
    text = emit_layer_architecture_comments(
        {
            "layer_plans": [
                {
                    "name": "dense0",
                    "op_type": "Dense",
                    "architecture": {
                        "pipeline": {"ii": 2},
                        "parallelism": {"pe": 4, "simd": 8},
                        "partitioning": {
                            "input": 2,
                            "output": 4,
                            "weight": 8,
                        },
                        "tiling": {"input": 16},
                        "memory": {"weights": "bram"},
                    },
                    "architecture_signature": "abc123",
                }
            ]
        }
    )

    assert "FPGAI effective per-layer architecture" in text
    assert "dense0" in text
    assert "ii=2" in text
    assert "pe=4" in text
    assert "simd=8" in text
    assert "part_in=2" in text
    assert "part_out=4" in text
    assert "part_w=8" in text
    assert "abc123" in text


def test_top_emitters_are_wrapped_with_architecture_comments() -> None:
    for path in [
        "fpgai/backends/hls/emit/top_cpp.py",
        "fpgai/backends/hls/emit/top_train_cpp.py",
    ]:
        source = Path(path).read_text(encoding="utf-8")
        assert "emit_layer_architecture_comments" in source
        assert "architecture-comment wrapper" in source
