from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fpgai.backends.hls.emit.dense_tiling_codegen import (
    apply_dense_tiling_to_top_source,
    dense_tile_for_layer,
    rewrite_dense_calls_with_tiling,
)


def test_dense_tile_for_layer_reads_legacy_tile_keys() -> None:
    assert dense_tile_for_layer(
        {
            "tile": {
                "in": 16,
                "out": 8,
            }
        }
    ) == (16, 8)


def test_dense_tile_for_layer_reads_typed_architecture_keys() -> None:
    assert dense_tile_for_layer(
        {
            "architecture": {
                "tiling": {
                    "input": 32,
                    "output": 4,
                }
            }
        }
    ) == (32, 4)


def test_rewrite_dense_call_inserts_tile_template_arguments() -> None:
    source = (
        "dense_out_in<128, 64, in_t, out_t, w_t, b_t, acc_t>"
        "(x, y, W0, B0);"
    )

    rewritten = rewrite_dense_calls_with_tiling(
        source,
        [(16, 8)],
    )

    assert (
        "dense_out_in_tiled<128, 64, 16, 8, "
        "in_t, out_t, w_t, b_t, acc_t>"
    ) in rewritten
    assert "FPGAI real dense tiling helper" in rewritten
    assert (
        "for (int out_base = 0; out_base < OUT; "
        "out_base += TILE_OUT)"
    ) in rewritten
    assert (
        "for (int in_base = 0; in_base < IN; "
        "in_base += TILE_IN)"
    ) in rewritten


def test_apply_dense_tiling_to_top_source_matches_graph_order() -> None:
    graph = SimpleNamespace(
        ops=[
            SimpleNamespace(
                name="dense0",
                op_type="Dense",
            ),
            SimpleNamespace(
                name="relu0",
                op_type="Relu",
            ),
            SimpleNamespace(
                name="dense1",
                op_type="Dense",
            ),
        ]
    )

    source = "\n".join(
        [
            "dense_out_in<4, 3>(a, b, W0, B0);",
            "relu_inplace<3>(b);",
            "dense_out_in<3, 2>(b, c, W1, B1);",
        ]
    )

    plan = {
        "layer_plans": [
            {
                "node_name": "dense0",
                "tile": {
                    "in": 2,
                    "out": 1,
                },
            },
            {
                "node_name": "dense1",
                "architecture": {
                    "tiling": {
                        "input": 3,
                        "output": 2,
                    }
                },
            },
        ]
    }

    rewritten = apply_dense_tiling_to_top_source(
        source,
        graph,
        plan,
    )

    assert "dense_out_in_tiled<4, 3, 2, 1>" in rewritten
    assert "dense_out_in_tiled<3, 2, 3, 2>" in rewritten


def test_top_cpp_is_wrapped_for_dense_tiling() -> None:
    source = Path(
        "fpgai/backends/hls/emit/top_cpp.py"
    ).read_text(encoding="utf-8")

    assert "apply_dense_tiling_to_top_source" in source
    assert "_fpgai_dense_tiling_original_emit_top_cpp" in source


def test_dense_tile_for_layer_reads_nested_sizes() -> None:
    assert dense_tile_for_layer(
        {
            "architecture": {
                "tiling": {
                    "sizes": {
                        "in": 8,
                        "out": 4,
                    }
                }
            }
        }
    ) == (8, 4)


def test_rewrite_dense_call_handles_multiline_call_and_explicit_types() -> None:
    source = """
void top() {
    dense_out_in<676, 10, op3_act_t, op4_act_t, op4_wgt_t, op4_bias_t, op4_acc_t, 1, 1, 1, 1, 1, 1>(
        layer_3_out,
        layer_4_out,
        W1,
        B1
    );
}
"""

    rewritten = rewrite_dense_calls_with_tiling(source, [(64, 10)])

    assert "dense_out_in_tiled<676, 10, 64, 10, op3_act_t, op4_act_t, op4_wgt_t" in rewritten
    assert "typename IN_T = act_t" not in rewritten
    assert "const W_T weights[OUT * IN]" in rewritten
    assert "const W_T weights[OUT][IN]" in rewritten



def test_yaml_dense_tiling_reaches_planner_and_generated_top_cpp() -> None:
    from types import SimpleNamespace

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
    from fpgai.engine.layerwise_precision import resolve_layerwise_precision
    from fpgai.engine.planner import make_compile_plan
    from tests.test_mixed_precision_codegen import _base_config, _dense_graph

    graph = _dense_graph()
    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    cfg = SimpleNamespace(
        raw={
            "targets": {
                "platform": {
                    "board": "unit_test",
                    "part": "xck26-sfvc784-2LV-c",
                }
            },
            "data_movement": {
                "ps_pl": {
                    "weights": {
                        "mode": "embedded",
                    }
                }
            },
            "optimization": {
                "parallel_policy": "Balanced",
                "tiling": {
                    "dense": {
                        "tile_in": 2,
                        "tile_out": 3,
                    }
                },
            },
        }
    )
    desc = SimpleNamespace(
        node_name="dense0",
        op_type="Dense",
        attrs={
            "in_features": 4,
            "out_features": 3,
        },
        compute_hint="compute_bound",
        backend_kernel=None,
        param_bytes=0,
        activation_bytes_in=0,
        activation_bytes_out=0,
    )

    plan = make_compile_plan(
        cfg,
        [desc],
    )

    assert plan.layer_plans[0].tile == {
        "in": 2,
        "out": 3,
    }
    assert plan.layer_plans[0].architecture.tiling.sizes == {
        "in": 2,
        "out": 3,
    }

    generated = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=plan,
    )

    assert "FPGAI real dense tiling helper" in generated
    assert "dense_out_in_tiled<4, 3, 2, 3" in generated


def test_yaml_conv_tiling_reaches_planner_and_generated_top_cpp() -> None:
    from types import SimpleNamespace

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
    from fpgai.engine.layerwise_precision import resolve_layerwise_precision
    from fpgai.engine.planner import make_compile_plan
    from tests.test_mixed_precision_codegen import _base_config, _conv_graph

    graph = _conv_graph()
    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    cfg = SimpleNamespace(
        raw={
            "targets": {
                "platform": {
                    "board": "unit_test",
                    "part": "xck26-sfvc784-2LV-c",
                }
            },
            "data_movement": {
                "ps_pl": {
                    "weights": {
                        "mode": "embedded",
                    }
                }
            },
            "optimization": {
                "parallel_policy": "Balanced",
                "tiling": {
                    "conv": {
                        "tile_ic": 1,
                        "tile_oc": 2,
                        "tile_oh": 2,
                        "tile_ow": 2,
                    }
                },
            },
        }
    )
    desc = SimpleNamespace(
        node_name="conv0",
        op_type="Conv",
        attrs={
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "dilations": [1, 1],
        },
        compute_hint="compute_bound",
        backend_kernel=None,
        param_bytes=0,
        activation_bytes_in=0,
        activation_bytes_out=0,
    )

    plan = make_compile_plan(
        cfg,
        [desc],
    )

    assert plan.layer_plans[0].tile == {
        "oh": 2,
        "ow": 2,
        "oc": 2,
        "ic": 1,
    }
    assert plan.layer_plans[0].architecture.tiling.sizes == {
        "oh": 2,
        "ow": 2,
        "oc": 2,
        "ic": 1,
    }

    generated = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=plan,
    )

    assert "FPGAI real convolution tiling helper" in generated
    assert "conv2d_tiled<" in generated
    assert ", 2, 2, 2, 1," in generated



def test_yaml_pipeline_style_reaches_planner_and_generated_top_cpp() -> None:
    from types import SimpleNamespace

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
    from fpgai.engine.layerwise_precision import resolve_layerwise_precision
    from fpgai.engine.planner import make_compile_plan
    from tests.test_mixed_precision_codegen import _base_config, _dense_graph

    graph = _dense_graph()
    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    desc = SimpleNamespace(
        node_name="dense0",
        op_type="Dense",
        attrs={
            "in_features": 4,
            "out_features": 3,
        },
        compute_hint="compute_bound",
        backend_kernel=None,
        param_bytes=0,
        activation_bytes_in=0,
        activation_bytes_out=0,
    )

    aggressive_cfg = SimpleNamespace(
        raw={
            "targets": {"platform": {"board": "unit_test", "part": "xck26-sfvc784-2LV-c"}},
            "data_movement": {"ps_pl": {"weights": {"mode": "embedded"}}},
            "optimization": {
                "parallel_policy": "Balanced",
                "pipeline": {
                    "style": "aggressive",
                },
            },
        }
    )
    conservative_cfg = SimpleNamespace(
        raw={
            "targets": {"platform": {"board": "unit_test", "part": "xck26-sfvc784-2LV-c"}},
            "data_movement": {"ps_pl": {"weights": {"mode": "embedded"}}},
            "optimization": {
                "parallel_policy": "Balanced",
                "pipeline": {
                    "style": "conservative",
                },
            },
        }
    )
    explicit_ii_cfg = SimpleNamespace(
        raw={
            "targets": {"platform": {"board": "unit_test", "part": "xck26-sfvc784-2LV-c"}},
            "data_movement": {"ps_pl": {"weights": {"mode": "embedded"}}},
            "optimization": {
                "parallel_policy": "Balanced",
                "pipeline": {
                    "style": "aggressive",
                    "ii": 5,
                },
            },
        }
    )

    aggressive_plan = make_compile_plan(aggressive_cfg, [desc])
    conservative_plan = make_compile_plan(conservative_cfg, [desc])
    explicit_ii_plan = make_compile_plan(explicit_ii_cfg, [desc])

    assert aggressive_plan.layer_plans[0].pipeline_ii == 1
    assert aggressive_plan.layer_plans[0].architecture.pipeline.ii == 1
    assert aggressive_plan.layer_plans[0].architecture.pipeline.style == "aggressive"

    assert conservative_plan.layer_plans[0].pipeline_ii == 3
    assert conservative_plan.layer_plans[0].architecture.pipeline.ii == 3
    assert conservative_plan.layer_plans[0].architecture.pipeline.style == "conservative"

    assert explicit_ii_plan.layer_plans[0].pipeline_ii == 5
    assert explicit_ii_plan.layer_plans[0].architecture.pipeline.ii == 5

    aggressive_top = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=aggressive_plan,
    )
    conservative_top = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=conservative_plan,
    )
    explicit_ii_top = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=explicit_ii_plan,
    )

    assert ", 1, " in aggressive_top
    assert "pipeline_ii=1" in aggressive_top
    assert ", 3, " in conservative_top
    assert "pipeline_ii=3" in conservative_top
    assert ", 5, " in explicit_ii_top
    assert "pipeline_ii=5" in explicit_ii_top
    assert aggressive_top != conservative_top
    assert aggressive_top != explicit_ii_top



def test_yaml_parallel_knobs_reach_planner_and_generated_top_cpp() -> None:
    from types import SimpleNamespace

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
    from fpgai.engine.layerwise_precision import resolve_layerwise_precision
    from fpgai.engine.planner import make_compile_plan
    from tests.test_mixed_precision_codegen import _base_config, _dense_graph

    graph = _dense_graph()
    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    desc = SimpleNamespace(
        node_name="dense0",
        op_type="Dense",
        attrs={
            "in_features": 4,
            "out_features": 3,
        },
        compute_hint="compute_bound",
        backend_kernel=None,
        param_bytes=0,
        activation_bytes_in=0,
        activation_bytes_out=0,
    )

    serial_cfg = SimpleNamespace(
        raw={
            "targets": {"platform": {"board": "unit_test", "part": "xck26-sfvc784-2LV-c"}},
            "data_movement": {"ps_pl": {"weights": {"mode": "embedded"}}},
            "optimization": {
                "parallel_policy": "Balanced",
                "parallel": {
                    "pe": 1,
                    "simd": 1,
                    "unroll_factor": 1,
                    "partition_factor": 1,
                    "array_partition_mode": "block",
                },
            },
        }
    )
    parallel_cfg = SimpleNamespace(
        raw={
            "targets": {"platform": {"board": "unit_test", "part": "xck26-sfvc784-2LV-c"}},
            "data_movement": {"ps_pl": {"weights": {"mode": "embedded"}}},
            "optimization": {
                "parallel_policy": "Balanced",
                "parallel": {
                    "pe": 3,
                    "simd": 2,
                    "unroll_factor": 4,
                    "partition_factor": 5,
                    "array_partition_mode": "block",
                },
            },
        }
    )

    serial_plan = make_compile_plan(serial_cfg, [desc])
    parallel_plan = make_compile_plan(parallel_cfg, [desc])

    serial_layer = serial_plan.layer_plans[0]
    parallel_layer = parallel_plan.layer_plans[0]

    assert serial_layer.unroll == {
        "in": 1,
        "out": 1,
    }
    assert serial_layer.architecture.parallelism.pe == 1
    assert serial_layer.architecture.parallelism.simd == 1
    assert serial_layer.architecture.partitioning.factor == 1
    assert serial_layer.architecture.partitioning.mode == "block"

    assert parallel_layer.unroll == {
        "in": 2,
        "out": 3,
    }
    assert parallel_layer.architecture.parallelism.pe == 3
    assert parallel_layer.architecture.parallelism.simd == 2
    assert parallel_layer.architecture.partitioning.factor == 5
    assert parallel_layer.architecture.partitioning.mode == "block"
    assert parallel_layer.architecture.partitioning.targets["input"] == 5
    assert parallel_layer.architecture.partitioning.targets["output"] == 5
    assert parallel_layer.architecture.partitioning.targets["weight"] == 6

    serial_top = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=serial_plan,
    )
    parallel_top = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=parallel_plan,
    )

    assert "input_unroll=1 output_unroll=1 input_partition=1 output_partition=1 weight_partition=1" in serial_top
    assert "input_unroll=2 output_unroll=3 input_partition=5 output_partition=5 weight_partition=6" in parallel_top
    assert serial_top != parallel_top


def test_yaml_array_partition_mode_changes_generated_layer_pragmas() -> None:
    from types import SimpleNamespace

    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)
    compiler.cfg = SimpleNamespace(
        raw={
            "optimization": {
                "parallel": {
                    "array_partition_mode": "block",
                }
            }
        }
    )

    source = "#pragma HLS ARRAY_PARTITION variable=x cyclic factor=4 dim=1\n"
    rewritten = compiler._apply_hls_array_partition_mode(
        source,
        compiler._hls_array_partition_mode(None),
    )

    assert "#pragma HLS ARRAY_PARTITION variable=x block factor=4 dim=1" in rewritten
    assert "FPGAI array partition mode: block" in rewritten
    assert "cyclic factor=4" not in rewritten
