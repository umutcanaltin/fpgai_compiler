from __future__ import annotations

import numpy as np

from fpgai.analysis.training_capability import audit_training_capabilities
from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp
from fpgai.benchmark.training_reference import run_training_reference_step
from fpgai.ir import Graph
from fpgai.ir.passes.infer_shapes import infer_shapes


def _cfg():
    return {
        "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 8}, "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 16}}},
        "targets": {"hls": {"control_protocol": "s_axilite"}},
        "training": {"loss": {"type": "mse"}, "optimizer": {"type": "sgd", "learning_rate": 0.01}},
    }


def test_concat_slice_resize_shape_and_hls_forward_backward() -> None:
    # Concat
    g = Graph("concat")
    g.inputs = ["a", "b"]; g.outputs = ["y"]
    for name, shape in {"a": (1, 2, 2), "b": (1, 3, 2), "y": ()}.items(): g.add_tensor(name, shape)
    g.add_op("Concat", ["a", "b"], ["y"], name="cat", attrs={"axis": 1})
    infer_shapes(g)
    assert g.get_tensor("y").shape == (1, 5, 2)
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert 'concat_axis<1, 2, 3, 2' in source
    train = emit_top_train_cpp(graph=g, top_name="train", weights_mode="embedded", training_cfg=_cfg()["training"])
    assert "concat_axis_backward<1, 2, 3, 2" in train

    # Slice
    s = Graph("slice")
    s.inputs=["x"]; s.outputs=["z"]
    s.add_tensor("x", (1, 5, 2)); s.add_tensor("z", ())
    s.add_op("Slice", ["x"], ["z"], name="slice", attrs={"starts":[1], "ends":[4], "axes":[1], "steps":[1]})
    infer_shapes(s)
    assert s.get_tensor("z").shape == (1, 3, 2)
    hls = emit_dag_top_cpp(s, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "slice_axis<1, 5, 1, 3, 2" in hls
    train = emit_top_train_cpp(graph=s, top_name="train", weights_mode="embedded", training_cfg=_cfg()["training"])
    assert "slice_axis_backward<1, 5, 1, 3, 2" in train

    # Resize
    r = Graph("resize")
    r.inputs=["x"]; r.outputs=["z"]
    r.add_tensor("x", (1, 2, 2, 2)); r.add_tensor("z", ())
    r.add_op("Resize", ["x"], ["z"], name="resize", attrs={"mode":"nearest", "sizes":[1,2,4,4]})
    infer_shapes(r)
    assert r.get_tensor("z").shape == (1, 2, 4, 4)
    hls = emit_dag_top_cpp(r, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "resize_nearest_nchw<1, 2, 2, 2, 4, 4" in hls
    train = emit_top_train_cpp(graph=r, top_name="train", weights_mode="embedded", training_cfg=_cfg()["training"])
    assert "resize_nearest_nchw_backward<1, 2, 2, 2, 4, 4" in train


def test_gather_embedding_reference_and_hls_training(tmp_path) -> None:
    g = Graph("embedding")
    g.inputs=["ids"]; g.outputs=["emb"]
    g.add_tensor("table", (6, 4)); g.add_tensor("ids", (2,)); g.add_tensor("emb", ())
    g.constants["table"] = np.arange(24, dtype=np.float32).reshape(6, 4) / 10.0
    g.add_op("Gather", ["table", "ids"], ["emb"], name="embedding", attrs={"axis": 0})
    infer_shapes(g)
    assert g.get_tensor("emb").shape == (2, 4)
    report = audit_training_capabilities(g)
    assert report["complete"] is True
    assert report["hardware_complete"] is True

    result = run_training_reference_step(
        graph=g,
        raw_cfg=_cfg(),
        out_dir=tmp_path,
        x_input=np.asarray([1, 4], dtype=np.float32),
        target=np.zeros((2, 4), dtype=np.float32),
    )
    before = np.fromfile(result.weights_before_flat_path, dtype=np.float32)
    after = np.fromfile(result.weights_after_flat_path, dtype=np.float32)
    assert before.size == 24
    assert after.size == 24
    assert not np.array_equal(before, after)

    hls = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "gather_rows<6, 4, 2" in hls
    train = emit_top_train_cpp(graph=g, top_name="train", weights_mode="embedded", training_cfg=_cfg()["training"])
    assert "gather_rows_backward<6, 4, 2" in train
    assert "dW_embedding" in train
