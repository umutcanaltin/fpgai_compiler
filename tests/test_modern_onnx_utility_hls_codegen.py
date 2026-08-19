import numpy as np

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.ir.graph import Graph
from fpgai.ir.passes.infer_shapes import infer_shapes


def _cfg():
    return {
        "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}, "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12}}},
        "targets": {"hls": {"control_protocol": "s_axilite"}},
    }


def test_decomposed_norm_math_ops_emit_generic_hls() -> None:
    g = Graph("decomposed_norm")
    g.inputs = ["x"]
    g.outputs = ["y"]
    for name, shape in {"x": (2, 4), "sq": (2, 4), "mean": (2,), "root": (2,), "y": (2,)}.items():
        g.add_tensor(name, shape, "float32")
    g.add_tensor("two", (), "float32")
    g.constants["two"] = np.asarray([2.0], dtype=np.float32)
    g.add_op("Pow", ["x", "two"], ["sq"], name="square")
    g.add_op("ReduceMean", ["sq"], ["mean"], name="mean", attrs={"axes": [-1], "keepdims": 0})
    g.add_op("Sqrt", ["mean"], ["root"], name="sqrt")
    g.add_op("Identity", ["root"], ["y"], name="identity")
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "square_vec_typed<8" in source
    assert "reduce_mean_last_axis<2, 4" in source
    assert "sqrt_vec_typed<2" in source
    assert "reshape_copy_typed<2" in source


def test_squeeze_unsqueeze_shape_inference_is_static_and_conservative() -> None:
    g = Graph("shape_utils")
    g.inputs = ["x"]
    g.outputs = ["z"]
    g.add_tensor("x", (1, 4, 1), "float32")
    g.add_op("Squeeze", ["x"], ["y"], name="squeeze", attrs={"axes": [0, 2]})
    g.add_op("Unsqueeze", ["y"], ["z"], name="unsqueeze", attrs={"axes": [0, 2]})
    infer_shapes(g)
    assert g.get_tensor("y").shape == (4,)
    assert g.get_tensor("z").shape == (1, 4, 1)


def test_decomposed_rmsnorm_broadcast_profile_emits_hls() -> None:
    g = Graph("decomposed_rmsnorm")
    g.inputs = ["x"]
    g.outputs = ["y"]
    shapes = {
        "x": (2, 4), "sq": (2, 4), "mean": (2, 1), "eps_added": (2, 1),
        "root": (2, 1), "norm": (2, 4), "y": (2, 4),
    }
    for name, shape in shapes.items():
        g.add_tensor(name, shape, "float32")
    g.add_tensor("two", (), "float32"); g.constants["two"] = np.asarray([2.0], dtype=np.float32)
    g.add_tensor("eps", (), "float32"); g.constants["eps"] = np.asarray([1e-5], dtype=np.float32)
    g.add_tensor("gamma", (4,), "float32"); g.constants["gamma"] = np.asarray([1.0, 0.9, 1.1, 1.0], dtype=np.float32)
    g.add_op("Pow", ["x", "two"], ["sq"], name="square")
    g.add_op("ReduceMean", ["sq"], ["mean"], name="mean", attrs={"axes": [-1], "keepdims": 1})
    g.add_op("Add", ["mean", "eps"], ["eps_added"], name="epsilon")
    g.add_op("Sqrt", ["eps_added"], ["root"], name="sqrt")
    g.add_op("Div", ["x", "root"], ["norm"], name="normalize")
    g.add_op("Mul", ["norm", "gamma"], ["y"], name="scale")
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "add_scalar_typed<2" in source
    assert "div_rows_by_scalar_vector<2, 4" in source
    assert "mul_rows_by_col_vector<2, 4" in source
