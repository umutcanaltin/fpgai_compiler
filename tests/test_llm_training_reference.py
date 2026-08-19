from __future__ import annotations

import json
import numpy as np

from fpgai.benchmark.training_reference import run_training_reference_step
from fpgai.ir import Graph


def _add_const(g: Graph, name: str, value: np.ndarray) -> None:
    arr = np.asarray(value, dtype=np.float32)
    g.add_tensor(name, arr.shape)
    g.constants[name] = arr


def test_matmul_silu_mul_rmsnorm_have_reference_backward_and_updates(tmp_path):
    g = Graph("llm_train")
    g.inputs = ["x"]; g.outputs = ["y"]
    g.add_tensor("x", (1, 2, 4)); g.add_tensor("m", (1, 2, 4)); g.add_tensor("a", (1, 2, 4)); g.add_tensor("z", (1, 2, 4)); g.add_tensor("y", (1, 2, 4))
    _add_const(g, "w", np.eye(4, dtype=np.float32) * 0.5)
    _add_const(g, "scale", np.ones((4,), dtype=np.float32))
    g.add_op("MatMul", ["x", "w"], ["m"], name="mm")
    g.add_op("SiLU", ["m"], ["a"], name="silu")
    g.add_op("Mul", ["a", "a"], ["z"], name="mul")
    g.add_op("RMSNorm", ["z", "scale"], ["y"], name="norm", attrs={"axis": -1, "epsilon": 1e-5})
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 10.0
    target = np.ones((1, 2, 4), dtype=np.float32) * 0.2
    result = run_training_reference_step(
        graph=g,
        raw_cfg={"training": {"optimizer": {"type": "sgd", "learning_rate": 0.01}, "loss": {"type": "mse"}}},
        out_dir=tmp_path,
        x_input=x,
        target=target,
    )
    assert np.isfinite(result.loss_before)
    assert np.isfinite(result.loss_after)
    before = np.fromfile(result.weights_before_flat_path, dtype=np.float32)
    after = np.fromfile(result.weights_after_flat_path, dtype=np.float32)
    assert before.size == after.size == 20
    assert not np.array_equal(before, after)
    payload = json.loads(result.summary_json.read_text())
    assert payload["loss_before"] == result.loss_before


def test_rope_and_multihead_attention_reference_backward_execute(tmp_path):
    g = Graph("attn_train")
    g.inputs = ["q"]; g.outputs = ["y"]
    for name in ("q", "k", "v", "qr", "kr", "y"):
        g.add_tensor(name, (1, 2, 4))
    _add_const(g, "cos", np.ones((4, 2), dtype=np.float32))
    _add_const(g, "sin", np.zeros((4, 2), dtype=np.float32))
    # k/v are constants here only to keep the single-input training reference API.
    g.constants["k"] = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 20.0
    g.constants["v"] = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 30.0
    g.add_op("RotaryEmbedding", ["q", "cos", "sin"], ["qr"], name="q_rope", attrs={"rotary_dim": 4, "position_offset": 0})
    g.add_op("RotaryEmbedding", ["k", "cos", "sin"], ["kr"], name="k_rope", attrs={"rotary_dim": 4, "position_offset": 0})
    g.add_op("MultiHeadAttention", ["qr", "kr", "v"], ["y"], name="mha", attrs={"num_heads": 2, "causal": True})
    q = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 10.0
    result = run_training_reference_step(
        graph=g,
        raw_cfg={"training": {"optimizer": {"type": "sgd", "learning_rate": 0.0}, "loss": {"type": "mse"}}},
        out_dir=tmp_path,
        x_input=q,
        target=np.zeros((1, 2, 4), dtype=np.float32),
    )
    assert np.isfinite(result.loss_before)
    assert (result.layerwise_dir / "mha__bwd_output_grad.bin").is_file()
