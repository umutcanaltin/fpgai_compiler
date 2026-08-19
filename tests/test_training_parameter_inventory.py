from pathlib import Path

import numpy as np

from fpgai.engine.training_graph_utils import (
    derive_training_parameter_inventory,
    training_parameter_word_count,
)
from fpgai.frontend.mlir import import_mlir_program


def test_layerwise_transformer_parameter_inventory_matches_hls_stream_contract():
    root = Path(__file__).resolve().parents[1]
    graph = import_mlir_program(
        root / "examples/reference/tiny_transformer_training.mlir",
        source_framework="fpgai",
    )

    inventory = derive_training_parameter_inventory(graph)

    assert training_parameter_word_count(graph) == 656
    assert sum(int(item["count"]) for item in inventory) == 656

    roles = [(item["layer"], item["role"], item["count"]) for item in inventory]
    assert ("block__attn_norm", "scale", 8) in roles
    assert ("block__q_projection", "weight", 64) in roles
    assert ("block__k_projection", "weight", 64) in roles
    assert ("block__v_projection", "weight", 64) in roles
    assert ("block__o_projection", "weight", 64) in roles
    assert ("block__ffn_norm", "scale", 8) in roles
    assert ("block__gate_projection", "weight", 128) in roles
    assert ("block__up_projection", "weight", 128) in roles
    assert ("block__down_projection", "weight", 128) in roles


def test_parameter_inventory_is_not_model_name_dependent():
    class Tensor:
        def __init__(self, shape): self.shape = shape

    class Op:
        name = "community_linear"
        op_type = "MatMul"
        inputs = ["x", "w"]
        outputs = ["y"]
        attrs = {"provider": "ecosystem.example"}

    class Graph:
        ops = [Op()]
        constants = {"w": np.ones((3, 5), dtype=np.float32)}
        params = {}
        tensors = {"x": Tensor((1, 2, 3)), "y": Tensor((1, 2, 5))}
        def get_tensor(self, name): return self.tensors.get(name)

    inventory = derive_training_parameter_inventory(Graph())
    assert inventory == [{
        "layer": "community_linear",
        "role": "weight",
        "tensor": "w",
        "count": 15,
        "shape": [3, 5],
    }]
    assert training_parameter_word_count(Graph()) == 15
