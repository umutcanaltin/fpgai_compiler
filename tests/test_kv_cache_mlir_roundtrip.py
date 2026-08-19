from __future__ import annotations

from fpgai.frontend.mlir import export_fpgai_mlir, import_fpgai_mlir
from fpgai.ir import Graph
from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state


def test_kv_cache_state_contract_survives_mlir_bridge_roundtrip():
    g = Graph("kv_state")
    g.inputs = ["x"]
    g.outputs = ["x"]
    g.add_tensor("x", (1, 1, 8), "float32")
    g.add_tensor("k_cache", (1, 32, 8), "float32")
    g.add_tensor("v_cache", (1, 32, 8), "float32")
    configure_kv_cache_state(g, key_cache="k_cache", value_cache="v_cache", capacity=32, sequence_axis=1, storage="ddr")
    restored = import_fpgai_mlir(export_fpgai_mlir(g))
    k = restored.get_tensor("k_cache")
    assert k.semantics.state.kind == "kv_key_cache"
    assert k.semantics.state.mutable is True
    assert k.semantics.state.persistent_across_invocations is True
    assert k.semantics.state.update_policy == "append"
    assert k.semantics.state.sequence_axis == 1
    assert k.semantics.state.capacity == 32
    assert k.semantics.memory.lifetime == "runtime_session"
