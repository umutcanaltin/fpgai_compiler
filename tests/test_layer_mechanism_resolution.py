from __future__ import annotations

import numpy as np

from fpgai.ir import Graph
from fpgai.ir.contracts import ImplementationCandidate
from fpgai.ir.passes.mechanism_resolution import resolve_layer_mechanisms
from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state, plan_transformer_execution
from fpgai.layers.composites import CompositeLayerSpec, expand_composite_layers, register_composite_layer
from fpgai.ir.ops import Op


def _graph() -> Graph:
    g = Graph("mechanisms")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor("x", (1, 2, 4))
    g.add_tensor("w", (4, 4))
    g.add_tensor("y", (1, 2, 4))
    g.constants["w"] = np.eye(4, dtype=np.float32)
    op = Op("mm", "MatMul", ["x", "w"], ["y"], {})
    op.semantics.implementation_candidates = (
        ImplementationCandidate(backend="hls", implementation_id="fpgai.hls.matmul"),
        ImplementationCandidate(backend="vhdl", implementation_id="fpgai.vhdl.matmul"),
    )
    g.ops = [op]
    return g


def test_explicit_layer_knobs_override_without_forcing_unspecified_layers():
    g = _graph()
    report = resolve_layer_mechanisms(g, {
        "architecture": {
            "layers": [{
                "match": {"name": "mm"},
                "memory": {"weight_storage": "bram", "activation_storage": "uram", "gradient_storage": "ddr"},
                "implementation": {"backend": "vhdl"},
                "execution": {"mode": "phase_shared"},
                "buffering": {"storage": "bram"},
            }]
        }
    })
    op = g.ops[0]
    assert report["rejected_layer_count"] == 0
    assert op.semantics.selected_backend == "vhdl"
    assert op.semantics.schedule["execution_mode"] == "phase_shared"
    assert op.semantics.training["gradient_storage"] == "ddr"
    assert g.get_tensor("w").semantics.memory.storage == "bram"
    assert g.get_tensor("y").semantics.memory.storage == "uram"


def test_auto_mechanisms_do_not_force_memory_or_backend():
    g = _graph()
    report = resolve_layer_mechanisms(g, {"architecture": {"defaults": {"memory": {"weight_storage": "auto"}}}})
    assert report["rejected_layer_count"] == 0
    assert g.get_tensor("w").semantics.memory.storage == "unspecified"
    assert g.ops[0].semantics.selected_backend is None


def test_illegal_backend_is_reported_not_silently_replaced():
    g = _graph()
    report = resolve_layer_mechanisms(g, {"architecture": {"layers": [{"match": {"name": "mm"}, "implementation": {"backend": "external"}}]}})
    assert report["rejected_layer_count"] == 1
    assert g.ops[0].semantics.selected_backend is None
    assert report["layers"][0]["rejected"][0]["reason"] == "no_declared_candidate"


def test_transformer_planner_auto_does_not_force_ddr_or_bram():
    g = Graph("transformer_auto")
    g.add_tensor("x", (1, 2, 4))
    g.add_tensor("w", (4, 4))
    g.add_tensor("q", (1, 2, 4))
    g.add_tensor("k_cache", (1, 8, 4))
    g.add_tensor("v_cache", (1, 8, 4))
    g.constants["w"] = np.eye(4, dtype=np.float32)
    g.add_op("MatMul", ["x", "w"], ["q"], name="q", attrs={"projection_role": "q"})
    configure_kv_cache_state(g, key_cache="k_cache", value_cache="v_cache", capacity=8, storage="auto")
    plan_transformer_execution(g, model_dimension=4, num_heads=1, max_sequence_length=8)
    assert g.get_tensor("w").semantics.memory.storage == "unspecified"
    assert g.get_tensor("k_cache").semantics.memory.storage == "unspecified"
    assert "score_buffer" not in (g.ops[0].semantics.buffering or {})
    assert g.metadata["transformer_execution_policy"]["weight_storage"] == "auto"


def test_ecosystem_composite_expansion_keeps_provider_provenance():
    name = "CommunityBlockForTest"
    def expand(graph: Graph, op: Op):
        graph.add_tensor("community_out", tuple(graph.get_tensor(op.inputs[0]).shape))
        return [Op("community_relu", "Relu", [op.inputs[0]], [op.outputs[0]], {})]
    try:
        register_composite_layer(CompositeLayerSpec(name, expand, provider="community.test", version="2"), replace=True)
        g = Graph("community")
        g.inputs = ["x"]; g.outputs = ["y"]
        g.add_tensor("x", (4,)); g.add_tensor("y", (4,))
        g.add_op(name, ["x"], ["y"], name="block")
        expand_composite_layers(g)
        provenance = g.ops[0].attrs["_fpgai_composite_provider"]
        assert provenance["provider"] == "community.test"
        assert provenance["version"] == "2"
    finally:
        pass


def test_hls_mha_auto_execution_is_resolved_and_reported_not_silently_forced():
    from fpgai.ir import Graph
    from fpgai.ir.passes.mechanism_resolution import resolve_layer_mechanisms

    g = Graph("mha_auto")
    for name in ("q", "k", "v", "y"):
        g.add_tensor(name, (1, 4, 8))
    g.inputs = ["q", "k", "v"]
    g.outputs = ["y"]
    g.add_op("MultiHeadAttention", ["q", "k", "v"], ["y"], name="mha", attrs={"num_heads": 2, "execution_mode": "auto"})
    report = resolve_layer_mechanisms(g, {"architecture": {"defaults": {"implementation": {"backend": "hls"}, "execution": {"mode": "auto"}}}})
    item = report["layers"][0]
    assert item["resolved"]["execution_mode"] == "serialized"
    assert item["resolved"]["execution_mode_source"] == "backend_default:hls.multi_head_attention"
    assert g.ops[0].attrs["execution_mode"] == "serialized"
