from __future__ import annotations

from fpgai.benchmark.model_graphs import build_demo_transformer_block_graph
from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state, plan_transformer_execution


def test_transformer_plan_keeps_architecture_auto_until_user_or_backend_selects_it():
    g = build_demo_transformer_block_graph(sequence_length=4, model_dimension=8, num_heads=2, max_sequence_length=16)
    plans = g.metadata["transformer_execution_plans"]
    assert len(plans) == 1
    plan = plans[0]
    assert plan["schema"] == "fpgai.transformer-execution-plan/v1"
    assert plan["execution_mode"] == "auto"
    assert plan["num_heads"] == 2
    assert plan["head_dimension"] == 4
    assert plan["projection_ops"] == ["q_projection", "k_projection", "v_projection", "o_projection"]
    assert plan["rotary_ops"] == ["q_rope", "k_rope"]
    assert plan["attention_ops"] == ["mha"]
    assert set(plan["kv_cache_tensors"]) == {"k_cache", "v_cache"}
    assert g.semantics.runtime_contract["persistent_state"] is True
    assert g.get_tensor("k_cache").semantics.state.kind == "kv_key_cache"
    assert g.get_tensor("k_cache").semantics.state.capacity == 16
    assert g.get_tensor("k_cache").semantics.memory.storage in {None, "unspecified"}
    assert g.get_tensor("v_cache").semantics.state.persistent_across_invocations is True
    assert g.ops[5].semantics.schedule["reuse_group"] == "transformer_compute_engine_0"


def test_transformer_execution_mode_is_explicit_and_not_globally_forced():
    g = build_demo_transformer_block_graph(execution_mode="phase_shared")
    mha = next(op for op in g.ops if op.op_type == "MultiHeadAttention")
    assert mha.attrs["execution_mode"] == "phase_shared"
    assert mha.semantics.schedule["execution_mode"] == "phase_shared"
    candidate_ids = {x.implementation_id for x in mha.semantics.implementation_candidates}
    assert "fpgai.hls.mha.serialized" in candidate_ids
    assert "fpgai.vhdl.mha.streaming" in candidate_ids
