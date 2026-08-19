from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.capabilities.capabilities import capability_for
from fpgai.ir.graph import Graph
from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state


def _cfg():
    return {
        "numerics": {
            "defaults": {
                "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12},
            }
        },
        "targets": {"hls": {"control_protocol": "s_axilite"}},
    }


def _state_graph(storage: str = "uram") -> Graph:
    g = Graph("kv_state")
    g.inputs = ["new_k"]
    g.outputs = ["cache_view"]
    g.add_tensor("k_cache", (1, 3, 8, 4), "float32")
    g.add_tensor("new_k", (1, 3, 1, 4), "float32")
    g.add_tensor("cache_view", (1, 3, 8, 4), "float32")
    configure_kv_cache_state(
        g,
        key_cache="k_cache",
        value_cache="k_cache",
        capacity=8,
        sequence_axis=2,
        storage=storage,
    )
    # configure_kv_cache_state accepts two names; using the same tensor is enough
    # here to exercise the generic state storage/codegen path.
    g.add_op(
        "KVCacheUpdate",
        ["k_cache", "new_k"],
        ["cache_view"],
        name="append_k",
        attrs={"sequence_axis": 2, "capacity": 8, "update_policy": "append"},
    )
    return g


def test_persistent_tensor_gets_dedicated_nonreused_hls_slot() -> None:
    allocation = build_hls_buffer_allocation(_state_graph(), raw_cfg=_cfg())
    slot = next(row for row in allocation["slots"] if "k_cache" in row["tensors"])
    assert slot["persistent"] is True
    assert slot["storage"] == "uram"
    assert slot["name"].startswith("fpgai_state_k_cache")
    assert slot["tensors"] == ["k_cache"]


def test_kv_cache_update_emits_static_uram_state_and_append_kernel() -> None:
    source = emit_dag_top_cpp(_state_graph(), top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "static ap_fixed<16,6" in source
    assert "fpgai_state_k_cache[96]" in source
    assert "BIND_STORAGE variable=fpgai_state_k_cache type=ram_2p impl=uram" in source
    assert "static int fpgai_state_k_cache_cursor = 0;" in source
    assert "persistent_state_append_axis<3, 8, 1, 4" in source
    assert "persistent_state_snapshot<96" in source


def test_ddr_persistent_state_emits_explicit_external_memory_port() -> None:
    source = emit_dag_top_cpp(_state_graph("ddr"), top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "ap_fixed<16,6" in source
    assert "* fpgai_state_k_cache" in source
    assert "m_axi port=fpgai_state_k_cache" in source
    assert "bundle=gmem_state_0" in source
    assert "int* fpgai_state_k_cache_cursor" in source
    assert "m_axi port=fpgai_state_k_cache_cursor" in source
    assert "bundle=gmem_state_cursor_0" in source
    assert "FPGAI_EXTERNAL_PERSISTENT_STATE tensor=k_cache storage=ddr words=96 cursor=fpgai_state_k_cache_cursor[0]" in source
    assert "static ap_fixed<16,6" not in source.split("fpgai_state_k_cache", 1)[0][-80:]
    assert "persistent_state_append_axis<3, 8, 1, 4" in source


def test_ddr_persistent_state_is_not_reported_as_model_gap_blocker() -> None:
    from fpgai.analysis.model_gap import audit_model_gaps
    audit = audit_model_gaps(_state_graph("ddr"), pipeline_mode="inference")
    assert audit["runtime_state_blockers"] == []
    req = next(item for item in audit["runtime_state_requirements"] if item["name"] == "k_cache")
    assert req["hls_external_state_supported"] is True


def test_kv_cache_capability_is_limited_not_falsely_full() -> None:
    cap = capability_for("KVCacheUpdate", "inference")
    assert cap.status == "limited"
    assert "BRAM/URAM" in cap.detail
    assert "DDR/host" in cap.detail
