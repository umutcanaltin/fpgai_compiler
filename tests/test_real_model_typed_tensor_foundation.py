import numpy as np

from fpgai.analysis.model_gap import audit_model_gaps
from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.capabilities.capabilities import capability_for
from fpgai.ir.graph import Graph
from fpgai.layers.registry import get_layer_capability
from fpgai.operators import get_builtin_operator_contract


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


def _embedding_graph(index_dtype: str = "int64") -> Graph:
    g = Graph("typed_embedding")
    g.inputs = ["token_ids"]
    g.outputs = ["embedding"]
    g.add_tensor("token_ids", (2,), index_dtype)
    g.add_tensor("table", (8, 4), "float32")
    g.add_tensor("embedding", (2, 4), "float32")
    g.constants["table"] = np.arange(32, dtype=np.float32).reshape(8, 4)
    g.add_op("Gather", ["table", "token_ids"], ["embedding"], name="embedding_lookup", attrs={"axis": 0})
    return g


def test_integer_index_tensor_uses_integer_hls_buffer_and_transport() -> None:
    g = _embedding_graph("int64")
    allocation = build_hls_buffer_allocation(g, raw_cfg=_cfg())
    slot = next(slot for slot in allocation["slots"] if "token_ids" in slot["tensors"])
    assert slot["cpp_type"] == "ap_int<64>"

    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "typedef ap_axis<64, 0, 0, 0> axis_t;" in source
    assert "FPGAI_INPUT_SEGMENT tensor=token_ids words=2 bits=64" in source
    assert "fpgai_unpack_axis_value<ap_int<64>, 64>" in source
    assert "gather_rows<8, 4, 2" in source


def test_normal_float_only_graph_keeps_32_bit_axis_contract() -> None:
    g = Graph("float_only")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor("x", (4,), "float32")
    g.add_tensor("y", (4,), "float32")
    g.add_op("Identity", ["x"], ["y"], name="copy")
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "typedef ap_axis<32, 0, 0, 0> axis_t;" in source
    assert "FPGAI_INPUT_SEGMENT tensor=x words=4 bits=16" in source


def test_model_gap_reports_integer_and_index_tensor_semantics() -> None:
    report = audit_model_gaps(_embedding_graph("int64"), pipeline_mode="inference")
    assert [item["name"] for item in report["integer_tensors"]] == ["token_ids"]
    assert [item["name"] for item in report["index_tensors"]] == ["token_ids"]
    assert report["typed_tensor_blockers"] == []
    gather = report["layers"][0]
    index_record = next(item for item in gather["inputs"] if item["name"] == "token_ids")
    assert index_record["dtype_kind"] == "integer"
    assert index_record["index_tensor"] is True


def test_non_integer_gather_index_is_reported_as_typed_tensor_blocker() -> None:
    report = audit_model_gaps(_embedding_graph("float32"), pipeline_mode="inference")
    assert report["typed_tensor_blockers"] == ["token_ids"]


def test_split_is_an_ecosystem_visible_frontend_canonicalization_contract() -> None:
    capability = capability_for("Split", "inference")
    assert capability.status == "limited"
    assert "Slice" in capability.detail
    layer = get_layer_capability("Split")
    assert layer.inference_supported is True
    contract = get_builtin_operator_contract("Split")
    assert contract.canonical_op_type == "Split"
    assert any(attribute.name == "axis" for attribute in contract.attributes)
