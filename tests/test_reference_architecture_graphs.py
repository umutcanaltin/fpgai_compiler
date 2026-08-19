from __future__ import annotations

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.benchmark.model_graphs import (
    build_llm_like_decoder_graph,
    build_single_stage_detector_graph,
    build_yolo_like_multiscale_detector_graph,
)


def _cfg():
    return {"numerics":{"defaults":{"activation":{"type":"ap_fixed","total_bits":16,"int_bits":6},"accum":{"type":"ap_fixed","total_bits":32,"int_bits":12}}},"targets":{"hls":{"control_protocol":"s_axilite"}}}


def test_yolo_like_reference_architecture_reaches_generic_hls_and_detection_contracts() -> None:
    g = build_yolo_like_multiscale_detector_graph()
    assert g.metadata["reference_architecture"] == "yolo_like_multiscale_detector"
    assert g.semantics.runtime_contract["detection_output"]["nms_required"] is True
    assert g.semantics.runtime_contract["detection_decode"]["dfl_bins"] == 8
    src = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    for token in ("conv2d<", "conv2d_grouped<", "resize_nearest_nchw<", "concat_axis<", "softmax_rows<", "reduce_sum_axis_typed<"):
        assert token in src


def test_generic_single_stage_detector_reaches_hls_and_explicit_nms_partition() -> None:
    g = build_single_stage_detector_graph()
    contract = g.semantics.runtime_contract["detection_output"]
    assert contract["postprocess_partition"] == "ps_or_host"
    assert contract["nms_required"] is True
    src = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert src.count("conv2d<") >= 2


def test_llm_like_gqa_decoder_reaches_hls_with_state_rope_swiglu_and_tied_weights() -> None:
    g = build_llm_like_decoder_graph(num_heads=3, num_kv_heads=1, cache_storage="bram")
    session = g.semantics.runtime_contract["autoregressive_session"]
    assert session["supported_modes"] == ["prefill", "decode"]
    assert session["tied_parameter_groups"][0]["physical_owner"] == "embedding"
    src = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    for token in ("gather_rows<", "rotary_embedding_heads<", "multi_head_attention_cached_serialized<", "silu_vector<", "FPGAI_TIED_PARAMETER"):
        assert token in src


def test_llm_like_mha_variant_uses_same_generic_builder_and_contract() -> None:
    g = build_llm_like_decoder_graph(num_heads=3, num_kv_heads=3, cache_storage="uram")
    assert g.metadata["reference_architecture"] == "llm_like_mha_decoder"
    assert g.semantics.runtime_contract["autoregressive_session"]["cache_storage"] == "uram"
