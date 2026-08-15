from fpgai.analysis.hls_bottleneck_diagnostics import parse_hls_bottlenecks_text


def test_hls_memory_port_ii_violation_is_normalized():
    text = """
WARNING: [HLS 200-885] The II Violation in module 'deeplearn_Pipeline_VITIS_LOOP_180_5' (loop 'VITIS_LOOP_180_5'): Unable to schedule 'load' operation 16 bit ('value', ./src/deeplearn.cpp:190) on array 'layer_2_out' due to limited memory ports (II = 1). Please consider using a memory core with more ports or partitioning the array 'layer_2_out'.
INFO: [HLS 200-1470] Pipelining result : Target II = 1, Final II = 2, Depth = 3, loop 'VITIS_LOOP_180_5'
"""
    report = parse_hls_bottlenecks_text(text)
    assert report["warning_count"] == 1
    assert report["ii_violation_count"] == 1
    item = report["bottlenecks"][0]
    assert item["category"] == "memory_port_contention"
    assert item["resource"] == "layer_2_out"
    assert item["source_file"] == "./src/deeplearn.cpp"
    assert item["source_line"] == 190
    assert item["requested_ii"] == 1
    assert item["achieved_ii"] == 2
    assert "optimization.parallel.partition_factor" in item["applicable_yaml_mechanisms"]


def test_resource_provenance_resolves_generated_buffer_to_ir_tensor():
    from fpgai.analysis.hls_bottleneck_diagnostics import parse_hls_bottlenecks_text

    text = """WARNING: [HLS 200-885] The II Violation in module 'deeplearn_loop' (loop 'LOOP_X'): Unable to schedule 'load' operation on array 'layer_2_out' due to limited memory ports (II = 1).\nINFO: [HLS 200-1470] Pipelining result : Target II = 1, Final II = 2, Depth = 3, loop 'LOOP_X'\n"""
    liveness = {
        "tensors": {
            "output": {"producer": "sigmoid_0", "consumers": []},
        }
    }
    report = parse_hls_bottlenecks_text(
        text,
        tensor_liveness=liveness,
        resource_provenance={"layer_2_out": {"tensors": ["output"]}},
    )
    item = report["bottlenecks"][0]
    assert item["affected_tensor"] == "output"
    assert item["producer"] == "sigmoid_0"
