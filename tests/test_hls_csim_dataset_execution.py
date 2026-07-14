from fpgai.backends.hls.emit.csim_tcl import emit_csim_tcl


def test_csim_tcl_passes_dataset_execution_record_path() -> None:
    text = emit_csim_tcl(
        top_name="deeplearn",
        part="xck26-sfvc784-2LV-c",
        input_bin_path="/tmp/dataset/inputs.bin",
        output_bin_path="/tmp/output.bin",
        execution_record_path="/tmp/reports/hls_dataset_execution.json",
    )
    assert 'csim_design -argv "/tmp/dataset/inputs.bin /tmp/output.bin /tmp/reports/hls_dataset_execution.json"' in text
