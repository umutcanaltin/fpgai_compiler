from pathlib import Path

from fpgai.backends.hls.emit import top_train_cpp


def test_training_storage_comments_do_not_emit_literal_backslash_newline():
    source = (
        "static wgt_t W_conv0[2] = { 0.0f, 1.0f };\n"
        "static bias_t B_conv0[1] = { 0.0f };\n"
        "static grad_wgt_t dW_conv0[2];\n"
        "static grad_bias_t dB_conv0[1];\n"
    )

    updated = top_train_cpp._fpgai_insert_training_storage_bindings(
        source,
        compile_plan=None,
    )

    assert "\\n#pragma" not in updated
    assert ";\\n" not in updated
    assert "FPGAI training storage binding requested" in updated


def test_training_storage_comments_do_not_emit_file_scope_bind_storage():
    source = (
        "static wgt_t W_dense0[2] = { 0.0f, 1.0f };\n"
        "static bias_t B_dense0[1] = { 0.0f };\n"
        "static grad_wgt_t dW_dense0[2];\n"
        "static grad_bias_t dB_dense0[1];\n"
    )

    updated = top_train_cpp._fpgai_insert_training_storage_bindings(
        source,
        compile_plan=None,
    )

    assert "#pragma HLS BIND_STORAGE" not in updated
    assert "file-scope BIND_STORAGE disabled" in updated
