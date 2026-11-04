"""Real ONNX initializer -> Model IR -> HLS C++ constant integrity checks.

Source-only: no Vitis, Vivado, board hardware or GitHub Actions involved.
"""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from fpgai_research.toolchains.weights_audit import audit_weight_constants, main


def _make_case(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.model_ir.contracts import ArchitectureRequest
    from fpgai_research.model_ir.lowering import lower
    from fpgai_research.codegen.hls.emitter import emit_hls

    weights = np.array(
        [[0.125, -0.25], [0.0625, 0.5], [-0.75, 0.3125]], dtype=np.float32,
    )
    bias = np.array([0.25, -0.375], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "weights"], ["mm"], name="mm"),
            helper.make_node("Add", ["mm", "bias"], ["out"], name="bias"),
        ],
        "audited-constant-graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])],
        initializer=[
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    proto = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    path = tmp_path / "trained_model.onnx"
    onnx.save(proto, str(path))
    model = load_onnx(str(path))
    emitted = emit_hls(lower(resolve(model, ArchitectureRequest())))
    case = tmp_path / "case"
    case.mkdir()
    cpp = case / "model.cpp"
    cpp.write_text(emitted.cpp)
    (case / "manifest.json").write_text(json.dumps(emitted.manifest))
    (case / "inputs_provenance.json").write_text(json.dumps({
        "onnx_sha256": sha256(path.read_bytes()).hexdigest(),
    }))
    return path, case, cpp


def test_onxx_initializer_values_reach_generated_hls_literals(tmp_path):
    onnx, case, cpp = _make_case(tmp_path)
    report = audit_weight_constants(onnx, case)
    assert report["status"] == "ONNX_FLOAT32_CONSTANT_LITERALS_VERIFIED"
    assert report["float32_constant_tensor_count"] == 2
    assert report["float32_constant_value_count"] == 8
    assert report["constant_literals_match_imported_onnx_values"] is True
    assert report["model_ir_graph_and_constant_hash_matches_case"] is True
    assert report["raw_weight_values_printed"] is False
    assert report["rtl_memory_cell_values_independently_verified"] is False
    assert report["physical_fpga_weights_verified"] is False
    assert report["hls_fixed_point_format"] == "ap_fixed<16,6,AP_RND,AP_SAT>"
    assert "data_t(0.125)" in cpp.read_text()
    assert "data_t(-0.375)" in cpp.read_text()


def test_wrong_onnx_file_is_rejected(tmp_path):
    onnx, case, cpp = _make_case(tmp_path)
    onnx.write_bytes(onnx.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="ONNX file differs"):
        audit_weight_constants(onnx, case)


def test_modified_cpp_source_cannot_pass_original_receipt(tmp_path):
    onnx, case, cpp = _make_case(tmp_path)
    src = cpp.read_text()
    assert "data_t(0.125)" in src
    cpp.write_text(src.replace("data_t(0.125)", "data_t(0.5)", 1))
    with pytest.raises(ValueError, match="Generated C\\+\\+ has changed"):
        audit_weight_constants(onnx, case)


def test_modified_weight_with_updated_cpp_hash_still_fails_value_check(tmp_path):
    onnx, case, cpp = _make_case(tmp_path)
    cpp.write_text(cpp.read_text().replace("data_t(0.125)", "data_t(0.5)", 1))
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["cpp_sha256"] = sha256(cpp.read_bytes()).hexdigest()
    (case / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="initializer mismatch"):
        audit_weight_constants(onnx, case)


def test_modified_model_ir_fingerprint_blocks_forged_source(tmp_path):
    onnx, case, cpp = _make_case(tmp_path)
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["functional_sha256"] = "f" * 64
    (case / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="graph or constant payload"):
        audit_weight_constants(onnx, case)


def test_unexpected_source_initializer_is_rejected(tmp_path):
    onnx, case, cpp = _make_case(tmp_path)
    cpp.write_text(cpp.read_text() + "\nstatic const data_t tensor_999[1] = {data_t(1)};\n")
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["cpp_sha256"] = sha256(cpp.read_bytes()).hexdigest()
    (case / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Number of C\\+\\+ constant declarations"):
        audit_weight_constants(onnx, case)


def test_cli_reports_only_metadata_and_is_read_only(tmp_path, capsys):
    onnx, case, cpp = _make_case(tmp_path)
    files = {p.name: p.read_bytes() for p in case.iterdir()}
    assert main(["--onnx", str(onnx), "--case", str(case)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["float32_constant_value_count"] == 8
    assert payload["raw_weight_values_printed"] is False
    assert {p.name: p.read_bytes() for p in case.iterdir()} == files


def test_cli_rejects_wrong_source_and_does_not_change_case(tmp_path, capsys):
    onnx, case, cpp = _make_case(tmp_path)
    other = tmp_path / "wrong.onnx"
    other.write_bytes(onnx.read_bytes() + b"\n")
    assert main(["--onnx", str(other), "--case", str(case)]) == 2
    assert "differs" in capsys.readouterr().err
    assert not (case / "weights_audit.json").exists()



def test_changing_a_weight_changes_model_output_and_emitted_cpp(tmp_path):
    """The weight values are computationally used, not only stored as metadata."""
    onnx_path, case, cpp = _make_case(tmp_path)
    import onnx
    from onnx import numpy_helper
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.validation.reference import evaluate
    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.model_ir.contracts import ArchitectureRequest
    from fpgai_research.model_ir.lowering import lower
    from fpgai_research.codegen.hls.emitter import emit_hls

    sample = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
    original = load_onnx(str(onnx_path))
    y_before = evaluate(original, {"x": sample})["out"]
    cpp_before = cpp.read_text()

    proto = onnx.load(str(onnx_path))
    initializers = list(proto.graph.initializer)
    original_w = numpy_helper.to_array(initializers[0])
    changed_w = original_w.copy()
    changed_w[0, 0] += np.float32(0.5)
    proto.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(changed_w, name=initializers[0].name)
    )
    changed_path = tmp_path / "new_weights.onnx"
    onnx.save(proto, str(changed_path))
    changed = load_onnx(str(changed_path))
    y_after = evaluate(changed, {"x": sample})["out"]
    emitted_after = emit_hls(lower(resolve(changed, ArchitectureRequest())))
    assert not np.allclose(y_before, y_after)
    assert cpp_before != emitted_after.cpp
    # An old source case cannot be audited as though it used these weights.
    with pytest.raises(ValueError, match="ONNX file differs"):
        audit_weight_constants(changed_path, case)
