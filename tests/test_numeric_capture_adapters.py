from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

from fpgai.validation.capture_adapters import (
    bind_hls_training_capture,
    compare_flat_f32,
    publish_python_training_reference,
)


def _write(path: Path, values) -> Path:
    np.asarray(values, dtype=np.float32).tofile(path)
    return path


def test_publish_python_reference_binds_existing_artifacts(tmp_path: Path) -> None:
    result = SimpleNamespace(
        out_dir=tmp_path,
        loss_before=2.0,
        loss_after=1.0,
        grads_flat_path=_write(tmp_path / "grads.bin", [1, 2]),
        weights_after_flat_path=_write(tmp_path / "weights.bin", [3, 4]),
        optimizer_state_after_flat_path=_write(tmp_path / "state.bin", [0.1, 0.2, 1]),
        optimizer_type="adam",
        summary_json=tmp_path / "summary.json",
    )
    path = publish_python_training_reference(
        result=result,
        output_path=tmp_path / "capture.json",
        workload_fingerprint_sha256="w",
        implementation_stack_fingerprint_sha256="i",
    )
    payload = json.loads(path.read_text())
    assert payload["producer"]["kind"] == "python_reference"
    assert payload["captures"]["post_update_loss"]["status"] == "captured"
    assert payload["captures"]["optimizer_step_after"]["path"].endswith("optimizer_step_after_ref.bin")


def test_bind_hls_capture_uses_existing_testbench_outputs(tmp_path: Path) -> None:
    _write(tmp_path / "weights_after.bin", [1, 2])
    _write(tmp_path / "gradients_after.bin", [0.1, 0.2])
    _write(tmp_path / "optimizer_state_after.bin", [0.01, 0.02, 1])
    (tmp_path / "training_loss_curve.csv").write_text("phase,epoch,loss\ninitial,0,2.0\nafter,1,1.0\n")
    path = bind_hls_training_capture(
        artifact_dir=tmp_path,
        output_path=tmp_path / "hls_capture.json",
        workload_fingerprint_sha256="w",
        implementation_stack_fingerprint_sha256="i2",
        optimizer_type="adam",
    )
    payload = json.loads(path.read_text())
    assert payload["captures"]["pre_update_loss"]["status"] == "captured"
    assert payload["captures"]["optimizer_v_after"]["layout"] == "packed_adam_state_m_v_step"


def test_compare_flat_f32_reports_numeric_metrics(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.bin", [1.0, 2.0])
    got = _write(tmp_path / "got.bin", [1.0, 2.000001])
    report = compare_flat_f32(ref, got)
    assert report["status"] == "compared"
    assert report["passed"] is True
    assert report["max_abs_error"] > 0

from fpgai.validation.capture_adapters import (
    canonical_parameter_layout,
    compare_capture_manifests,
    materialize_canonical_capture_files,
    promote_gradient_equivalence_status,
    write_numeric_equivalence_report,
)


def _manifest(path: Path, *, producer: str, fingerprint: str, bundle: Path, state: Path, step: Path | None = None) -> Path:
    captures = {
        "weights_after": {"path": str(bundle), "status": "captured", "required": True},
        "biases_after": {"path": str(bundle), "status": "captured", "required": True},
        "optimizer_m_after": {"path": str(state), "status": "captured", "required": True},
        "optimizer_v_after": {"path": str(state), "status": "captured", "required": True},
        "optimizer_step_after": {"path": str(step) if step else str(state), "status": "captured", "required": True},
    }
    payload = {
        "artifact_kind": "fpgai_numeric_capture_contract",
        "schema_version": 2,
        "workload_fingerprint_sha256": fingerprint,
        "implementation_stack_fingerprint_sha256": producer,
        "producer": {"kind": "python_reference" if producer == "ref" else "hls_csim", "id": producer},
        "captures": captures,
        "metadata": {},
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_canonical_layout_splits_parameter_bundle_and_adam_state(tmp_path: Path) -> None:
    layout = canonical_parameter_layout([
        {"name": "dense.weight", "role": "weight", "count": 4},
        {"name": "dense.bias", "role": "bias", "count": 2},
    ])
    bundle = _write(tmp_path / "parameters.bin", [1, 2, 3, 4, 5, 6])
    state = _write(tmp_path / "state.bin", list(range(12)) + [3])
    manifest = _manifest(tmp_path / "capture.json", producer="hls", fingerprint="w", bundle=bundle, state=state)
    materialize_canonical_capture_files(
        manifest_path=manifest,
        parameter_layout=layout,
        optimizer_type="adam",
    )
    payload = json.loads(manifest.read_text())
    assert np.fromfile(payload["captures"]["weights_after"]["path"], dtype=np.float32).tolist() == [1, 2, 3, 4]
    assert np.fromfile(payload["captures"]["biases_after"]["path"], dtype=np.float32).tolist() == [5, 6]
    assert np.fromfile(payload["captures"]["optimizer_m_after"]["path"], dtype=np.float32).size == 6
    assert np.fromfile(payload["captures"]["optimizer_v_after"]["path"], dtype=np.float32).size == 6
    assert np.fromfile(payload["captures"]["optimizer_step_after"]["path"], dtype=np.float32).tolist() == [3]


def test_compare_manifests_and_promote_equivalence(tmp_path: Path) -> None:
    layout = [
        {"name": "dense.weight", "role": "weight", "count": 2},
        {"name": "dense.bias", "role": "bias", "count": 1},
    ]
    ref_bundle = _write(tmp_path / "ref_params.bin", [1, 2, 3])
    got_bundle = _write(tmp_path / "got_params.bin", [1, 2, 3])
    ref_state = _write(tmp_path / "ref_state.bin", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    got_state = _write(tmp_path / "got_state.bin", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1])
    ref_step = _write(tmp_path / "ref_step.bin", [1])
    ref = _manifest(tmp_path / "ref.json", producer="ref", fingerprint="same", bundle=ref_bundle, state=ref_state, step=ref_step)
    got = _manifest(tmp_path / "got.json", producer="got", fingerprint="same", bundle=got_bundle, state=got_state)
    materialize_canonical_capture_files(manifest_path=ref, parameter_layout=layout, optimizer_type="adam")
    materialize_canonical_capture_files(manifest_path=got, parameter_layout=layout, optimizer_type="adam")
    report = compare_capture_manifests(ref, got)
    assert report["status"] == "passed"
    report_path = write_numeric_equivalence_report(ref, got, tmp_path / "numeric_report.json")
    eq_path = tmp_path / "gradient_mechanism_equivalence.json"
    eq_path.write_text(json.dumps({"numeric_equivalence_status": "capture_pending", "required_comparisons": {}}))
    promote_gradient_equivalence_status(eq_path, report_path)
    promoted = json.loads(eq_path.read_text())
    assert promoted["numeric_equivalence_status"] == "passed"
    assert promoted["claim_status"] == "numeric_equivalence_validated"


def test_compare_manifests_rejects_workload_mismatch(tmp_path: Path) -> None:
    bundle = _write(tmp_path / "params.bin", [1])
    state = _write(tmp_path / "state.bin", [0, 0, 1])
    ref = _manifest(tmp_path / "ref.json", producer="ref", fingerprint="a", bundle=bundle, state=state)
    got = _manifest(tmp_path / "got.json", producer="got", fingerprint="b", bundle=bundle, state=state)
    report = compare_capture_manifests(ref, got)
    assert report["status"] == "workload_mismatch"


def test_bind_hls_capture_discovers_nested_csim_outputs(tmp_path: Path) -> None:
    nested = tmp_path / "fpgai_hls_proj" / "sol1" / "csim" / "build"
    nested.mkdir(parents=True)
    _write(nested / "weights_after.bin", [1, 2])
    _write(nested / "grads.bin", [0.1, 0.2])
    _write(nested / "optimizer_state_after.bin", [0.01, 0.02, 0.03, 0.04, 1])
    (nested / "training_loss_curve.csv").write_text("phase,loss\ninitial,2\nafter,1\n")
    path = bind_hls_training_capture(
        artifact_dir=tmp_path,
        output_path=tmp_path / "capture.json",
        workload_fingerprint_sha256="w",
        implementation_stack_fingerprint_sha256="i",
        optimizer_type="adam",
    )
    payload = json.loads(path.read_text())
    assert payload["captures"]["weights_after"]["path"] == str(nested / "weights_after.bin")
    assert payload["captures"]["parameter_gradients"]["path"] == str(nested / "grads.bin")


def test_derive_parameter_layout_uses_ir_operator_order(monkeypatch) -> None:
    import fpgai.engine.training_graph_utils as utils
    from fpgai.validation.capture_adapters import derive_canonical_parameter_layout_from_graph

    dense = SimpleNamespace(op_type="Dense", name="dense0", inputs=["x"], outputs=["y"])
    conv = SimpleNamespace(op_type="Conv", name="conv0", inputs=["y"], outputs=["z"])
    graph = SimpleNamespace(ops=[dense, conv])
    monkeypatch.setattr(utils, "resolve_dense_arrays", lambda graph, op: (np.zeros((2, 3)), np.zeros((2,)), None, None))
    monkeypatch.setattr(utils, "resolve_conv_arrays", lambda graph, op: (np.zeros((4, 2, 3, 3)), np.zeros((4,)), None))
    layout = derive_canonical_parameter_layout_from_graph(graph)
    assert [entry["name"] for entry in layout] == [
        "dense0.weight", "dense0.bias", "conv0.weight", "conv0.bias"
    ]
    assert [entry["offset"] for entry in layout] == [0, 6, 8, 80]
    assert sum(entry["count"] for entry in layout) == 84


def test_bind_hls_capture_reads_training_epoch_curve(tmp_path: Path) -> None:
    nested = tmp_path / "fpgai_hls_proj" / "sol1" / "csim" / "build"
    nested.mkdir(parents=True)
    _write(nested / "weights_after.bin", [1, 2])
    _write(nested / "grads.bin", [0.1, 0.2])
    _write(nested / "optimizer_state_after.bin", [0.01, 0.02, 0.03, 0.04, 1])
    (nested / "training_epoch_curve.csv").write_text(
        "epoch,optimizer_updates,records_consumed,dataset_loss,accuracy,checkpoint\n"
        "0,0,0,2.0,,weights_before.bin\n"
        "1,1,1,1.0,,epoch_0001_weights.bin\n"
    )
    path = bind_hls_training_capture(
        artifact_dir=tmp_path,
        output_path=tmp_path / "capture.json",
        workload_fingerprint_sha256="w",
        implementation_stack_fingerprint_sha256="i",
        optimizer_type="adam",
    )
    payload = json.loads(path.read_text())
    assert payload["captures"]["pre_update_loss"]["status"] == "captured"
    assert payload["captures"]["post_update_loss"]["status"] == "captured"
    assert np.fromfile(payload["captures"]["pre_update_loss"]["path"], dtype=np.float32).tolist() == [2.0]
    assert np.fromfile(payload["captures"]["post_update_loss"]["path"], dtype=np.float32).tolist() == [1.0]


def test_partial_numeric_report_lists_missing_roles(tmp_path: Path) -> None:
    ref = tmp_path / "ref.json"
    got = tmp_path / "got.json"
    base = {
        "artifact_kind": "fpgai_numeric_capture_contract",
        "schema_version": 2,
        "workload_fingerprint_sha256": "same",
        "implementation_stack_fingerprint_sha256": "stack",
        "producer": {"kind": "python_reference", "id": "x"},
        "metadata": {},
    }
    ref_payload = {**base, "captures": {
        "weights_after": {"path": str(_write(tmp_path / "rw.bin", [1])), "status": "captured", "required": True},
        "pre_update_loss": {"path": str(_write(tmp_path / "rl.bin", [2])), "status": "captured", "required": True},
    }}
    got_payload = {**base, "producer": {"kind": "hls_csim", "id": "y"}, "captures": {
        "weights_after": {"path": str(_write(tmp_path / "gw.bin", [1])), "status": "captured", "required": True},
        "pre_update_loss": {"path": None, "status": "missing", "required": True},
    }}
    ref.write_text(json.dumps(ref_payload))
    got.write_text(json.dumps(got_payload))
    report = compare_capture_manifests(ref, got)
    assert report["status"] == "partial"
    assert report["completed_captures"] == ["weights_after"]
    assert report["missing_required_captures"] == ["pre_update_loss"]
    assert report["required_capture_completion"]["percentage"] == 50.0


def test_compare_flat_f32_localizes_worst_parameter_mismatch(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.bin", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    got = _write(tmp_path / "got.bin", [0.0, 1.0, 2.0, 3.0, 40.0, 5.0])
    capture = {
        "parameter_layout": [
            {"name": "dense0.weight", "layer": "dense0", "role": "weight", "offset": 0, "count": 4, "shape": [2, 2]},
            {"name": "dense0.bias", "layer": "dense0", "role": "bias", "offset": 4, "count": 2, "shape": [2]},
        ]
    }
    report = compare_flat_f32(ref, got, capture=capture)
    worst = report["worst_mismatch"]
    assert worst["flat_index"] == 4
    assert worst["parameter"] == "dense0.bias"
    assert worst["layer"] == "dense0"
    assert worst["parameter_role"] == "bias"
    assert worst["parameter_flat_index"] == 0
    assert worst["tensor_index"] == [0]
    assert worst["canonical_flat_index"] == 4
    assert worst["reference_value"] == 4.0
    assert worst["candidate_value"] == 40.0
    assert worst["abs_error"] == 36.0


def test_compare_manifest_localization_uses_role_packed_offsets(tmp_path: Path) -> None:
    ref_weights = _write(tmp_path / "ref_weights.bin", [1.0, 2.0, 3.0, 4.0])
    got_weights = _write(tmp_path / "got_weights.bin", [1.0, 2.0, 30.0, 4.0])
    layout = [
        {"name": "dense0.weight", "layer": "dense0", "role": "weight", "offset": 0, "count": 2, "shape": [1, 2]},
        {"name": "dense1.weight", "layer": "dense1", "role": "weight", "offset": 3, "count": 2, "shape": [1, 2]},
    ]
    ref_payload = {
        "artifact_kind": "fpgai_numeric_capture_contract", "schema_version": 2,
        "workload_fingerprint_sha256": "same", "implementation_stack_fingerprint_sha256": "ref",
        "producer": {"kind": "python_reference", "id": "ref"},
        "captures": {"weights_after": {"path": str(ref_weights), "status": "captured", "required": True, "parameter_layout": layout}},
        "metadata": {},
    }
    got_payload = {
        **ref_payload,
        "implementation_stack_fingerprint_sha256": "got",
        "producer": {"kind": "hls_csim", "id": "got"},
        "captures": {"weights_after": {"path": str(got_weights), "status": "captured", "required": True, "parameter_layout": layout}},
    }
    ref_manifest = tmp_path / "ref_manifest.json"
    got_manifest = tmp_path / "got_manifest.json"
    ref_manifest.write_text(json.dumps(ref_payload))
    got_manifest.write_text(json.dumps(got_payload))
    report = compare_capture_manifests(ref_manifest, got_manifest)
    worst = report["comparisons"]["weights_after"]["worst_mismatch"]
    assert worst["flat_index"] == 2
    assert worst["parameter"] == "dense1.weight"
    assert worst["parameter_flat_index"] == 0
    assert worst["tensor_index"] == [0, 0]
    assert worst["canonical_flat_index"] == 3


def test_canonicalization_attaches_gradient_layout_and_validates_adam_v(tmp_path: Path) -> None:
    layout = canonical_parameter_layout([
        {"name": "dense.weight", "layer": "dense", "role": "weight", "count": 4, "shape": [2, 2]},
        {"name": "dense.bias", "layer": "dense", "role": "bias", "count": 2, "shape": [2]},
    ])
    bundle = _write(tmp_path / "parameters.bin", [1, 2, 3, 4, 5, 6])
    gradients = _write(tmp_path / "gradients.bin", [0, 1, 2, 3, 4, 5])
    state = _write(tmp_path / "state.bin", [0.1] * 6 + [0.2] * 6 + [1])
    manifest = _manifest(tmp_path / "capture.json", producer="hls", fingerprint="w", bundle=bundle, state=state)
    payload = json.loads(manifest.read_text())
    payload["captures"]["parameter_gradients"] = {
        "path": str(gradients), "status": "captured", "required": True,
    }
    manifest.write_text(json.dumps(payload))

    materialize_canonical_capture_files(
        manifest_path=manifest,
        parameter_layout=layout,
        optimizer_type="adam",
    )
    result = json.loads(manifest.read_text())
    gradient_capture = result["captures"]["parameter_gradients"]
    assert gradient_capture["parameter_layout"] == layout
    assert gradient_capture["count"] == 6
    validation = result["metadata"]["optimizer_state_validation"]
    assert validation["finite"] is True
    assert validation["v_nonnegative"] is True
    assert validation["step_present"] is True


def test_adam_state_validation_detects_negative_second_moment(tmp_path: Path) -> None:
    layout = canonical_parameter_layout([
        {"name": "dense.weight", "role": "weight", "count": 2},
    ])
    bundle = _write(tmp_path / "parameters.bin", [1, 2])
    state = _write(tmp_path / "state.bin", [0.1, 0.2, -0.3, 0.4, 1])
    manifest = _manifest(tmp_path / "capture.json", producer="hls", fingerprint="w", bundle=bundle, state=state)
    materialize_canonical_capture_files(
        manifest_path=manifest,
        parameter_layout=layout,
        optimizer_type="adam",
    )
    validation = json.loads(manifest.read_text())["metadata"]["optimizer_state_validation"]
    assert validation["v_nonnegative"] is False
    assert validation["v_min"] < 0.0


def test_direct_training_numeric_report_writes_execution_trace(tmp_path: Path) -> None:
    ref_value = _write(tmp_path / "ref_grad.bin", [1.0])
    got_value = _write(tmp_path / "got_grad.bin", [2.0])
    base = {
        "artifact_kind": "fpgai_numeric_capture_contract",
        "schema_version": 2,
        "workload_fingerprint_sha256": "same",
        "implementation_stack_fingerprint_sha256": "same-stack",
        "metadata": {},
    }
    ref = tmp_path / "ref_training.json"
    got = tmp_path / "got_training.json"
    ref.write_text(json.dumps({
        **base,
        "producer": {"kind": "python_reference", "id": "ref"},
        "captures": {"parameter_gradients": {
            "path": str(ref_value), "status": "captured", "required": True,
        }},
    }))
    got.write_text(json.dumps({
        **base,
        "producer": {"kind": "hls_csim", "id": "got"},
        "captures": {"parameter_gradients": {
            "path": str(got_value), "status": "captured", "required": True,
        }},
    }))

    report_path = write_numeric_equivalence_report(ref, got, tmp_path / "numeric_equivalence_report.json")
    trace_path = tmp_path / "training_execution_trace.json"

    assert report_path.exists()
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text())
    assert trace["first_divergence"]["stage"] == "parameter_gradient"


def test_inference_only_numeric_report_does_not_write_training_trace(tmp_path: Path) -> None:
    ref_value = _write(tmp_path / "ref_out.bin", [1.0])
    got_value = _write(tmp_path / "got_out.bin", [1.0])
    base = {
        "artifact_kind": "fpgai_numeric_capture_contract",
        "schema_version": 2,
        "workload_fingerprint_sha256": "same",
        "implementation_stack_fingerprint_sha256": "same-stack",
        "metadata": {},
    }
    ref = tmp_path / "ref_inference.json"
    got = tmp_path / "got_inference.json"
    ref.write_text(json.dumps({
        **base,
        "producer": {"kind": "python_reference", "id": "ref"},
        "captures": {"model_output": {
            "path": str(ref_value), "status": "captured", "required": True,
        }},
    }))
    got.write_text(json.dumps({
        **base,
        "producer": {"kind": "hls_csim", "id": "got"},
        "captures": {"model_output": {
            "path": str(got_value), "status": "captured", "required": True,
        }},
    }))

    write_numeric_equivalence_report(ref, got, tmp_path / "numeric_equivalence_report.json")
    assert not (tmp_path / "training_execution_trace.json").exists()
