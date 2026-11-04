"""HW-RL-02 source/native boundary: no mocks may fabricate HLS qualification.

These CPU tests call the original emitter/CSim package builder only.
They never launch licensed Vitis or change historical Gym case folders.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
from rl_environment.hardware_space import (
    HardwareEdit, compile_state, initial_state, transition,
)
from rl_environment import hardware_native as native


PART = "xc7z020clg400-1"
CAPS = (20000, 30000, 100, 80, 0)


@pytest.fixture
def task(tmp_path):
    pytest.importorskip("onnx")
    from examples.make_neural_network_onnx import save_model
    path = save_model("cnn_pool_dense", tmp_path / "toy.onnx")
    model = load_onnx(str(path))
    input_shape = model.tensor(model.inputs[0]).shape
    samples = np.zeros((2, *input_shape), dtype=np.float32)
    samples[1].fill(0.125)
    state = initial_state(
        model, part=PART, clock_ns=10.0,
        resource_caps=CAPS, max_absolute_error=0.1,
    )
    return path, state, samples


def _prepare_without_vendor(root, state, samples):
    """Create only real compiler source, numerical TB and signed task request."""
    plan = native.planned_case(state, samples)
    artifact = compile_state(state).emission
    write_vitis_package(
        str(root), state.model, artifact, samples, part=state.part,
        clock_ns=state.clock_ns, tolerance=state.max_absolute_error,
        clock_uncertainty_ns=plan["explicit_clock_uncertainty_ns"],
    )
    (root / "hardware_action.json").write_text(json.dumps(plan) + "\n")
    return plan


def test_source_plan_identifies_exact_model_and_samples_without_vitis(task):
    _, state, samples = task
    plan = native.planned_case(state, samples)
    assert plan["status"] == "SOURCE_READY_NOT_NATIVE_EXECUTED"
    assert plan["state_sha256"] == state.state_sha256
    assert plan["sample_shape"] == list(samples.shape)
    assert len(plan["sample_sha256"]) == 64
    assert plan["cpp_sha256"] == compile_state(state).emission.manifest["cpp_sha256"]
    assert plan["explicit_clock_uncertainty_ns"] == 2.7
    assert plan["resource_caps"]["lut"] == CAPS[0]
    assert plan["native_vitis_calls"] == 0
    assert plan["reward"] is None


def test_source_only_cli_plans_real_cnn_and_creates_no_case(task, tmp_path, capsys):
    path, state, samples = task
    sample_file = tmp_path / "samples.npy"
    np.save(sample_file, samples)
    case = tmp_path / "not_created"
    args = [
        "--onnx", str(path), "--samples", str(sample_file),
        "--part", PART, "--clock-ns", "10",
        "--resource-caps", *(str(x) for x in CAPS),
        "--max-absolute-error", "0.1",
        "--output", str(case),
    ]
    assert native.main(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["state_sha256"] == state.state_sha256
    assert data["reward"] is None
    assert data["native_vitis_calls"] == 0
    assert not case.exists()
    assert sample_file.is_file()


def test_native_resolution_is_0_01ns_not_a_discrete_clock_grid(task):
    _, state, samples = task
    assert native.planned_case(state, samples)["explicit_clock_uncertainty_ns"] == 2.7
    edited = transition(state, HardwareEdit("clock", "clock_ns", 13.125))
    assert edited.status == "SOURCE_ACCEPTED"
    with pytest.raises(ValueError, match="0.01ns-canonical"):
        native.planned_case(edited.next, samples)
    suitable = transition(state, HardwareEdit("clock", "clock_ns", 13.12))
    assert suitable.status == "SOURCE_ACCEPTED"
    assert native.planned_case(suitable.next, samples)["requested_clock_ns"] == 13.12


@pytest.mark.parametrize("mutation", ["float64", "shape", "nan", "noncontiguous", "empty"])
def test_native_samples_do_not_implicitly_change_numeric_cohort(task, mutation):
    _, state, samples = task
    if mutation == "float64":
        bad = samples.astype(np.float64)
    elif mutation == "shape":
        bad = samples[:, :, :, :, :-1].copy()
    elif mutation == "nan":
        bad = samples.copy()
        bad.flat[0] = np.nan
    elif mutation == "noncontiguous":
        bad = np.asfortranarray(samples)
    else:
        bad = samples[:0]
    with pytest.raises(ValueError, match="samples|numerical"):
        native.planned_case(state, bad)


def test_independent_audit_never_qualifies_e1_without_vitis(task, tmp_path):
    _, state, samples = task
    case = tmp_path / "no_vendor"
    _prepare_without_vendor(case, state, samples)
    result = native.audit_native_case(case, state, samples)
    assert result["qualified"] is False
    assert result["hls_estimated_latency_ns"] is None
    assert result["hls_worst_case_cycles"] is None
    assert "ORIGINAL_NATIVE_E3_CSIM_SOURCE_NOT_QUALIFIED" in result["gate_failures"]
    assert result["original_hls_resource_consumed"] == {
        "lut": None, "ff": None, "dsp": None,
        "bram_18k": None, "uram": None,
    }
    assert result["reward"] is None
    assert result["native_calls_by_audit"] == 0


@pytest.mark.parametrize("kind", ["cpp", "tb", "manifest", "action", "tile", "samples"])
def test_independent_audit_rejects_tampered_source_or_task(task, tmp_path, kind):
    _, state, samples = task
    case = tmp_path / "attested_case"
    _prepare_without_vendor(case, state, samples)
    audited_samples = samples
    if kind == "cpp":
        (case / "model.cpp").write_text((case / "model.cpp").read_text() + "\n// tampered\n")
    elif kind == "tb":
        (case / "tb.cpp").write_text((case / "tb.cpp").read_text() + "\n// tampered\n")
    elif kind == "manifest":
        path = case / "manifest.json"
        meta = json.loads(path.read_text())
        meta["clock_ns"] = 11
        path.write_text(json.dumps(meta))
    elif kind == "action":
        path = case / "hardware_action.json"
        record = json.loads(path.read_text())
        record["resource_caps"]["dsp"] = 0
        path.write_text(json.dumps(record))
    elif kind == "tile":
        path = case / "hardware_action.json"
        record = json.loads(path.read_text())
        record["complete_architecture"][0]["request"]["tile"] = [1, 2, 3]
        path.write_text(json.dumps(record))
    else:
        audited_samples = samples.copy()
        audited_samples.flat[0] = 0.3333
    with pytest.raises(ValueError, match="differ|mismatch|reproduce"):
        native.audit_native_case(case, state, audited_samples)


def test_fabricated_vendor_receipt_and_missing_xml_never_qualify(task, tmp_path):
    _, state, samples = task
    case = tmp_path / "fake_vendor"
    _prepare_without_vendor(case, state, samples)
    (case / "vitis_invocation.json").write_text(
        json.dumps({"tool": "vitis_hls", "returncode": 0,
                    "completed_successfully": True})
    )
    result = native.audit_native_case(case, state, samples)
    assert result["qualified"] is False
    assert result["evidence"] != "E3-HLS-synthesis-report"
    assert result["hls_estimated_latency_ns"] is None


def test_preflight_stale_source_refuses_allocating_case(task, tmp_path, monkeypatch):
    _, state, samples = task
    case = tmp_path / "untouched_case"
    def stale():
        raise RuntimeError("outdated compiler")
    monkeypatch.setattr(native, "verify_local_compiler_package", stale)
    with pytest.raises(RuntimeError, match="outdated compiler"):
        native.execute_native_case(case, state, samples)
    assert not case.exists()


def test_fake_runner_never_returns_qualified_e3_without_original_logs(
        task, tmp_path, monkeypatch):
    _, state, samples = task
    case = tmp_path / "fake_runner_no_reports"
    monkeypatch.setattr(
        native, "verify_local_compiler_package",
        lambda: {"source_package_matches_checkout": True},
    )
    import shutil
    monkeypatch.setattr(shutil, "which", lambda x: "/fake/vitis_hls")
    mock_calls = []
    def no_vendor(*a, **kw):
        mock_calls.append((a, kw))
        return {"returncode": 0, "completed_successfully": True}
    monkeypatch.setattr(native, "run_vitis", no_vendor)
    report = native.execute_native_case(case, state, samples)
    assert len(mock_calls) == 1
    assert report["qualified"] is False
    assert report["hls_estimated_latency_ns"] is None
    assert report["reward"] is None
    assert (case / "hardware_action.json").is_file()
    assert (case / "model.cpp").is_file()
    with pytest.raises(FileExistsError, match="never overwrite"):
        native.execute_native_case(case, state, samples)


def test_native_cli_action_edits_change_real_hls_source_state(task, tmp_path, capsys):
    path, state, samples = task
    sample_file = tmp_path / "samples.npy"
    np.save(sample_file, samples)
    idx = next(i for i, op in enumerate(state.model.operators) if op.kind == "Conv")
    actions = tmp_path / "actions.json"
    actions.write_text(json.dumps([{
        "scope": "operator", "field": "pes",
        "operator_index": idx, "value": 2,
    }]))
    args = [
        "--onnx", str(path), "--samples", str(sample_file),
        "--part", PART, "--clock-ns", "10",
        "--resource-caps", *(str(x) for x in CAPS),
        "--max-absolute-error", "0.1", "--edits-json", str(actions),
    ]
    assert native.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state_sha256"] != state.state_sha256
    assert result["complete_architecture"][idx]["request"]["pes"] == 2
    assert result["reward"] is None



def test_complete_action_plan_is_json_native_before_persisting(task, tmp_path):
    """Regression: asdict(ArchitectureRequest).tile is tuple, JSON uses list.

    The strict exact audit comparison must accept real saved plans without
    dropping nested hardware action fields or bypassing hash checks.
    """
    _, state, samples = task
    plan = native.planned_case(state, samples)
    assert json.loads(json.dumps(plan, sort_keys=True)) == plan
    assert len(plan["complete_architecture"]) == len(state.requests)
    for row in plan["complete_architecture"]:
        assert row["request"]["tile"] == []
        assert type(row["request"]["tile"]) is list
    case = tmp_path / "json_roundtrip_only"
    _prepare_without_vendor(case, state, samples)
    assert json.loads((case / "hardware_action.json").read_text()) == plan
    # The baseline contains no vendor artifacts, so it must never qualify.
    audited = native.audit_native_case(case, state, samples)
    assert audited["qualified"] is False
    assert audited["hls_estimated_latency_ns"] is None
    assert audited["reward"] is None


def test_nonempty_conv_tile_survives_json_case_audit(task, tmp_path):
    """Multi-axis Conv tuple becomes a list on disk without losing exact shape."""
    _, state, samples = task
    idx = next(i for i, op in enumerate(state.model.operators) if op.kind == "Conv")
    from rl_environment.hardware_space import knob_inventory
    tile_candidates = knob_inventory(state)["operators"][idx]["active_operator_knobs"]["tile"]
    tile = next(candidate for candidate in tile_candidates if candidate)
    result = transition(state, HardwareEdit("operator", "tile", tile, idx))
    assert result.status == "SOURCE_ACCEPTED", result.reason
    selected = result.next
    plan = native.planned_case(selected, samples)
    assert plan["complete_architecture"][idx]["request"]["tile"] == tile
    assert type(plan["complete_architecture"][idx]["request"]["tile"]) is list
    case = tmp_path / "nonempty_tile_no_vendor"
    _prepare_without_vendor(case, selected, samples)
    assert json.loads((case / "hardware_action.json").read_text()) == plan
    audit = native.audit_native_case(case, selected, samples)
    assert audit["qualified"] is False
    assert audit["source_cpp_sha256"] == plan["cpp_sha256"]
    assert audit["hls_estimated_latency_ns"] is None
