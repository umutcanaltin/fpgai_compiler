"""Source-only regression for complete typed RL architecture / clock actions.

These tests do NOT launch Vitis/Vivado, train a policy or claim FPGA speedups.
"""
from __future__ import annotations

import json

import pytest

from fpgai_research.architecture.resolver import TargetLimits
from fpgai_research.frontend.onnx import load_onnx
from rl_environment.hardware_space import (
    ARCHITECTURE_FIELDS, OPERATOR_FIELDS, FUTURE_FIELDS,
    HardwareEdit, HardwarePlanningEnvironment, action_mask,
    action_features, candidate_actions, graph_observation, initial_state,
    knob_inventory, native_evaluation_plan, transition, validate_source, main,
)

PART = "xc7z020clg400-1"


@pytest.fixture()
def cnn(tmp_path):
    pytest.importorskip("onnx")
    from examples.make_neural_network_onnx import save_model
    return load_onnx(str(save_model("cnn_pool_dense", tmp_path / "cnn.onnx")))


@pytest.fixture()
def softmax_cnn(tmp_path):
    pytest.importorskip("onnx")
    from examples.make_neural_network_onnx import save_model
    return load_onnx(str(save_model("cnn_two_conv_softmax", tmp_path / "softmax.onnx")))


def test_rl_inventory_exposes_all_eight_real_compiler_fields(cnn):
    state = initial_state(cnn, part=PART, limits=TargetLimits(max_parallel_units=8))
    inv = knob_inventory(state)
    assert ARCHITECTURE_FIELDS == (
        "precision_bits", "integer_bits", "parallelism",
        "pes", "simd", "serialization", "tile", "pipeline_ii",
    )
    assert set(OPERATOR_FIELDS) == set(ARCHITECTURE_FIELDS[2:])
    assert set(inv["model_global_numeric"]) == {
        "precision_bits", "integer_bits", "current", "reason",
    }
    assert inv["part_is_fixed_task_context_not_rl_action"] is True
    assert inv["candidate_domains_exhaustive"] is False
    assert inv["global_continuous_clock"]["resolution_ns"] == .001
    assert inv["global_continuous_clock"]["max_ns"] == 32
    assert inv["global_continuous_clock"]["min_ns"] == 2
    assert inv["native_tool_calls"] == 0
    assert len(inv["operators"]) == len(cnn.operators)
    for index, op in enumerate(cnn.operators):
        fields = inv["operators"][index]["active_operator_knobs"]
        assert set(fields) == set(OPERATOR_FIELDS) if op.kind in (
            "MatMul", "Gemm", "Conv"
        ) else fields == {}
    assert set(FUTURE_FIELDS).issubset({
        row["field"] for row in inv["declared_not_yet_actionable"]
    })


def test_rl_baseline_real_compiler_source_attestation(cnn):
    state = initial_state(cnn, part=PART)
    receipt = validate_source(state)
    assert receipt["state_sha256"] == state.state_sha256
    assert receipt["functional_sha256"] == state.functional_sha256
    assert len(receipt["cpp_sha256"]) == 64
    assert receipt["source_compilable"] is True
    assert receipt["hardware_qor_known"] is False
    assert receipt["native_vitis_calls"] == 0


def test_multiple_operator_kinds_and_weights_hidden(softmax_cnn):
    state = initial_state(softmax_cnn, part=PART)
    observation = graph_observation(state)
    assert observation["reward"] is None
    assert observation["native_metrics"] is None
    assert observation["no_raw_model_weights"] is True
    assert len(observation["nodes"]) == (
        len(softmax_cnn.tensors) + len(softmax_cnn.operators)
    )
    kinds = {n["kind"] for n in observation["nodes"] if n["type"] == "operator"}
    assert {"Conv", "Relu", "Flatten", "Gemm", "Softmax"} <= kinds
    tensors = [n for n in observation["nodes"] if n["type"] == "tensor"]
    assert any(n["constant"] for n in tensors)
    assert all("data" not in n and "weight_values" not in n for n in tensors)
    assert all(set(edge) <= {"src", "dst", "type", "port"}
               for edge in observation["edges"])
    # The private functional digest is kept in the source receipt, not
    # leaked into a numeric graph feature.
    assert state.functional_sha256 not in json.dumps(observation)


def test_per_operator_knob_is_immutable_and_source_compiled(cnn):
    state = initial_state(cnn, part=PART)
    idx = next(i for i, o in enumerate(cnn.operators) if o.kind == "Conv")
    edit = HardwareEdit("operator", "pes", 2, idx)
    result = transition(state, edit)
    assert result.status == "SOURCE_ACCEPTED", result.reason
    assert result.next.requests[idx].pes == 2
    assert state.requests[idx].pes == 1
    assert result.next.state_sha256 != state.state_sha256
    assert result.receipt()["native_vitis_calls"] == 0
    assert result.receipt()["reward"] is None
    assert result.receipt()["hardware_feasibility"] == "UNKNOWN"
    assert result.receipt()["functional_sha256"] == state.functional_sha256
    assert len(result.source_cpp_sha256) == 64


def test_conv_tile_is_rank_seven_and_only_output_axes_change(cnn):
    state = initial_state(cnn, part=PART)
    idx = next(i for i, o in enumerate(cnn.operators) if o.kind == "Conv")
    tiles = knob_inventory(state)["operators"][idx]["active_operator_knobs"]["tile"]
    shape = tuple(knob_inventory(state)["operators"][idx]["iteration_shape"])
    assert len(shape) == 7
    assert [] in tiles
    assert list(shape) in tiles
    assert all(len(t) == 0 or (len(t) == 7 and
                               all(t[j] == shape[j] for j in (0, 4, 5, 6)))
               for t in tiles)
    proposed = next(t for t in tiles if t and tuple(t) != shape)
    result = transition(state, HardwareEdit("operator", "tile", proposed, idx))
    assert result.status == "SOURCE_ACCEPTED", result.reason
    assert result.next.requests[idx].tile == tuple(proposed)


def test_global_numeric_action_preserves_complete_uniform_format(cnn):
    state = initial_state(cnn, part=PART)
    result = transition(state, HardwareEdit("numeric", "precision_bits", 12))
    assert result.status == "SOURCE_ACCEPTED", result.reason
    assert all(r.precision_bits == 12 for r in result.next.requests)
    assert all(r.integer_bits == 6 for r in result.next.requests)
    assert state.requests[0].precision_bits == 16
    bad = transition(result.next, HardwareEdit("numeric", "integer_bits", 16))
    assert bad.status == "REJECTED"
    assert bad.next is result.next
    assert bad.source_cpp_sha256 is None


def test_continuous_clock_is_an_explicit_global_action_not_physical_fmax(cnn):
    state = initial_state(cnn, part=PART)
    result = transition(state, HardwareEdit("clock", "clock_ns", 13.125))
    assert result.status == "SOURCE_ACCEPTED"
    assert result.next.clock_ns == 13.125
    assert result.next.state_sha256 != state.state_sha256
    assert result.source_cpp_sha256 == validate_source(state)["cpp_sha256"]
    assert result.receipt()["hls_latency_ns"] is None
    assert result.receipt()["reward"] is None
    for invalid in (1.99, 32.001, 10.12345, float("nan"), True):
        rejected = transition(state, HardwareEdit("clock", "clock_ns", invalid))
        assert rejected.status == "REJECTED"
        assert rejected.next is state
    assert transition(state, HardwareEdit("clock", "clock_ns", 10.0)).status == "REJECTED"


def test_fail_closed_unimplemented_knobs_and_scalar_operator_mutation(cnn):
    state = initial_state(cnn, part=PART)
    scalar = next(i for i, op in enumerate(cnn.operators) if op.kind == "Relu")
    scenarios = (
        HardwareEdit("operator", "pes", 4, scalar),
        HardwareEdit("operator", "dma_width_bits", 64, 0),
        HardwareEdit("operator", "clock_domain", 2, 0),
        HardwareEdit("operator", "precision_bits", 8, 0),
        HardwareEdit("numeric", "precision_bits", 8, 0),
        HardwareEdit("clock", "clock_ns", 12.5, 0),
        HardwareEdit("operator", "pes", 4, len(state.requests)),
        HardwareEdit("operator", "pes", True, 0),
        HardwareEdit("operator", "tile", [1.0, 2.0], 0),
    )
    for action in scenarios:
        decision = transition(state, action)
        assert decision.status == "REJECTED", action
        assert decision.next is state
        assert decision.receipt()["source_cpp_sha256"] is None


def test_mask_uses_real_source_compiler_no_hls_and_no_fabricated_reward(cnn):
    state = initial_state(cnn, part=PART)
    idx = next(i for i, op in enumerate(cnn.operators) if op.kind == "Conv")
    actions = (
        HardwareEdit("operator", "pes", 2, idx),
        HardwareEdit("operator", "pes", 2, next(
            i for i, op in enumerate(cnn.operators) if op.kind == "Relu"
        )),
        HardwareEdit("numeric", "integer_bits", 0),
        HardwareEdit("clock", "clock_ns", 15.125),
    )
    mask = action_mask(state, actions)
    assert mask["source_compilable"] == [True, False, False, True]
    assert mask["native_vitis_calls"] == 0
    assert mask["actual_hls_resource_estimates"] is None
    assert mask["actual_hls_latency_ns"] is None
    assert mask["reward"] is None
    assert all(x is None for x, ok in zip(
        mask["source_cpp_sha256"], mask["source_compilable"]
    ) if not ok)
    assert state.requests[idx].pes == 1


def test_planning_session_is_not_a_fake_reward_gym(cnn):
    environment = HardwarePlanningEnvironment(cnn, part=PART)
    with pytest.raises(RuntimeError, match="reset"):
        environment.propose(HardwareEdit("clock", "clock_ns", 11.0))
    observation, info = environment.reset()
    assert info["native_vitis_calls"] == 0
    assert info["hardware_reward"] is None
    origin = environment.current.state_sha256
    observation, rejected = environment.propose(
        HardwareEdit("operator", "memory_banking", 2, 0)
    )
    assert rejected["status"] == "REJECTED"
    assert environment.current.state_sha256 == origin
    assert environment.source_steps == 0
    observation, accepted = environment.propose(
        HardwareEdit("clock", "clock_ns", 11.125)
    )
    assert accepted["status"] == "SOURCE_ACCEPTED"
    assert environment.source_steps == 1
    assert environment.current.clock_ns == 11.125
    with pytest.raises(RuntimeError, match="No fake reward"):
        environment.step(HardwareEdit("clock", "clock_ns", 12.0))
    assert environment.current.clock_ns == 11.125


def test_read_only_cli_inspects_actual_onnx_without_writing_case(
        tmp_path, capsys):
    pytest.importorskip("onnx")
    from examples.make_neural_network_onnx import save_model
    file = save_model("cnn_pool_dense", tmp_path / "model.onnx")
    original = file.read_bytes()
    assert main(["--onnx", str(file), "--part", "xc7z020clg400-1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source_receipt"]["source_compilable"] is True
    assert result["training_started"] is False
    assert result["native_experiments_started"] == 0
    assert result["knobs"]["global_continuous_clock"]["field"] == "clock_ns"
    assert file.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.onnx"]



@pytest.mark.parametrize("field,value", [
    ("parallelism", 2),
    ("pes", 2),
    ("simd", 2),
    ("serialization", 2),
    ("pipeline_ii", 2),
])
def test_each_existing_compute_hardware_knob_has_real_source_transition(
        cnn, field, value):
    state = initial_state(cnn, part=PART)
    idx = next(i for i, op in enumerate(cnn.operators) if op.kind == "MatMul")
    result = transition(state, HardwareEdit("operator", field, value, idx))
    assert result.status == "SOURCE_ACCEPTED", (field, result.reason)
    assert getattr(result.next.requests[idx], field) == value
    assert result.next.functional_sha256 == state.functional_sha256
    assert result.receipt()["reward"] is None


@pytest.mark.parametrize("field,value", [
    ("pes", 1024), ("serialization", 99999), ("pipeline_ii", 32),
    ("simd", -1),
])
def test_unlisted_extreme_values_are_not_rl_actions_even_if_source_can_parse(
        cnn, field, value):
    state = initial_state(cnn, part=PART)
    idx = next(i for i, op in enumerate(cnn.operators) if op.kind == "MatMul")
    result = transition(state, HardwareEdit("operator", field, value, idx))
    assert result.status == "REJECTED"
    assert result.next is state
    assert "bounded hardware knob menu" in result.reason



def test_deterministic_complete_action_menu_and_typed_policy_features(cnn):
    state = initial_state(cnn, part=PART)
    actions = candidate_actions(state, max_actions=512)
    assert actions == candidate_actions(state, max_actions=512)
    assert len(actions) == len(set(
        (a.scope, a.field, repr(a.value), a.operator_index) for a in actions
    ))
    assert {a.field for a in actions} >= set(ARCHITECTURE_FIELDS)
    assert all(a.field != "clock_ns" for a in actions)
    assert all(a.scope != "operator" or
               cnn.operators[a.operator_index].kind in ("MatMul", "Gemm", "Conv")
               for a in actions)
    features = action_features(state, actions)
    assert len(features) == len(actions)
    assert all(x["native_qor"] is None for x in features)
    assert all(type(x["field_index"]) is int for x in features)
    with pytest.raises(ValueError, match="refusing silent truncation"):
        candidate_actions(state, max_actions=1)


def test_environment_exposes_compiler_mask_without_native_evaluation(cnn):
    env = HardwarePlanningEnvironment(cnn, part=PART)
    env.reset()
    actions, bundle = env.offers(max_actions=512)
    assert len(actions) > 8
    assert bundle["reward"] is None
    assert bundle["native_vitis_calls"] == 0
    assert bundle["continuous_clock_domain"]["field"] == "clock_ns"
    mask = bundle["source_only_action_mask"]
    assert mask["action_count"] == len(actions)
    assert any(mask["source_compilable"])
    assert len(bundle["action_features"]) == len(actions)
    assert mask["reward"] is None



def test_budget_constraints_are_observed_not_fabricated(cnn):
    unspecified = initial_state(cnn, part=PART)
    obs = graph_observation(unspecified)
    assert obs["global"]["resource_caps"] is None
    assert obs["global"]["max_absolute_error"] is None
    assert obs["global"]["resource_caps_present"] is False
    assert obs["global"]["numerical_error_bound_present"] is False
    constrained = initial_state(
        cnn, part=PART, resource_caps=(20000, 25000, 100, 80, 0),
        max_absolute_error=0.125,
    )
    assert constrained.state_sha256 != unspecified.state_sha256
    data = graph_observation(constrained)["global"]
    assert data["resource_caps"] == {
        "lut": 20000, "ff": 25000, "dsp": 100,
        "bram_18k": 80, "uram": 0,
    }
    assert data["max_absolute_error"] == 0.125
    assert data["resource_caps_present"] is True
    assert data["numerical_error_bound_present"] is True
    assert validate_source(constrained)["source_compilable"] is True
    assert validate_source(constrained)["hardware_qor_known"] is False


@pytest.mark.parametrize("caps,error", [
    ((1, 2, 3, 4), None),
    ((1, 2, 3, -1, 5), None),
    ((1, 2, True, 4, 5), None),
    ((1, 2, 3, 4, 5), float("nan")),
    (None, 0),
])
def test_resource_or_numerical_task_constraints_fail_closed(cnn, caps, error):
    with pytest.raises(ValueError):
        initial_state(cnn, part=PART, resource_caps=caps, max_absolute_error=error)



def test_native_evaluation_handoff_is_a_plan_not_fabricated_feedback(cnn):
    unbound = initial_state(cnn, part=PART)
    with pytest.raises(ValueError, match="all five explicit resource caps"):
        native_evaluation_plan(unbound)
    exact = initial_state(
        cnn, part=PART, resource_caps=(20000, 25000, 100, 80, 0),
        max_absolute_error=0.1,
    )
    result = native_evaluation_plan(exact)
    assert result["status"] == "SOURCE_ONLY_NATIVE_EVALUATION_NOT_EXECUTED"
    assert result["state_sha256"] == exact.state_sha256
    assert result["functional_sha256"] == exact.functional_sha256
    assert result["cpp_sha256"] == validate_source(exact)["cpp_sha256"]
    assert result["requested_clock_ns"] == 10.0
    assert result["resource_caps"]["uram"] == 0
    assert result["requires_fixed_numerical_samples_and_testbench"] is True
    assert result["requires_effective_hls_timing_qualification"] is True
    assert result["reward"] is None
    assert result["native_vitis_calls"] == 0
    assert result["actual_native_hls_latency_ns"] is None
    assert result["physical_timing_achieved"] is False
    assert len(result["complete_architecture"]) == len(cnn.operators)



def test_explicit_fpga_part_is_mandatory_in_source_hardware_tasks(cnn):
    with pytest.raises(TypeError):
        initial_state(cnn)
    with pytest.raises(TypeError):
        HardwarePlanningEnvironment(cnn)
    with pytest.raises(ValueError, match="explicit hardware part"):
        initial_state(cnn, part="injection;bad")
