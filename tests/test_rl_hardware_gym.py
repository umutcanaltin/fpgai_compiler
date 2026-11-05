"""HW-RL-03 CPU-only Gym lifecycle tests; no licensed vendor is invoked.

Mocked receipts in these tests are *not* native qualification evidence.
Real reward requires unmocked execute_native_case + independent audit.
"""
from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("onnx")
from examples.make_neural_network_onnx import save_model
from fpgai_research.frontend.onnx import load_onnx
from rl_environment.hardware_space import HardwareEdit, transition
from rl_environment.hardware_gym import GymHardwareNativeEnv
from rl_environment import hardware_native


PART = "xc7z020clg400-1"
CAPS = (53200, 106400, 220, 280, 0)


@pytest.fixture
def env_task(tmp_path):
    file = save_model("cnn_pool_dense", tmp_path / "cnn.onnx")
    model = load_onnx(str(file))
    shape = model.tensor(model.inputs[0]).shape
    samples = np.zeros((2, *shape), np.float32)
    samples[1].fill(.125)
    return model, samples, tmp_path / "gym_hw03"


def _new_env(env_task, *, execute_native=False, max_vitis_calls=3):
    model, samples, root = env_task
    return GymHardwareNativeEnv(
        model, samples, root, part=PART, resource_caps=CAPS,
        max_absolute_error=.1, initial_clock_ns=14.,
        max_vitis_calls=max_vitis_calls,
        execute_native=execute_native,
        vitis_executable="/unused/mock-vitis_hls",
    )


def _fake_hls_result(case: Path, state, samples, *, qualified, cycles=646):
    plan = hardware_native.planned_case(state, samples)
    if qualified:
        failures = []
        latency = cycles * state.clock_ns
        timing = {"hls_timing_feasible": True, "hls_estimated_clock_ns": 10.213}
    else:
        failures = ["ORIGINAL_HLS_SCHEDULING_NOT_FEASIBLE"]
        latency = None
        timing = {"hls_timing_feasible": False, "hls_estimated_clock_ns": 8.462}
    return {
        "schema": hardware_native.AUDIT_SCHEMA,
        "state_sha256": state.state_sha256,
        "functional_sha256": state.functional_sha256,
        "source_cpp_sha256": plan["cpp_sha256"],
        "sample_sha256": plan["sample_sha256"],
        "case": str(case.absolute()),
        "requested_clock_ns": state.clock_ns,
        "native_calls_by_audit": 0,
        "native_invocation_present": True,
        "reward": None,
        "qualified": qualified,
        "gate_failures": failures,
        "board_qualified": False,
        "original_vitis_completed_successfully": True,
        "csim_passed": True,
        "csim_max_absolute_error": .00039,
        "hls_worst_case_cycles": cycles if qualified else None,
        "raw_hls_worst_case_cycles": cycles,
        "hls_estimated_latency_ns": latency,
        "hls_timing": timing,
        "evidence": "E3-HLS-synthesis-report",
        "original_hls_resource_consumed": {
            "lut": 10238, "ff": 3734, "dsp": 45, "bram_18k": 1, "uram": 0,
        },
    }


def test_hw03_reset_is_source_only_gym_observation_with_variable_graph(env_task):
    env = _new_env(env_task)
    model, _, root = env_task
    observation, info = env.reset(seed=101)
    assert env.observation_space.contains(observation)
    assert info["native_vitis_calls"] == 0
    assert info["reward"] is None
    assert not root.exists()
    assert observation["node_mask"].sum() == len(model.operators) + len(model.tensors)
    assert observation["edge_mask"].sum() > 0
    assert observation["action_mask"][0] == 1
    assert observation["action_mask"].sum() > 8
    assert observation["action_target_node"][0] == -1
    assert observation["history_mask"].sum() == 0
    assert observation["history"].shape == (30, 18)
    assert observation["clock_bounds_ns"].tolist() == [2., 32.]
    assert np.isfinite(observation["nodes"]).all()
    assert np.isfinite(observation["global"]).all()
    assert np.isfinite(observation["action_features"]).all()
    assert not any("weights" in str(k) for k in observation.keys())
    with pytest.raises(RuntimeError, match="execute_native"):
        env.step((0, 15.0))
    assert not root.exists()
    assert env.calls == 0


@pytest.mark.parametrize("action,match", [
    ((0, 14.0), "No-op"),
    ((999, 14.0), "Invalid/masked"),
    ((0, 100.0), "outside"),
    ((0, float("nan")), "finite"),
    ((0, -5.0), "outside"),
    (("0", 14.0), "Expected"),
])
def test_hw03_invalid_actions_spend_zero_vendor_calls(env_task, action, match):
    env = _new_env(env_task, execute_native=True)
    env.reset()
    with pytest.raises(ValueError, match=match):
        env.step(action)
    assert env.calls == 0
    assert not env_task[2].exists()


def test_hw03_qualified_infeasible_and_improvement_rewards_are_original_latency_only(
        env_task, monkeypatch):
    env = _new_env(env_task, execute_native=True, max_vitis_calls=3)
    calls = []

    def fake_execute(case, state, samples, **kwargs):
        calls.append(("single-original-vitis-call", str(case), state.state_sha256))
        assert not case.exists()
        case.mkdir()  # CPU fixture only; never creates vendor artifacts.
        (case / "UNIT_TEST_NOT_VENDOR").write_text("not real evidence")
        qualified = state.clock_ns != 13.0
        report = _fake_hls_result(case, state, samples, qualified=qualified)
        return report

    def fake_independent_audit(case, state, samples):
        calls.append(("read-only-independent-replay", str(case), state.state_sha256))
        assert (case / "UNIT_TEST_NOT_VENDOR").is_file()
        return _fake_hls_result(
            case, state, samples, qualified=state.clock_ns != 13.0,
        )

    monkeypatch.setattr(hardware_native, "execute_native_case", fake_execute)
    monkeypatch.setattr(hardware_native, "audit_native_case", fake_independent_audit)
    obs, info = env.reset(seed=111)
    assert info["native_vitis_calls"] == 0
    obs, reward, done, truncated, info = env.step((0, 15.0))
    assert reward == pytest.approx(-.005)
    assert info["qualified"] is True
    assert info["hls_estimated_latency_ns"] == 646 * 15.
    assert env.best_latency_ns == 9690.
    assert done is False and truncated is False
    assert obs["history_mask"].sum() == 1
    assert obs["history"][0, 2] == 1.
    assert obs["history"][0, 3] == pytest.approx(-.005)
    first_case = Path(info["native_case"])
    assert first_case.name == "case_0000"
    assert env.observation_space.contains(obs)
    obs, reward, done, truncated, info = env.step((0, 13.0))
    assert reward == -.05
    assert info["qualified"] is False
    assert info["hls_estimated_latency_ns"] is None
    assert env.best_latency_ns == 9690.
    assert done is False and truncated is False
    assert obs["history_mask"].sum() == 2
    assert obs["history"][1, 2] == 0.
    assert obs["history"][1, 3] == pytest.approx(-.05)
    obs, reward, done, truncated, info = env.step((0, 14.5))
    assert reward == pytest.approx(log(9690 / (646 * 14.5)) - .005)
    assert info["qualified"] is True
    assert info["best_hls_estimated_latency_ns"] == 646 * 14.5
    assert obs["history_mask"].sum() == 3
    assert obs["history"][2, 2] == 1.
    assert truncated is True and done is False
    assert len(calls) == 6
    assert env.calls == 3
    assert first_case.is_dir()
    assert len(list(first_case.parent.glob("case_*"))) == 3
    assert len(list(first_case.parent.glob("step_*.json"))) == 3
    with pytest.raises(RuntimeError, match="completed"):
        env.step((0, 16.0))


def test_hw03_source_knob_and_clock_can_change_together(env_task, monkeypatch):
    env = _new_env(env_task, execute_native=True, max_vitis_calls=1)
    obs, _ = env.reset()
    idx = next(i for i, a in enumerate(env.offers) if a is not None
               and a.scope == "operator" and a.field == "pes"
               and env.current.model.operators[a.operator_index].kind == "Conv"
               and a.value == 2)
    assert obs["action_mask"][idx] == 1
    seen = []

    def fake_execute(case, state, samples, **kwargs):
        seen.append((case, state))
        case.mkdir()
        return _fake_hls_result(case, state, samples, qualified=True)

    monkeypatch.setattr(hardware_native, "execute_native_case", fake_execute)
    monkeypatch.setattr(hardware_native, "audit_native_case", lambda case, state, samples:
                        _fake_hls_result(case, state, samples, qualified=True))
    obs, reward, terminated, truncated, info = env.step((idx, 15.0))
    assert len(seen) == 1
    assert seen[0][1].clock_ns == 15.
    assert any(req.pes == 2 for req in seen[0][1].requests)
    assert info["qualified"] is True
    assert reward == -.005
    assert truncated is True


def test_hw03_contradictory_replay_aborts_reward_and_preserves_case(
        env_task, monkeypatch):
    env = _new_env(env_task, execute_native=True)
    env.reset()
    cases = []

    def fake_execute(case, state, samples, **kwargs):
        cases.append(case)
        case.mkdir()
        return _fake_hls_result(case, state, samples, qualified=True)

    def corrupt_replay(case, state, samples):
        report = _fake_hls_result(case, state, samples, qualified=True)
        report["source_cpp_sha256"] = "0" * 64
        return report

    monkeypatch.setattr(hardware_native, "execute_native_case", fake_execute)
    monkeypatch.setattr(hardware_native, "audit_native_case", corrupt_replay)
    with pytest.raises(RuntimeError, match="independent replay differ"):
        env.step((0, 15.0))
    assert env.done is True
    assert env.calls == 1
    assert env.best_latency_ns is None
    assert cases[0].is_dir()
    assert not list(cases[0].parent.glob("step_*.json"))
    with pytest.raises(RuntimeError, match="completed"):
        env.step((0, 17.0))


def test_hw03_does_not_silently_use_existing_episode_dir(env_task):
    env = _new_env(env_task, execute_native=True)
    root = env_task[2]
    (root / "episode_000000").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Existing episode"):
        env.reset()


def test_hw03_requires_real_task_constraints_and_numerical_cohort(env_task):
    model, samples, root = env_task
    with pytest.raises(ValueError, match="five"):
        GymHardwareNativeEnv(
            model, samples, root, part=PART,
            resource_caps=(1, 2, 3, 4), max_absolute_error=.1)
    with pytest.raises(ValueError, match="samples"):
        GymHardwareNativeEnv(
            model, samples.astype(np.float64), root, part=PART,
            resource_caps=CAPS, max_absolute_error=.1)



def test_hw03_cli_preview_is_read_only_and_does_not_start_training(
        env_task, tmp_path, capsys):
    from rl_environment.hardware_gym import main
    model, samples, root = env_task
    original = tmp_path / "model.onnx"
    save_model("cnn_pool_dense", original)
    sample_file = tmp_path / "cohort.npy"
    np.save(sample_file, samples)
    argv = [
        "--onnx", str(original), "--samples", str(sample_file),
        "--part", PART,
        "--resource-caps", *(str(v) for v in CAPS),
        "--max-absolute-error", ".1",
        "--initial-clock-ns", "14.0",
        "--output-root", str(root),
    ]
    assert main(argv) == 0
    import json
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "SOURCE_ACTION_PREVIEW_ONLY"
    assert report["native_vitis_calls"] == 0
    assert report["training_started"] is False
    assert report["source_accepted_mask_count"] > 8
    assert report["first_action_choices"][0]["edit"] is None
    assert report["graph_nodes"] == len(model.tensors) + len(model.operators)
    assert not root.exists()
    assert sample_file.is_file()


def test_hw03_cli_requires_explicit_clock_for_native_opt_in(
        env_task, tmp_path, capsys):
    from rl_environment.hardware_gym import main
    model, samples, root = env_task
    original = tmp_path / "model.onnx"
    save_model("cnn_pool_dense", original)
    sample_file = tmp_path / "cohort.npy"
    np.save(sample_file, samples)
    argv = [
        "--onnx", str(original), "--samples", str(sample_file),
        "--part", PART,
        "--resource-caps", *(str(v) for v in CAPS),
        "--max-absolute-error", ".1", "--initial-clock-ns", "14.0",
        "--output-root", str(root), "--execute-native",
    ]
    assert main(argv) == 2
    assert "--action-clock-ns required" in capsys.readouterr().err
    assert not root.exists()



@pytest.mark.parametrize("missing", [
    "original_vitis_completed_successfully",
    "evidence", "csim_passed",
    "raw_hls_worst_case_cycles", "hls_timing",
    "original_hls_resource_consumed",
])
def test_hw03_unattributed_infrastructure_failures_never_become_negative_rewards(
        env_task, monkeypatch, missing):
    env = _new_env(env_task, execute_native=True, max_vitis_calls=1)
    env.reset()
    cases = []

    def bad_result(case, state, samples):
        result = _fake_hls_result(case, state, samples, qualified=False)
        if missing == "hls_timing":
            result[missing] = None
        elif missing == "original_hls_resource_consumed":
            result[missing] = None
        elif missing == "evidence":
            result[missing] = "E1-source-ready-for-vitis"
        elif missing in ("original_vitis_completed_successfully", "csim_passed"):
            result[missing] = False
        else:
            result[missing] = None
        return result

    def fake_execute(case, state, samples, **kwargs):
        cases.append(case)
        case.mkdir()
        return bad_result(case, state, samples)

    monkeypatch.setattr(hardware_native, "execute_native_case", fake_execute)
    monkeypatch.setattr(hardware_native, "audit_native_case", bad_result)
    with pytest.raises(RuntimeError, match="Incomplete original E3/CSim"):
        env.step((0, 15.0))
    assert env.done is True
    assert env.calls == 1
    assert env.best_latency_ns is None
    assert cases[0].is_dir()
    assert not list(cases[0].parent.glob("step_*.json"))


def test_hw03_source_preview_is_pure_and_rejects_visited_candidates(env_task):
    """Shared campaign/Gym preflight never consumes a native call."""
    env = _new_env(env_task, execute_native=True)
    observation, _ = env.reset(seed=101)
    first = env.preview_action((0, 15.0))
    assert first.clock_ns == 15.0
    assert first.state_sha256 != env.current.state_sha256
    assert env.preview_action((0, 15.0)).state_sha256 == first.state_sha256
    assert env.calls == 0
    assert env.current.clock_ns == 14.0
    assert not env_task[2].exists()
    assert observation["action_mask"][0] == 1
    with pytest.raises(ValueError, match="No-op"):
        env.preview_action((0, 14.0))
    env.visited.add(first.state_sha256)
    with pytest.raises(ValueError, match="Repeated complete"):
        env.preview_action((0, 15.0))
    assert env.calls == 0
    assert not env_task[2].exists()


@pytest.mark.parametrize("reward_mode", ["best_improvement", "dense_latency"])
def test_hw09_reward_profiles_use_each_original_audited_step_not_a_surrogate(
        env_task, monkeypatch, reward_mode):
    """Synthetic CPU fixture checks reward plumbing; this is NOT Vitis evidence."""
    model, samples, root = env_task
    env = GymHardwareNativeEnv(
        model, samples, root, part=PART, resource_caps=CAPS,
        max_absolute_error=.1, initial_clock_ns=14.,
        max_vitis_calls=2, execute_native=True,
        reward_mode=reward_mode, reward_reference_ns=10000.,
    )
    audit_calls = []

    def mock_execute(case, state, values, **kwargs):
        audit_calls.append(("fake_original", state.clock_ns))
        assert not case.exists()
        case.mkdir(parents=True)
        (case / "CPU_MOCK_NOT_E3_HLS").write_text("not vendor evidence")
        return _fake_hls_result(
            case, state, values, qualified=state.clock_ns != 13.,
        )

    def mock_audit(case, state, values):
        audit_calls.append(("fake_readonly_replay", state.clock_ns))
        assert (case / "CPU_MOCK_NOT_E3_HLS").exists()
        return _fake_hls_result(
            case, state, values, qualified=state.clock_ns != 13.,
        )

    monkeypatch.setattr(hardware_native, "execute_native_case", mock_execute)
    monkeypatch.setattr(hardware_native, "audit_native_case", mock_audit)
    obs, info = env.reset(seed=101)
    assert info["native_vitis_calls"] == 0
    assert not root.exists()

    _, failed_reward, _, _, failed_info = env.step((0, 13.))
    assert failed_reward == -.05
    assert failed_info["qualified"] is False
    assert failed_info["reward_mode"] == reward_mode
    _, feasible_reward, _, truncated, good_info = env.step((0, 15.))
    expected_ns = 646 * 15.
    assert good_info["hls_estimated_latency_ns"] == expected_ns
    assert good_info["qualified"] is True
    assert truncated is True
    if reward_mode == "best_improvement":
        assert feasible_reward == -.005
    else:
        assert feasible_reward == 10000. / (10000. + expected_ns) - .005
    assert [kind for kind, _ in audit_calls] == [
        "fake_original", "fake_readonly_replay",
        "fake_original", "fake_readonly_replay",
    ]
    first = root / "episode_000000" / "step_0000.json"
    second = root / "episode_000000" / "step_0001.json"
    assert first.is_file() and second.is_file()
    assert json.loads(first.read_text())["reward"] == -.05
    assert json.loads(second.read_text())["reward"] == feasible_reward
