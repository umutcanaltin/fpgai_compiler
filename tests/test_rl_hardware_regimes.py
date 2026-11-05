"""Source-only six-way HW-RL regime and fixed-clock policy tests.

The ONNX generators and policy passes are CPU-only. No original Vitis HLS case
is created and no synthetic result can be labeled vendor evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from examples.make_neural_network_onnx import save_model
from fpgai_research.frontend.onnx import load_onnx
from rl_environment.agents.hardware_agent import HardwareGraphTrainer
from rl_environment.agents.hardware_algorithms import (
    AuditedExperience, RandomHardwarePolicy, create_hardware_method,
)
from rl_environment.environments.hardware_gym import GymHardwareNativeEnv
from rl_environment.experiments import hardware_campaign as campaign
from rl_environment.experiments.hardware_regimes import onnx_structure_sha256

PART = "xc7z020clg400-1"
CAPS = (53200, 106400, 220, 280, 0)


def _task(name, file_name, *, clock=14.0):
    return {
        "id": name, "split": "train", "onnx": file_name,
        "samples": "samples.npy", "part": PART,
        "resource_caps": list(CAPS), "max_absolute_error": 0.1,
        "initial_clock_ns": clock,
    }


def _manifest(tmp_path, *, general=False, task_fixed=False):
    a = save_model("cnn_pool_dense", tmp_path / "a.onnx")
    assert Path(a).is_file()
    if general:
        b = save_model("cnn_two_conv_softmax", tmp_path / "b.onnx")
        assert Path(b).is_file()
    np.save(tmp_path / "samples.npy", np.zeros((2, 1, 6, 6), np.float32))
    tasks = [_task("a", "a.onnx")]
    if general or task_fixed:
        tasks.append(_task("b", "b.onnx" if general else "a.onnx",
                           clock=15.0 if task_fixed else 14.0))
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({
        "schema": campaign.TASK_SCHEMA, "tasks": tasks,
    }))
    return manifest


def _plan(tasks, network, clock, *, episodes):
    return campaign.campaign_plan(
        tasks=tasks, split="train", method="ppo", episodes=episodes,
        calls_per_episode=2, total_authorized_calls=None, seed=101,
        evaluation=False, checkpoint=None, hidden=32,
        ppo_epochs=2, ppo_clip=.2,
        network_regime=network, clock_regime=clock,
    )


@pytest.mark.parametrize("network", ["same", "general"])
@pytest.mark.parametrize("clock", ["fixed", "task_fixed", "search"])
def test_six_explicit_network_and_clock_regimes_are_source_only(tmp_path, network, clock):
    path = _manifest(tmp_path, general=network == "general",
                     task_fixed=clock == "task_fixed")
    tasks = campaign.read_tasks(path)
    plan = _plan(tasks, network, clock,
                 episodes=2 if (network == "general" or clock == "task_fixed") else 1)
    assert plan["network_regime"] == network
    assert plan["clock_regime"] == clock
    assert plan["native_execution_authorized"] is False
    assert plan["source_only_plan"] is True
    assert plan["planned_max_original_vitis_calls"] == plan["episodes"] * 2
    if clock == "fixed":
        assert plan["fixed_clock_ns"] == 14.0
        assert plan["clock_action_bounds_ns"] == [14.0, 14.0]
    elif clock == "task_fixed":
        assert plan["fixed_clock_ns"] is None
        assert set(plan["fixed_clock_ns_by_task"].values()) == {14.0, 15.0}
    else:
        assert plan["clock_action_bounds_ns"] == [2.0, 32.0]
    if network == "general":
        assert plan["distinct_onnx_graph_count"] == 2
    assert not any(tmp_path.glob("episode_*"))


def test_general_network_disallows_one_model_and_weight_only_changes(tmp_path):
    single = campaign.read_tasks(_manifest(tmp_path))
    with pytest.raises(ValueError, match="at least two distinct ONNX models"):
        _plan(single, "general", "search", episodes=1)

    p = tmp_path / "a.onnx"
    modified = onnx.load(str(p))
    modified.producer_name = "metadata-change-only"
    onnx.save(modified, tmp_path / "b.onnx")
    assert p.read_bytes() != (tmp_path / "b.onnx").read_bytes()
    assert onnx_structure_sha256(p) == onnx_structure_sha256(tmp_path / "b.onnx")
    data = json.loads((tmp_path / "tasks.json").read_text())
    data["tasks"].append(_task("b", "b.onnx"))
    (tmp_path / "tasks.json").write_text(json.dumps(data))
    tasks = campaign.read_tasks(tmp_path / "tasks.json")
    with pytest.raises(ValueError, match="distinct ONNX graph architectures"):
        _plan(tasks, "general", "search", episodes=2)


def test_same_network_refuses_multiple_structures_and_fixed_refuses_mixed_clocks(tmp_path):
    tasks = campaign.read_tasks(_manifest(tmp_path, general=True, task_fixed=True))
    with pytest.raises(ValueError, match="exactly one immutable ONNX"):
        _plan(tasks, "same", "search", episodes=2)
    with pytest.raises(ValueError, match="one identical initial_clock"):
        _plan(tasks, "general", "fixed", episodes=2)
    equal_dir = tmp_path / "equal_clock_cohort"
    equal_dir.mkdir()
    equal = campaign.read_tasks(_manifest(equal_dir, general=True))
    with pytest.raises(ValueError, match="at least two different initial clock"):
        _plan(equal, "general", "task_fixed", episodes=2)


def _environment(tmp_path, *, fixed):
    path = save_model("cnn_pool_dense", tmp_path / "env_model.onnx")
    model = load_onnx(str(path))
    shape = model.tensor(model.inputs[0]).shape
    samples = np.zeros((2, *shape), np.float32)
    return GymHardwareNativeEnv(
        model, samples, tmp_path / "never_allocate_native",
        part=PART, resource_caps=CAPS, max_absolute_error=.1,
        max_vitis_calls=2, initial_clock_ns=14.,
        fixed_clock_ns=14. if fixed else None,
        execute_native=False,
    )


def test_fixed_clock_env_masks_clock_only_and_rejects_any_clock_change(tmp_path):
    env = _environment(tmp_path, fixed=True)
    obs, _ = env.reset(seed=101)
    assert list(obs["clock_bounds_ns"]) == [14., 14.]
    assert obs["action_mask"][0] == 0
    legal = np.flatnonzero(obs["action_mask"])
    assert len(legal)
    with pytest.raises(ValueError, match="Invalid/masked"):
        env.preview_action((0, 14.))
    with pytest.raises(ValueError, match="Fixed-clock"):
        env.preview_action((int(legal[0]), 15.))
    assert env.preview_action((int(legal[0]), 14.)).clock_ns == 14.
    assert env.calls == 0
    assert not env.root.exists()


def test_search_clock_keeps_clock_only_action(tmp_path):
    env = _environment(tmp_path, fixed=False)
    obs, _ = env.reset(seed=101)
    assert list(obs["clock_bounds_ns"]) == [2., 32.]
    assert obs["action_mask"][0] == 1
    assert env.preview_action((0, 15.)).clock_ns == 15.
    assert env.calls == 0


def test_fixed_clock_ppo_and_actor_critic_have_no_beta_sample_or_clock_gradients(tmp_path):
    env = _environment(tmp_path, fixed=True)
    obs, _ = env.reset(seed=101)
    agent = HardwareGraphTrainer(seed=101, hidden=16)
    action, tape = agent.choose(obs, training=True)
    assert action[1] == 14.
    assert tape.sampled_clock_fraction is None
    assert float(tape.requested_clock_fraction[0]) == .5
    logits, _ = agent.policy(obs)
    expected = torch.distributions.Categorical(logits=logits).log_prob(
        torch.tensor(action[0])
    )
    assert torch.allclose(tape.log_probability, expected)
    agent.optimizer.zero_grad(set_to_none=True)
    (-tape.log_probability).backward()
    assert all(p.grad is None for p in agent.policy.clock_head.parameters())
    ppo = create_hardware_method("ppo", seed=101, hidden=16)
    _, sample = ppo.choose(obs, training=True)
    trace = [AuditedExperience(sample, -.05, False, "CPU_ONLY_NOT_HLS")]
    pieces = ppo._ppo_objective(trace, torch.tensor([-.05]))
    assert all(bool(torch.isfinite(p)) for p in pieces[:4])
    ppo.optimizer.zero_grad(set_to_none=True)
    sum(pieces[:4]).backward()
    assert all(p.grad is None for p in ppo.policy.clock_head.parameters())
    index, clock = RandomHardwarePolicy(seed=101).choose(obs)
    assert clock == 14.
    assert index != 0


def test_fixed_clock_must_match_source_initial_state(tmp_path):
    p = save_model("cnn_pool_dense", tmp_path / "mismatch.onnx")
    model = load_onnx(str(p))
    x = np.zeros((2, *model.tensor(model.inputs[0]).shape), np.float32)
    with pytest.raises(ValueError, match="Fixed clock must equal"):
        GymHardwareNativeEnv(
            model, x, tmp_path / "no_vendor", part=PART,
            resource_caps=CAPS, max_absolute_error=.1,
            initial_clock_ns=14., fixed_clock_ns=15.,
            execute_native=False,
        )
    assert not (tmp_path / "no_vendor").exists()


def test_regime_cli_flags_plan_source_only_without_vendor(tmp_path, capsys):
    path = _manifest(tmp_path)
    root = tmp_path / "must_never_be_created_by_plan"
    assert campaign.main([
        "--tasks", str(path), "--network-regime", "same",
        "--clock-regime", "fixed", "--method", "actor_critic",
        "--episodes", "1", "--calls-per-episode", "2",
        "--output-root", str(root),
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["source_only_plan"] is True
    assert plan["native_execution_authorized"] is False
    assert plan["network_regime"] == "same"
    assert plan["clock_regime"] == "fixed"
    assert plan["clock_action_bounds_ns"] == [14.0, 14.0]
    assert not root.exists()


def test_explicit_regimes_require_identical_hardware_constraints(tmp_path):
    path = _manifest(tmp_path, general=True)
    source = json.loads(path.read_text())
    source["tasks"][1]["resource_caps"][0] -= 1
    path.write_text(json.dumps(source))
    tasks = campaign.read_tasks(path)
    with pytest.raises(ValueError, match="same FPGA part"):
        _plan(tasks, "general", "search", episodes=2)


def test_hw08_family_and_cross_family_questions_are_source_only(tmp_path):
    """Two CNNs are within-family, never evidence of cross-family transfer."""
    tasks = campaign.read_tasks(_manifest(tmp_path, general=True))
    within = _plan(tasks, "family", "fixed", episodes=2)
    assert within["distinct_onnx_graph_count"] == 2
    assert within["distinct_onnx_operator_family_count"] == 1
    assert set(within["onnx_operator_family_by_task"].values()) == {
        "convolutional"
    }
    with pytest.raises(ValueError, match="two distinct ONNX operator families"):
        _plan(tasks, "cross_family", "fixed", episodes=2)


def test_hw08_cross_family_rejects_same_family_even_if_graph_is_different(tmp_path):
    tasks = campaign.read_tasks(_manifest(tmp_path, general=True))
    # General mixes graph structures; cross-family additionally requires
    # different operator families. No HLS, Gym or synthetic vendor calls.
    assert _plan(tasks, "general", "search", episodes=2)[
        "distinct_onnx_operator_family_count"
    ] == 1
    with pytest.raises(ValueError, match="two distinct ONNX operator families"):
        _plan(tasks, "cross_family", "search", episodes=2)


def test_hw08_fixed_family_refuses_mixed_conv_and_dense_graphs(tmp_path):
    from onnx import TensorProto, helper, numpy_helper
    from rl_environment.experiments.hardware_regimes import onnx_operator_family

    manifest = _manifest(tmp_path)
    data = json.loads(manifest.read_text())
    dense = tmp_path / "dense.onnx"
    weight = numpy_helper.from_array(
        np.ones((4, 2), dtype=np.float32), name="w"
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["out"])],
        "dense_network",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    dense.write_bytes(model.SerializeToString())
    assert onnx_operator_family(dense) == "dense_matmul"
    data["tasks"].append(_task("dense", "dense.onnx"))
    manifest.write_text(json.dumps(data))
    tasks = campaign.read_tasks(manifest)
    with pytest.raises(ValueError, match="one ONNX operator family"):
        _plan(tasks, "family", "fixed", episodes=2)
    different = _plan(tasks, "cross_family", "fixed", episodes=2)
    assert different["distinct_onnx_operator_family_count"] == 2
    assert set(different["onnx_operator_family_by_task"].values()) == {
        "convolutional", "dense_matmul"
    }
    assert different["native_execution_authorized"] is False


def test_hw08_hardware_infeasibility_is_real_negative_rl_feedback():
    from rl_environment.reward import latency_only_reward
    assert latency_only_reward(None, None, qualified=False) == -.05
    assert latency_only_reward(9072., 9072., qualified=False) == -.05
    assert latency_only_reward(None, 9072., qualified=True) == -.005
