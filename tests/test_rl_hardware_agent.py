"""HW-RL-04 shared hardware-GNN policy CPU tests; zero native Vitis calls.

Synthetic audited-looking dictionaries below are test fixtures ONLY:
they must never be published as evidence of licensed vendor synthesis.
"""
from __future__ import annotations

import copy
import json
from math import isfinite

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
pytest.importorskip("onnx")

from examples.make_neural_network_onnx import save_model
from fpgai_research.frontend.onnx import load_onnx
from rl_environment.hardware_agent import (
    HardwareGraphTrainer, SharedHardwareGraphPolicy, HardwareActionTape,
)
from rl_environment.hardware_gym import (
    GymHardwareNativeEnv, MAX_ACTIONS, MAX_HISTORY, MAX_NODES,
    HISTORY_FEATURES,
)
from rl_environment import hardware_native


PART = "xc7z020clg400-1"
CAPS = (53200, 106400, 220, 280, 0)


def _env(tmp_path, variant):
    model_file = save_model(variant, tmp_path / (variant + ".onnx"))
    model = load_onnx(str(model_file))
    shape = model.tensor(model.inputs[0]).shape
    samples = np.zeros((2, *shape), np.float32)
    env = GymHardwareNativeEnv(
        model, samples, tmp_path / ("run_" + variant),
        part=PART, resource_caps=CAPS, max_absolute_error=.1,
        max_vitis_calls=2, execute_native=False,
        initial_clock_ns=14.,
    )
    return env


def _fake_transition_info(env, *, qualified=True):
    # This is a deliberately fake CPU-only fixture for testing gradients.
    # It can pass the trainer's shape/metadata gate but does NOT represent
    # an original Vitis invocation or independently verified E3 record.
    latency = 646 * 15. if qualified else None
    return {
        "schema": "fpgai-research/rl-hardware-gym-native/v1",
        "state_sha256": env.current.state_sha256,
        "native_vitis_calls": 1,
        "native_case": str(env.root / "CPU_TEST_NOT_NATIVE_CASE"),
        "qualified": qualified,
        "original_vitis_audit": {
            "schema": "fpgai-research/rl-hardware-native-audit/v1",
            "state_sha256": env.current.state_sha256,
            "qualified": qualified,
            "native_invocation_present": True,
            "native_calls_by_audit": 0,
            "original_vitis_completed_successfully": True,
            "csim_passed": True,
            "evidence": "E3-HLS-synthesis-report",
            "hls_estimated_latency_ns": latency,
            "hls_worst_case_cycles": 646 if qualified else None,
            "gate_failures": [] if qualified else ["ORIGINAL_HLS_SCHEDULING_NOT_FEASIBLE"],
        },
    }


def test_hw04_shared_policy_handles_actual_8_operator_cnn_and_source_masks(tmp_path):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, info = env.reset(seed=123)
    policy = SharedHardwareGraphPolicy(hidden=32)
    logits, value, contexts = policy(obs, with_context=True)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
    assert logits.shape == (MAX_ACTIONS,)
    assert contexts.shape == (MAX_ACTIONS, 32 * 5)
    assert value.ndim == 0
    assert torch.isfinite(logits[mask]).all()
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(value)
    assert obs["action_target_node"][0] == -1
    assert obs["action_target_node"][1] == -1
    conv_slots = [
        i for i, action in enumerate(env.offers)
        if action is not None and action.scope == "operator"
        and env.current.model.operators[action.operator_index].kind == "Conv"
        and obs["action_mask"][i]
    ]
    assert conv_slots
    for idx in conv_slots:
        action = env.offers[idx]
        target = int(obs["action_target_node"][idx])
        assert target == len(env.current.model.tensors) + action.operator_index
        assert obs["nodes"][target, 1] == 1.
    assert obs["history"].shape == (MAX_HISTORY, HISTORY_FEATURES)
    assert int(obs["history_mask"].sum()) == 0
    assert torch.isfinite(logits[mask]).all()
    assert not (tmp_path / "run_cnn_pool_dense").exists()


def test_hw04_one_policy_handles_multiple_network_graph_sizes(tmp_path):
    a = _env(tmp_path, "cnn_pool_dense")
    b = _env(tmp_path, "cnn_two_conv_softmax")
    first, _ = a.reset(seed=11)
    second, _ = b.reset(seed=12)
    assert first["node_mask"].sum() != second["node_mask"].sum()
    shared = SharedHardwareGraphPolicy(hidden=24)
    la, va = shared(first)
    lb, vb = shared(second)
    assert la.shape == lb.shape == (MAX_ACTIONS,)
    assert torch.isfinite(va) and torch.isfinite(vb)
    assert torch.isfinite(la[torch.as_tensor(first["action_mask"], dtype=torch.bool)]).all()
    assert torch.isfinite(lb[torch.as_tensor(second["action_mask"], dtype=torch.bool)]).all()


def test_hw04_actor_samples_masked_categorical_plus_continuous_clock(tmp_path):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset(seed=101)
    trainer = HardwareGraphTrainer(seed=101, hidden=24)
    before = trainer.updates
    for i in range(8):
        action, tape = trainer.choose(obs, training=True)
        index, period = action
        assert obs["action_mask"][index] == 1
        assert 2. <= period <= 32.
        assert period == round(period, 2)
        assert isinstance(tape, HardwareActionTape)
        assert tape.source_action_index == index
        assert torch.isfinite(tape.log_probability)
        assert torch.isfinite(tape.entropy)
        assert torch.isfinite(tape.value)
        assert 0. <= tape.predicted_feasibility <= 1.
    frozen, frozen_tape = trainer.choose(obs, training=False, sample=False)
    assert frozen_tape is None
    assert obs["action_mask"][frozen[0]] == 1
    assert trainer.updates == before
    assert not (tmp_path / "run_cnn_pool_dense").exists()


def test_hw04_no_training_from_preview_or_missing_original_audit(tmp_path):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset()
    trainer = HardwareGraphTrainer(seed=201, hidden=16)
    _, tape = trainer.choose(obs, training=True)
    weights = [x.detach().clone() for x in trainer.policy.parameters()]
    with pytest.raises(ValueError, match="Missing original-native"):
        trainer.update(tape, -.005, info={}, env=env, next_obs=obs, terminated=True)
    assert trainer.updates == 0
    for previous, now in zip(weights, trainer.policy.parameters()):
        assert torch.equal(previous, now)
    info = _fake_transition_info(env, qualified=True)
    info["original_vitis_audit"]["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Unattributed"):
        trainer.update(tape, -.005, info=info, env=env, terminated=True)
    assert trainer.updates == 0


def test_hw04_cpu_gradient_update_requires_audited_shape_only_fixture(
        tmp_path, monkeypatch):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset(seed=101)
    trainer = HardwareGraphTrainer(seed=101, hidden=24)
    _, tape = trainer.choose(obs, training=True)
    before = [v.detach().clone() for v in trainer.policy.parameters()]
    # Synthetic metadata solely exercises backprop; no Vitis is invoked.
    info = _fake_transition_info(env, qualified=True)
    # CPU test only: stub an environment state and read-only audit to test
    # gradients. This does NOT prove a real native vendor execution.
    env.calls = 1
    env.last_audit = info["original_vitis_audit"]
    env.history = [{
        "audit": env.last_audit, "action_index": tape.source_action_index,
        "clock_ns": 15., "reward": -.005,
        "best_ns": info["original_vitis_audit"]["hls_estimated_latency_ns"],
    }]
    monkeypatch.setattr(hardware_native, "audit_native_case",
                        lambda case, state, samples: info["original_vitis_audit"])
    metrics = trainer.update(
        tape, -.005, info=info, env=env, next_obs=None,
        terminated=True,
    )
    assert trainer.updates == 1
    assert metrics["qualified"] is True
    assert metrics["reward"] == -.005
    assert metrics["hls_estimated_latency_ns"] == 9690.
    assert metrics["no_new_vendor_calls_by_trainer"] is True
    assert metrics["independent_read_only_vendor_case_replay"] is True
    assert all(isfinite(float(metrics[k])) for k in (
        "loss", "actor_loss", "critic_loss", "supervised_feasibility_loss",
        "gradient_norm_before_clip"
    ))
    assert any(not torch.equal(a, b) for a, b in zip(
        before, trainer.policy.parameters()
    ))
    assert not (tmp_path / "run_cnn_pool_dense").exists()


def test_hw04_cpu_negative_label_and_shared_td_bootstrap(tmp_path, monkeypatch):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset()
    trainer = HardwareGraphTrainer(seed=103, hidden=16)
    _, tape = trainer.choose(obs, training=True)
    info = _fake_transition_info(env, qualified=False)
    env.calls = 1
    env.last_audit = info["original_vitis_audit"]
    env.history = [{
        "audit": env.last_audit, "action_index": tape.source_action_index,
        "clock_ns": 15., "reward": -.05, "best_ns": None,
    }]
    monkeypatch.setattr(hardware_native, "audit_native_case",
                        lambda case, state, samples: info["original_vitis_audit"])
    metrics = trainer.update(
        tape, -.05, info=info, env=env, next_obs=obs, terminated=False,
    )
    assert metrics["qualified"] is False
    assert metrics["hls_estimated_latency_ns"] is None
    assert metrics["shared_policy_updates"] == 1
    assert metrics["no_new_vendor_calls_by_trainer"] is True


def test_hw04_checkpoint_is_private_tensor_only_and_eval_restore(tmp_path):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset()
    agent = HardwareGraphTrainer(seed=123, hidden=16)
    chosen, _ = agent.choose(obs, training=False, sample=False)
    folder = tmp_path / "private_policy_01"
    agent.save_private(folder, {"scope": "TEST_ONLY_UNTRAINED"})
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["pretrained"] is False
    assert manifest["trained_on_original_vitis_hls"] is False
    assert manifest["full_training_resume_supported"] is False
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        agent.save_private(folder, {})
    restored = HardwareGraphTrainer(seed=999, hidden=16)
    with pytest.raises(ValueError, match="Exact training resume"):
        restored.load_private(folder)
    restored.load_private(folder, load_optimizer=False)
    second, tape = restored.choose(obs, training=False, sample=False)
    assert second == chosen
    assert tape is None
    with open(folder / "shared_hardware_gnn.npz", "ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="provenance mismatch"):
        restored.load_private(folder, load_optimizer=False)


def test_hw04_history_attention_ignores_masked_garbage(tmp_path):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset()
    policy = SharedHardwareGraphPolicy(hidden=16)
    x, value = policy(obs)
    altered = copy.deepcopy(obs)
    altered["history"][:] = 999.
    y, value2 = policy(altered)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
    assert torch.allclose(x[mask], y[mask])
    assert torch.allclose(value, value2)
    altered["history_mask"][0] = 1
    altered["history"][0] = 0.
    z, v3 = policy(altered)
    assert torch.isfinite(z[mask]).all()
    assert torch.isfinite(v3)
    assert not torch.allclose(x[mask], z[mask]) or not torch.allclose(value, v3)



def test_hw04_metadata_only_cannot_update_without_original_case(tmp_path):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset()
    trainer = HardwareGraphTrainer(seed=313, hidden=16)
    _, tape = trainer.choose(obs, training=True)
    info = _fake_transition_info(env, qualified=True)
    # A structure-matching dictionary never authorizes a gradient update
    # without the live original Gym state and a fresh case replay.
    with pytest.raises(ValueError, match="matching live native Gym"):
        trainer.update(tape, -.005, info=info, env=env)
    assert trainer.updates == 0


def test_hw04_native_replay_mismatch_rejects_optimizer_step(tmp_path, monkeypatch):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset()
    trainer = HardwareGraphTrainer(seed=311, hidden=16)
    _, tape = trainer.choose(obs, training=True)
    info = _fake_transition_info(env, qualified=True)
    env.calls = 1
    env.last_audit = info["original_vitis_audit"]
    env.history = [{
        "audit": env.last_audit, "action_index": tape.source_action_index,
        "clock_ns": 15., "reward": -.005,
        "best_ns": info["original_vitis_audit"]["hls_estimated_latency_ns"],
    }]
    tampered = dict(info["original_vitis_audit"])
    tampered["state_sha256"] = "0" * 64
    monkeypatch.setattr(hardware_native, "audit_native_case",
                        lambda case, state, samples: tampered)
    with pytest.raises(ValueError, match="Independent original Vitis replay differs"):
        trainer.update(tape, -.005, info=info, env=env)
    assert trainer.updates == 0



def test_hw04_rejects_reward_substitution_and_wrong_action_tape(tmp_path, monkeypatch):
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset()
    trainer = HardwareGraphTrainer(seed=271, hidden=16)
    _, tape = trainer.choose(obs, training=True)
    info = _fake_transition_info(env, qualified=True)
    env.calls = 1
    env.last_audit = info["original_vitis_audit"]
    env.history = [{
        "audit": env.last_audit, "action_index": tape.source_action_index,
        "clock_ns": 15., "reward": -.005,
        "best_ns": info["original_vitis_audit"]["hls_estimated_latency_ns"],
    }]
    monkeypatch.setattr(hardware_native, "audit_native_case",
                        lambda case, state, samples: info["original_vitis_audit"])
    with pytest.raises(ValueError, match="exact reward"):
        trainer.update(tape, +0.5, info=info, env=env)
    wrong = HardwareActionTape(
        tape.log_probability, tape.entropy, tape.value,
        tape.selected_context, tape.requested_clock_fraction,
        tape.predicted_feasibility,
        (tape.source_action_index + 1) % MAX_ACTIONS,
    )
    with pytest.raises(ValueError, match="exact reward"):
        trainer.update(wrong, -.005, info=info, env=env)
    assert trainer.updates == 0



def test_hw04_original_hls_diagnostics_forward_real_audit_without_proxy():
    """CLI diagnostics expose gate failures, never fabricate qualified ns."""
    from rl_environment.hardware_agent import original_hls_diagnostics

    native_audit = {
        "gate_failures": ["ORIGINAL_HLS_SCHEDULING_NOT_FEASIBLE"],
        "csim_passed": True,
        "csim_max_absolute_error": 0.0004,
        "raw_hls_worst_case_cycles": 646,
        "requested_clock_ns": 6.55,
        "hls_timing": {
            "hls_timing_feasible": False,
            "estimated_clock_period_ns": 10.2,
        },
        "original_hls_resource_consumed": {"lut": 10238},
        "resource_caps": {"lut": 53200},
        "qualified": False,
        "hls_estimated_latency_ns": None,
    }
    snapshot = copy.deepcopy(native_audit)
    reported = original_hls_diagnostics(native_audit)
    assert reported["gate_failures"] == ["ORIGINAL_HLS_SCHEDULING_NOT_FEASIBLE"]
    assert reported["raw_hls_worst_case_cycles"] == 646
    assert reported["requested_clock_ns"] == 6.55
    assert reported["hls_timing"]["hls_timing_feasible"] is False
    assert "hls_estimated_latency_ns" not in reported
    assert native_audit == snapshot


def test_hw04_untrained_agent_cli_preview_no_vitis_or_training(tmp_path, capsys):
    from rl_environment.hardware_agent import main
    original = save_model("cnn_pool_dense", tmp_path / "source.onnx")
    model = load_onnx(str(original))
    sample_file = tmp_path / "samples.npy"
    np.save(sample_file, np.zeros(
        (2, *model.tensor(model.inputs[0]).shape), dtype=np.float32
    ))
    root = tmp_path / "never_created_gnn_case"
    argv = [
        "--onnx", str(original), "--samples", str(sample_file),
        "--part", PART, "--resource-caps", *(str(x) for x in CAPS),
        "--max-absolute-error", ".1", "--initial-clock-ns", "14.0",
        "--seed", "101", "--hidden", "16", "--output-root", str(root),
    ]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "SHARED_GNN_SOURCE_ONLY_PREVIEW"
    assert result["original_vitis_calls"] == 0
    assert result["training_started"] is False
    assert result["online_gradient_updates"] == 0
    assert result["pretrained"] is False
    assert 2 <= result["requested_clock_ns"] <= 32
    assert not root.exists()
    assert sample_file.is_file()

def test_hw05_episodic_methods_audit_before_real_policy_updates(
        tmp_path, monkeypatch):
    """These are shape-only CPU fixtures; never paper/native evidence."""
    from rl_environment.hardware_algorithms import create_hardware_method
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset(seed=101)
    for method in ("reinforce", "ppo"):
        model = create_hardware_method(method, seed=101, hidden=16,
                                      ppo_epochs=2)
        action, tape = model.choose(obs, training=True)
        assert tape.observation is not None
        assert tape.sampled_clock_fraction is not None
        fake = _fake_transition_info(env, qualified=False)
        env.calls = 1
        env.last_audit = fake["original_vitis_audit"]
        env.history = [{
            "audit": env.last_audit, "action_index": tape.source_action_index,
            "clock_ns": action[1], "reward": -.05, "best_ns": None,
        }]
        monkeypatch.setattr(hardware_native, "audit_native_case",
                            lambda case, state, samples: fake["original_vitis_audit"])
        status = model.observe(tape, -.05, info=fake, env=env)
        assert status["buffered_steps"] == 1
        assert model.updates == 0
        with pytest.raises(ValueError, match="Interrupted"):
            model.finish_episode(complete=False)
        receipt = model.finish_episode()
        assert receipt["algorithm"] == method
        assert receipt["qualified_steps"] == 0
        assert receipt["audited_steps"] == 1
        assert model.updates == (2 if method == "ppo" else 1)
        assert all(isfinite(x["loss"]) for x in receipt["gradient_epochs"])
        assert not model.pending


def test_hw05_episodic_methods_reject_metadata_only_and_random_is_separate(
        tmp_path):
    from rl_environment.hardware_algorithms import (
        RandomHardwarePolicy, create_hardware_method,
    )
    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset(seed=101)
    a = create_hardware_method("random", seed=1001)
    b = RandomHardwarePolicy(seed=1001)
    assert a.choose(obs) == b.choose(obs)
    for method in ("reinforce", "ppo"):
        model = create_hardware_method(method, seed=101, hidden=16)
        _, tape = model.choose(obs, training=True)
        with pytest.raises(ValueError, match="Missing original-native"):
            model.observe(tape, -.05, info={}, env=env)
        assert model.updates == 0
        assert model.pending == []
    with pytest.raises(ValueError, match="Unsupported"):
        create_hardware_method("fake-ppo")



def test_hw05_frozen_ranked_fallback_keeps_gnn_deterministic_and_source_novel(
        tmp_path):
    """Frozen heldout evaluation can fill budgets without hidden RNG calls."""
    from rl_environment.hardware_campaign import _choose_admitted_action

    env = _env(tmp_path, "cnn_pool_dense")
    obs, _ = env.reset(seed=101)
    trainer = HardwareGraphTrainer(seed=101, hidden=24)
    before_rng = torch.get_rng_state().clone()
    ranked = trainer.ranked_frozen_actions(obs)
    initial_choice, tape = trainer.choose(obs, training=False, sample=False)
    assert tape is None
    assert ranked[0] == initial_choice
    assert len(ranked) == int(obs["action_mask"].sum())
    assert len({index for index, _ in ranked}) == len(ranked)
    assert all(obs["action_mask"][index] and 2. <= clock <= 32.
               for index, clock in ranked)
    assert torch.equal(torch.get_rng_state(), before_rng)
    first = env.preview_action(ranked[0])
    env.visited.add(first.state_sha256)
    action, accepted_tape, _, admitted, rejected = _choose_admitted_action(
        trainer, env, obs, train=False, stochastic_evaluation=False,
    )
    assert accepted_tape is None
    assert rejected >= 1
    assert action == ranked[rejected]
    assert admitted.state_sha256 != first.state_sha256
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert env.calls == 0
    assert not env.root.exists()
