"""HW-RL-05 source-only scientific campaign tests (no licensed vendor).

Synthetic stubs exercise control-flow safety only; they are NOT original
Vitis receipts and must never appear as measured paper evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl_environment import hardware_campaign as campaign


def _manifest(tmp_path: Path, *, heldout: bool = False):
    # Read-only planning never needs to load a model or invoke Vitis.
    (tmp_path / "train.onnx").write_bytes(b"shape-only ONNX plan fixture")
    (tmp_path / "train.npy").write_bytes(b"fake samples for file-hash tests")
    tasks = [{
        "id": "training_network_001", "split": "train",
        "onnx": "train.onnx", "samples": "train.npy",
        "part": "xc7z020clg400-1",
        "resource_caps": [53200, 106400, 220, 280, 0],
        "max_absolute_error": 0.1,
    }]
    if heldout:
        # Reusing a model, even under another filename, leaks the train split.
        (tmp_path / "val.onnx").write_bytes(b"shape-only ONNX plan fixture")
        (tmp_path / "val.npy").write_bytes(b"separate sample bytes")
        tasks.append({
            **tasks[0], "id": "heldout_network_001", "split": "test",
            "onnx": "val.onnx", "samples": "val.npy"
        })
    path = tmp_path / "hardware_tasks.json"
    path.write_text(json.dumps({"schema": campaign.TASK_SCHEMA, "tasks": tasks}))
    return path


def test_hw05_plan_is_pure_and_method_schedule_is_matched(tmp_path, capsys):
    path = _manifest(tmp_path)
    tasks = campaign.read_tasks(path)
    assert tasks[0].resource_caps == (53200, 106400, 220, 280, 0)
    assert tasks[0].metadata()["resource_caps"] == [53200, 106400, 220, 280, 0]
    assert campaign.main([
        "--tasks", str(path), "--episodes", "3",
        "--calls-per-episode", "2", "--method", "ppo",
        "--seed", "101",
    ]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["planned_max_original_vitis_calls"] == 6
    assert preview["method"] == "ppo"
    assert preview["native_execution_authorized"] is False
    assert preview["unique_network_task_ids"] == 1
    assert preview["task_repetitions"] == 2
    for method in ("actor_critic", "reinforce", "random"):
        matched = campaign.campaign_plan(
            tasks=tasks, split="train", method=method, episodes=3,
            calls_per_episode=2, total_authorized_calls=None,
            seed=101, evaluation=False, checkpoint=None,
            hidden=32, ppo_epochs=3, ppo_clip=.2,
        )
        assert matched["schedule"] == preview["schedule"]
    assert not (tmp_path / "build").exists()


def test_hw05_manifest_rejects_heldout_data_leakage(tmp_path):
    with pytest.raises(ValueError, match="Identical ONNX"):
        campaign.read_tasks(_manifest(tmp_path, heldout=True))


@pytest.mark.parametrize("method", ["actor_critic", "reinforce", "ppo"])
def test_hw05_frozen_learned_methods_require_private_checkpoint(tmp_path, method):
    tasks = campaign.read_tasks(_manifest(tmp_path))
    with pytest.raises(ValueError, match="checkpoint"):
        campaign.campaign_plan(
            tasks=tasks, split="train", method=method, episodes=1,
            calls_per_episode=2, total_authorized_calls=None,
            seed=101, evaluation=True, checkpoint=None,
            hidden=32, ppo_epochs=3, ppo_clip=.2,
        )


def test_hw05_native_requires_exact_budget_and_fresh_root(tmp_path, capsys):
    path = _manifest(tmp_path)
    root = tmp_path / "never-created-by-unauthorized-call"
    assert campaign.main([
        "--tasks", str(path), "--episodes", "2",
        "--calls-per-episode", "3",
        "--output-root", str(root), "--execute-native",
    ]) == 2
    assert not root.exists()
    tasks = campaign.read_tasks(path)
    with pytest.raises(ValueError, match="budget"):
        campaign.campaign_plan(
            tasks=tasks, split="train", method="actor_critic",
            episodes=2, calls_per_episode=3,
            total_authorized_calls=5, seed=101, evaluation=False,
            checkpoint=None, hidden=64, ppo_epochs=3, ppo_clip=.2,
        )


def test_hw05_stubbed_random_campaign_only_proves_journal_plumbing(
        tmp_path, monkeypatch):
    """A fake Gym is explicitly *not* evidence of Vitis result integrity."""
    tasks = campaign.read_tasks(_manifest(tmp_path))
    plan = campaign.campaign_plan(
        tasks=tasks, split="train", method="random", episodes=1,
        calls_per_episode=2, total_authorized_calls=2, seed=101,
        evaluation=False, checkpoint=None, hidden=16,
        ppo_epochs=3, ppo_clip=.2,
    )

    class FakeEnvironment:
        def __init__(self, **kwargs):
            self.max_calls = kwargs["calls_per_episode"]
            self.calls = 0
            self.offers = (None,)
            self.history = []
            self.observation = {
                "action_mask": np.asarray([1], np.int8),
                "clock_bounds_ns": np.asarray([2., 32.], np.float32),
            }

        def preview_action(self, action):
            return SimpleNamespace(state_sha256=f"fake-state-{self.calls + 1}")

        def reset(self, *, seed):
            return self.observation, {"state_sha256": "cpu-fixture"}

        def step(self, action):
            self.calls += 1
            audit = {
                "gate_failures": ["ORIGINAL_HLS_SCHEDULING_NOT_FEASIBLE"],
                "csim_passed": True,
                "csim_max_absolute_error": 0.0004,
                "raw_hls_worst_case_cycles": 780,
                "requested_clock_ns": action[1],
                "hls_timing": {"hls_timing_feasible": False},
                "original_hls_resource_consumed": {},
                "resource_caps": {},
                "qualified": False,
                "hls_estimated_latency_ns": None,
            }
            self.history.append({"audit": audit, "reward": -.05})
            info = {
                "original_vitis_audit": audit,
                "state_sha256": f"fake-state-{self.calls}",
                "native_case": f"FAKE_CPU_CASE_{self.calls}",
                "qualified": False, "best_hls_estimated_latency_ns": None,
            }
            return self.observation, -.05, False, (
                self.calls == self.max_calls
            ), info

    monkeypatch.setattr(
        campaign, "_environment",
        lambda task, **kwargs: FakeEnvironment(**kwargs),
    )
    monkeypatch.setattr(campaign, "_source_revision", lambda: "f" * 40)
    root = tmp_path / "synthetic_pipeline_only"
    receipt = campaign.run_campaign(
        tasks=tasks, plan=plan, output_root=root, authorized_calls=2,
        checkpoint=None, vitis="DO_NOT_CALL_VITIS", timeout=1,
        stochastic_evaluation=False,
    )
    assert receipt["native_calls"] == 2
    assert receipt["status"] == "COMPLETE_PENDING_INDEPENDENT_CAMPAIGN_AUDIT"
    assert len((root / "trials.jsonl").read_text().splitlines()) == 2
    assert len((root / "episodes.jsonl").read_text().splitlines()) == 1
    with pytest.raises(FileExistsError, match="already exists"):
        campaign.run_campaign(
            tasks=tasks, plan=plan, output_root=root, authorized_calls=2,
            checkpoint=None, vitis="DO_NOT_CALL_VITIS", timeout=1,
            stochastic_evaluation=False,
        )


def test_hw05_source_admission_retries_novelty_without_native_calls():
    """No-op and duplicate proposals are discarded before Gym.step/Vitis."""
    class SequencePolicy(campaign.RandomHardwarePolicy):
        def __init__(self):
            super().__init__(seed=101)
            self.actions = iter(((0, 14.0), (0, 15.0)))

        def choose(self, observation):
            return next(self.actions)

    class SourceOnly:
        offers = (None,)
        native_calls = 0

        def preview_action(self, action):
            if action[1] == 14.0:
                raise ValueError("No-op architecture-clock action; native case not started")
            return SimpleNamespace(state_sha256="novel-state")

    env = SourceOnly()
    action, tape, selected, preview, rejected = campaign._choose_admitted_action(
        SequencePolicy(), env, {"action_mask": np.asarray([1])},
        train=False, stochastic_evaluation=False,
    )
    assert (action, tape, selected, preview.state_sha256, rejected) == (
        (0, 15.0), None, None, "novel-state", 1,
    )
    assert env.native_calls == 0


def test_hw05_source_admission_fails_closed_when_all_proposals_repeat():
    class NeverNovel(campaign.RandomHardwarePolicy):
        def choose(self, observation):
            return 0, 14.0

    class SourceOnly:
        offers = (None,)

        def preview_action(self, action):
            raise ValueError("Repeated complete hardware state; no new native case")

    with pytest.raises(RuntimeError, match="after 2 source-only attempts"):
        campaign._choose_admitted_action(
            NeverNovel(seed=101), SourceOnly(), {},
            train=False, stochastic_evaluation=False,
            max_source_proposals=2,
        )


def test_hw05_trial_receipt_binds_exact_pre_step_edit(tmp_path):
    """The next state's re-enumerated offers must not corrupt action history."""
    from rl_environment.hardware_space import HardwareEdit

    task = campaign.read_tasks(_manifest(tmp_path))[0]
    original = HardwareEdit("operator", "pes", 2, 0)
    audit = {
        "qualified": False,
        "hls_estimated_latency_ns": None,
        "gate_failures": ["ORIGINAL_HLS_SCHEDULING_NOT_FEASIBLE"],
        "csim_passed": True,
        "csim_max_absolute_error": 0.0004,
        "raw_hls_worst_case_cycles": 780,
        "requested_clock_ns": 15.0,
        "hls_timing": {"hls_timing_feasible": False},
        "original_hls_resource_consumed": {},
        "resource_caps": {},
    }
    receipt = campaign._trial_record(
        0, 0, task, source_action=original, action=(2, 15.0),
        reward=-.05, info={
            "original_vitis_audit": audit,
            "state_sha256": "pre-step-result",
            "native_case": "CPU_NOT_VENDOR",
            "qualified": False,
            "best_hls_estimated_latency_ns": None,
        }, gradient=None, method="random",
        rejected_source_proposals=3,
    )
    assert receipt["source_action"] == {
        "scope": "operator", "field": "pes", "value": 2,
        "operator_index": 0,
    }
    assert receipt["source_action_index"] == 2
    assert receipt["rejected_source_proposals_before_native"] == 3
