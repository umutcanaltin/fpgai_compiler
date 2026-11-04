"""Recovery-unit tests deliberately avoid ALL Vitis and real source evidence.

These test preflight accounting/immutability only; real scientific qualification
requires the independently original-native audited evaluation result.
"""
import json
from hashlib import sha256
from pathlib import Path

import pytest

from rl_environment.gym_eval_recovery import run, training_reference


def _build_interrupted_fixture(root):
    root.mkdir()
    seed = 101
    original = "a" * 40
    (root / "campaign.json").write_text(json.dumps({
        "schema": "fpgai-gym-native-plan/v1",
        "training_seeds": [seed],
        "source_revision": original,
        "message_passing_enabled": True,
        "training_networks_per_seed": 100,
        "validation_networks_per_seed": 1,
        "test_networks_per_seed": 1,
        "calls_per_episode_max": 10,
        "fixed_clock_ablation_ns": None,
    }))
    checkpoint = root / "seed_101" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "shared_gnn_policy.npz").write_bytes(b"fake-test-policy")
    (checkpoint / "optimizer_private.pt").write_bytes(b"fake-test-optimizer")
    (checkpoint / "manifest.json").write_text(json.dumps({
        "source_revision": original,
        "online_updates": 1000,
        "train_task_shas": ["b" * 64] * 100,
        "policy_sha256": sha256(b"fake-test-policy").hexdigest(),
        "optimizer_sha256": sha256(b"fake-test-optimizer").hexdigest(),
    }))
    with ((root / "completed_episodes.jsonl").open("w")) as journal, \
         ((root / "native_steps.jsonl").open("w")) as trials:
        for n in range(100):
            task = f"train_{n:05d}"
            episode = root / "seed_101" / "train" / task / "episode_000000"
            episode.mkdir(parents=True)
            journal.write(json.dumps({
                "method": "online_train", "task_id": task,
                "private_episode_path": str(episode),
                "original_vitis_calls": 10,
            }) + "\n")
            for k in range(10):
                trials.write(json.dumps({
                    "method": "online_train", "task_id": task,
                    "native_call_index": k,
                }) + "\n")
    return root


def test_interrupted_100x10_training_is_not_reexecuted_in_dry_run(tmp_path):
    study = _build_interrupted_fixture(tmp_path / "original")
    before = sorted((p.relative_to(study), p.stat().st_size)
                    for p in study.rglob("*") if p.is_file())
    details = training_reference(study)
    assert details["completed_training_episodes"] == 100
    assert details["original_training_calls"] == 1000
    assert details["original_journal_rows"] == 1000
    target = tmp_path / "unused_eval"
    plan = run(training_study=study, output=target, execute_native=False)
    assert plan["new_evaluation_episodes"] == 9
    assert plan["new_evaluation_max_vitis_calls"] == 90
    assert plan["old_training_original_vitis_calls"] == 1000
    assert not target.exists()
    with pytest.raises(ValueError, match="budget"):
        run(training_study=study, output=target, execute_native=True,
            max_total_vitis_calls=89)
    assert not target.exists()
    after = sorted((p.relative_to(study), p.stat().st_size)
                   for p in study.rglob("*") if p.is_file())
    assert before == after


def test_recovery_refuses_training_without_all_100_original_episodes(tmp_path):
    root = _build_interrupted_fixture(tmp_path / "partial")
    journal = root / "completed_episodes.jsonl"
    journal.write_text("\n".join(journal.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="100 complete"):
        training_reference(root)


def test_recovery_refuses_tampered_original_checkpoint(tmp_path):
    root = _build_interrupted_fixture(tmp_path / "tamper")
    (root / "seed_101" / "checkpoint" / "shared_gnn_policy.npz").write_bytes(
        b"tampered-test-policy")
    with pytest.raises(ValueError, match="checkpoint provenance/hash"):
        training_reference(root)
