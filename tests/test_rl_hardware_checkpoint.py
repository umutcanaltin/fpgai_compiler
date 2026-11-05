"""Private tensor-only Adam/RNG snapshots; test data is NOT native Vitis."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rl_environment.hardware_algorithms import create_hardware_method
from rl_environment.hardware_checkpoint import (
    save_training_state, load_training_state,
)


def _cpu_optimizer_step_only(trainer):
    """Fixture initializes Adam moments; not a source-attested HLS update."""
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = sum(v.square().sum() for v in trainer.policy.parameters()) * .000001
    loss.backward()
    trainer.optimizer.step()
    trainer.updates += 1


def test_hw05_private_tensor_adam_state_and_rng_roundtrip(tmp_path):
    original = create_hardware_method("actor_critic", seed=101, hidden=16)
    _cpu_optimizer_step_only(original)
    root = tmp_path / "private_training_snapshot"
    revision = "a" * 40
    provenance = {"scope": "CPU_TEST_FIXTURE_ONLY", "episode": 0}
    save_training_state(
        original, root, method="actor_critic",
        source_revision_sha=revision, metadata=provenance,
    )
    expected_rng = torch.rand(5)
    restored = create_hardware_method("actor_critic", seed=101, hidden=16)
    manifest = load_training_state(
        restored, root, method="actor_critic",
        source_revision_sha=revision, expected_metadata=provenance,
    )
    assert manifest["optimizer_and_sampling_rng_included"] is True
    assert manifest["automatic_native_campaign_resume_authorized"] is False
    assert restored.updates == original.updates == 1
    assert torch.equal(expected_rng, torch.rand(5))
    for name, weight in original.policy.state_dict().items():
        assert torch.equal(restored.policy.state_dict()[name], weight)
    for pa, pb in zip(original.policy.parameters(), restored.policy.parameters()):
        sa = original.optimizer.state[pa]
        sb = restored.optimizer.state[pb]
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_hw05_private_training_checkpoint_rejects_tampering(tmp_path):
    agent = create_hardware_method("reinforce", seed=101, hidden=16)
    _cpu_optimizer_step_only(agent)
    root = tmp_path / "training_state"
    save_training_state(
        agent, root, method="reinforce",
        source_revision_sha="b"*40,
        metadata={"episode": 1, "fixture": True},
    )
    other = create_hardware_method("reinforce", seed=101, hidden=16)
    with pytest.raises(ValueError, match="provenance"):
        load_training_state(
            other, root, method="reinforce",
            source_revision_sha="c"*40,
        )
    with pytest.raises(FileExistsError, match="already exists"):
        save_training_state(
            agent, root, method="reinforce",
            source_revision_sha="b"*40, metadata={},
        )
    with open(root / "training_tensors.npz", "ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="provenance"):
        load_training_state(
            other, root, method="reinforce",
            source_revision_sha="b"*40,
        )


def test_hw05_wrong_algorithm_cannot_restore_optimizer_state(tmp_path):
    original = create_hardware_method("actor_critic", seed=101, hidden=16)
    _cpu_optimizer_step_only(original)
    directory = tmp_path / "training_state"
    save_training_state(
        original, directory, method="actor_critic",
        source_revision_sha="a"*40, metadata={"scope": "cpu"},
    )
    other = create_hardware_method("ppo", seed=101, hidden=16)
    with pytest.raises(ValueError, match="method"):
        load_training_state(
            other, directory, method="actor_critic",
            source_revision_sha="a"*40,
        )
