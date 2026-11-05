"""Strict tensor-only private optimizer snapshots for original-native RL.

These files are *not* a license to resume an interrupted campaign or call
old cases new. They preserve a source-pinned shared GNN, Adam moments and
torch CPU sampling RNG at a COMPLETED episode boundary, for future gated
continuation work. No torch.load(), Python pickle, GPU state or vendor call.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from ..agents.hardware_agent import HardwareGraphTrainer, _require_torch, torch

SCHEMA = "fpgai-research/hardware-rl-training-snapshot/v1"
_ADAM_KEYS = frozenset(("step", "exp_avg", "exp_avg_sq"))
_SHA = frozenset("0123456789abcdef")


def _revision(revision: str) -> str:
    if (not isinstance(revision, str) or len(revision) != 40
            or any(c not in _SHA for c in revision)):
        raise ValueError("A full source-pinned 40-hex revision is required")
    return revision


def _method(trainer: HardwareGraphTrainer) -> str:
    if not isinstance(trainer, HardwareGraphTrainer):
        raise TypeError("Only the shared real-source hardware trainer can be saved")
    return getattr(trainer, "algorithm", "actor_critic")


def _optimizer_group(trainer):
    optimizer = trainer.optimizer
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("Only the exact supported Adam optimizer can be checkpointed")
    state = optimizer.state_dict()
    if len(state["param_groups"]) != 1:
        raise ValueError("Only one explicit Adam parameter group is supported")
    names = list(trainer.policy.state_dict())
    group = state["param_groups"][0]
    if group["params"] != list(range(len(list(trainer.policy.parameters())))):
        raise ValueError("Adam parameter order differs from named GNN parameters")
    kwargs = {
        k: v for k, v in group.items() if k != "params"
    }
    # Convert tuple hyperparameters to JSON-native arrays and reject opaque.
    kwargs = json.loads(json.dumps(kwargs, allow_nan=False))
    return state, names, kwargs


def save_training_state(trainer: HardwareGraphTrainer, directory: str | Path,
                        *, method: str, source_revision_sha: str,
                        metadata: dict) -> Path:
    """Save only at an audited completed-episode optimizer boundary.

    A separate CLI with exact source and case-journal continuity is still
    required before the state may continue a scientific training campaign.
    """
    _require_torch()
    if _method(trainer) != method:
        raise ValueError("Saved RL method differs from actual trainer class")
    _revision(source_revision_sha)
    if getattr(trainer, "pending", ()):
        raise ValueError("Unfinished on-policy trajectory cannot be checkpointed")
    if type(trainer.updates) is not int or trainer.updates <= 0:
        raise ValueError("An actual completed optimizer update is required")
    if not isinstance(metadata, dict):
        raise TypeError("Source/task/episode provenance metadata required")
    # Reject non-serializable and nonfinite metadata before creating any dir.
    metadata = json.loads(json.dumps(metadata, allow_nan=False, sort_keys=True))
    state, names, group = _optimizer_group(trainer)
    tensors = {}
    model_state = trainer.policy.state_dict()
    for i, name in enumerate(names):
        tensors[f"weight_{i:04d}"] = model_state[name].detach().cpu().numpy()
    optimizer_names = {}
    for index in range(len(list(trainer.policy.parameters()))):
        item = state["state"].get(index, {})
        if set(item) - _ADAM_KEYS:
            raise ValueError("Unsupported Adam optimizer state; cannot safely serialize")
        if item and set(item) != _ADAM_KEYS:
            raise ValueError("Partial Adam optimizer state cannot be serialized")
        optimizer_names[str(index)] = sorted(item)
        for key, tensor in item.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Only tensor-valued Adam moments are supported")
            tensors[f"opt_{index:04d}_{key}"] = tensor.detach().cpu().numpy()
    tensors["torch_cpu_rng"] = torch.get_rng_state().cpu().numpy()
    target = Path(directory).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Private original training snapshot already exists")
    target.mkdir(parents=True, exist_ok=False)
    data_path = target / "training_tensors.npz"
    np.savez_compressed(data_path, **tensors)
    manifest = {
        "schema": SCHEMA,
        "source_revision_sha": source_revision_sha,
        "algorithm": method,
        "model_state_names": names,
        "optimizer_tensor_names": optimizer_names,
        "optimizer_parameter_count": len(optimizer_names),
        "optimizer_group": group,
        "torch_version": torch.__version__,
        "seed": trainer.seed,
        "updates": trainer.updates,
        "hidden": trainer.policy.hidden,
        "message_passing": trainer.policy.message_passing,
        "trial_history": trainer.policy.use_trial_history,
        "array_keys": sorted(tensors),
        "tensor_file_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "metadata": metadata,
        "private_training_state": True,
        "snapshot_itself_does_not_reaudit_vendor": True,
        "source_native_replay_required_for_publication": True,
        "optimizer_and_sampling_rng_included": True,
        "automatic_native_campaign_resume_authorized": False,
        "physical_fpga_measurements": False,
    }
    with (target / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    return target


def load_training_state(trainer: HardwareGraphTrainer, directory: str | Path,
                        *, method: str, source_revision_sha: str,
                        expected_metadata: dict | None = None) -> dict:
    """Load a local trusted, hash-validated tensor snapshot, without pickle.

    No automatic native continuation. User/driver must separately attest
    original case journals and remaining call budget before a new study.
    """
    _require_torch()
    _revision(source_revision_sha)
    if _method(trainer) != method:
        raise ValueError("Current trainer method does not match checkpoint")
    root = Path(directory).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing or symlinked private training-state directory")
    manifest_path, path = root / "manifest.json", root / "training_tensors.npz"
    if (manifest_path.is_symlink() or path.is_symlink()
            or not manifest_path.is_file() or not path.is_file()):
        raise ValueError("Private checkpoint manifest/tensors missing or symlinked")
    if manifest_path.stat().st_size > 1_000_000 or path.stat().st_size > 128*1024*1024:
        raise ValueError("Unsafe private checkpoint size")
    m = json.loads(manifest_path.read_text())
    state, names, group = _optimizer_group(trainer)
    if (m.get("schema") != SCHEMA
            or m.get("source_revision_sha") != source_revision_sha
            or m.get("algorithm") != method
            or m.get("model_state_names") != names
            or m.get("optimizer_group") != group
            or m.get("torch_version") != torch.__version__
            or m.get("seed") != trainer.seed
            or m.get("hidden") != trainer.policy.hidden
            or m.get("message_passing") is not trainer.policy.message_passing
            or m.get("trial_history") is not trainer.policy.use_trial_history
            or m.get("optimizer_parameter_count") != len(list(trainer.policy.parameters()))
            or m.get("automatic_native_campaign_resume_authorized") is not False
            or sha256(path.read_bytes()).hexdigest() != m.get("tensor_file_sha256")):
        raise ValueError("Private full optimizer/RNG/source provenance mismatch")
    if expected_metadata is not None and m.get("metadata") != expected_metadata:
        raise ValueError("Original native episode/task metadata mismatch")
    with np.load(path, allow_pickle=False) as stored:
        if sorted(stored.files) != m.get("array_keys"):
            raise ValueError("Private checkpoint tensor inventory mismatch")
        expected = trainer.policy.state_dict()
        weights = {}
        for i, name in enumerate(names):
            array = stored[f"weight_{i:04d}"]
            if array.shape != tuple(expected[name].shape):
                raise ValueError("Private GNN tensor shape mismatch")
            if array.dtype != expected[name].detach().cpu().numpy().dtype:
                raise ValueError("Private GNN tensor dtype mismatch")
            weights[name] = torch.from_numpy(array.copy())
        optimizer_state = {}
        for key, tensor_names in m["optimizer_tensor_names"].items():
            index = int(key)
            if str(index) != key or not 0 <= index < len(list(trainer.policy.parameters())):
                raise ValueError("Unsupported optimizer parameter ID")
            if tensor_names and set(tensor_names) != _ADAM_KEYS:
                raise ValueError("Missing optimizer moments in private checkpoint")
            optimizer_state[index] = {
                name: torch.from_numpy(
                    stored[f"opt_{index:04d}_{name}"].copy()
                )
                for name in tensor_names
            }
        if set(optimizer_state) != set(range(len(list(trainer.policy.parameters())))):
            raise ValueError("Missing original optimizer parameter slots")
        rng = stored["torch_cpu_rng"]
        if rng.dtype != np.uint8 or rng.shape != tuple(torch.get_rng_state().shape):
            raise ValueError("Private torch CPU sampling RNG shape mismatch")
        rng_tensor = torch.from_numpy(rng.copy())
    if type(m.get("updates")) is not int or m["updates"] < 0:
        raise ValueError("Invalid saved optimizer update count")
    trainer.policy.load_state_dict(weights, strict=True)
    # IDs are indexed and shape-matched in the exact parameter order;
    # names are verified before calling torch's order-based optimizer load.
    state["state"] = optimizer_state
    trainer.optimizer.load_state_dict(state)
    trainer.updates = m["updates"]
    torch.set_rng_state(rng_tensor)
    return m
