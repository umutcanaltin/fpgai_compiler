from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import copy
import json

from fpgai.util.fs import write_text


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = copy.deepcopy(base)
        for k, v in override.items():
            out[k] = _deep_merge(out.get(k), v)
        return out
    return copy.deepcopy(override if override is not None else base)


@dataclass(frozen=True)
class TrainingOpCaps:
    forward: bool = True
    backward_input: bool = False
    backward_params: bool = False
    update: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "forward": self.forward,
            "backward_input": self.backward_input,
            "backward_params": self.backward_params,
            "update": self.update,
        }


OP_TRAINING_CAPS: Dict[str, TrainingOpCaps] = {
    "Dense": TrainingOpCaps(True, True, True, True),
    "Conv": TrainingOpCaps(True, True, True, True),
    "BatchNormalization": TrainingOpCaps(True, True, True, True),
    "Relu": TrainingOpCaps(True, True, False, False),
    "LeakyRelu": TrainingOpCaps(True, True, False, False),
    "Sigmoid": TrainingOpCaps(True, True, False, False),
    "Softmax": TrainingOpCaps(True, True, False, False),
    "Add": TrainingOpCaps(True, True, False, False),
    "MaxPool": TrainingOpCaps(True, True, False, False),
    "AvgPool": TrainingOpCaps(True, True, False, False),
    "Flatten": TrainingOpCaps(True, True, False, False),
    "Reshape": TrainingOpCaps(True, True, False, False),
}


@dataclass(frozen=True)
class TrainingPlan:
    optimizer_type: str
    learning_rate: float
    loss_type: str
    batch_size: int
    epochs: int

    weights_mode: str
    weight_storage: str
    activation_storage: str
    gradient_storage: str
    optimizer_state_storage: str

    movement_policy: Dict[str, Any]
    numerics: Dict[str, Any]
    cache_policy: Dict[str, Any]
    phase_overrides: Dict[str, Any]
    estimator: Dict[str, Any]
    planner_policy: Dict[str, Any]

    op_sequence: List[str]
    op_capabilities: Dict[str, Dict[str, Any]]

    parameter_trainable_ops: List[str]
    backward_only_ops: List[str]
    unsupported_ops: List[str]
    fully_supported_ops: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "loss_type": self.loss_type,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weights_mode": self.weights_mode,
            "weight_storage": self.weight_storage,
            "activation_storage": self.activation_storage,
            "gradient_storage": self.gradient_storage,
            "optimizer_state_storage": self.optimizer_state_storage,
            "movement_policy": self.movement_policy,
            "numerics": self.numerics,
            "cache_policy": self.cache_policy,
            "phase_overrides": self.phase_overrides,
            "estimator": self.estimator,
            "planner_policy": self.planner_policy,
            "op_sequence": self.op_sequence,
            "op_capabilities": self.op_capabilities,
            "parameter_trainable_ops": self.parameter_trainable_ops,
            "backward_only_ops": self.backward_only_ops,
            "unsupported_ops": self.unsupported_ops,
            "fully_supported_ops": self.fully_supported_ops,
        }


def _default_training_cache_policy() -> Dict[str, Any]:
    return {
        "store_forward_activations": True,
        "store_pre_activations": True,
        "store_pool_indices": True,
        "gradient_checkpointing": False,
        "store_step_snapshots": True,
    }


def _default_estimator_cfg() -> Dict[str, Any]:
    return {
        "enabled": True,
        "include_forward": True,
        "include_backward_input": True,
        "include_backward_params": True,
        "include_update": True,
        "include_buffers": True,
    }


def _resolve_training_numerics(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "activation": _cfg_get(raw_cfg, "numerics.defaults.activation"),
        "weight": _cfg_get(raw_cfg, "numerics.defaults.weight"),
        "bias": _cfg_get(raw_cfg, "numerics.defaults.bias"),
        "accum": _cfg_get(raw_cfg, "numerics.defaults.accum"),
    }
    training_defaults = {
        "grad": defaults["accum"],
        "grad_accum": defaults["accum"],
        "master_weight": defaults["weight"],
        "optimizer_state": defaults["accum"],
    }
    training_override = _cfg_get(raw_cfg, "numerics.training", {}) or {}
    return {
        "forward": defaults,
        "training": _deep_merge(training_defaults, training_override),
    }


def _classify_caps(caps: TrainingOpCaps) -> str:
    if caps.backward_params and caps.update:
        return "parameter_trainable"
    if caps.backward_input:
        return "backward_only"
    return "unsupported"


def _resolve_storage_policy(raw_cfg: Dict[str, Any], compile_plan=None, memory_plan=None) -> Dict[str, str]:
    notes = getattr(compile_plan, "notes", {}) if compile_plan is not None else {}
    mem_notes = getattr(memory_plan, "notes", {}) if memory_plan is not None else {}

    weights_mode = str(
        _cfg_get(
            raw_cfg,
            "data_movement.ps_pl.weights.mode",
            notes.get("global_weights_mode_requested", "embedded"),
        )
    ).lower()

    weight_pref = notes.get("weight_region_preference") or ["BRAM"]
    act_pref = notes.get("activation_region_preference") or ["BRAM"]

    weight_storage = str(_cfg_get(raw_cfg, "training.storage.weights", weight_pref[0])).lower()
    activation_storage = str(_cfg_get(raw_cfg, "training.storage.activations", act_pref[0])).lower()
    gradient_storage = str(_cfg_get(raw_cfg, "training.storage.gradients", activation_storage)).lower()
    optimizer_state_storage = str(_cfg_get(raw_cfg, "training.storage.optimizer_state", weight_storage)).lower()

    return {
        "weights_mode": weights_mode,
        "weight_storage": weight_storage,
        "activation_storage": activation_storage,
        "gradient_storage": gradient_storage,
        "optimizer_state_storage": optimizer_state_storage,
    }


def _resolve_movement_policy(raw_cfg: Dict[str, Any], communication_plan=None) -> Dict[str, Any]:
    cp_notes = getattr(communication_plan, "notes", {}) if communication_plan is not None else {}
    return {
        "ps_pl": _cfg_get(raw_cfg, "data_movement.ps_pl", {}) or {},
        "pl_ps": _cfg_get(raw_cfg, "data_movement.pl_ps", {}) or {},
        "compression": _cfg_get(raw_cfg, "data_movement.compression", {}) or {},
        "planner_axi_word_bits": cp_notes.get("axi_word_bits"),
        "planner_burst_len": cp_notes.get("burst_len"),
        "planner_enable_bitpack": cp_notes.get("enable_bitpack"),
        "planner_enable_compression": cp_notes.get("enable_compression"),
    }


def _planner_policy_dict(compile_plan=None, memory_plan=None, communication_plan=None) -> Dict[str, Any]:
    cnotes = getattr(compile_plan, "notes", {}) if compile_plan is not None else {}
    mnotes = getattr(memory_plan, "notes", {}) if memory_plan is not None else {}
    comnotes = getattr(communication_plan, "notes", {}) if communication_plan is not None else {}
    return {
        "parallel_policy": cnotes.get("parallel_policy"),
        "parallel_pe": cnotes.get("parallel_pe"),
        "parallel_simd": cnotes.get("parallel_simd"),
        "parallel_unroll_factor": cnotes.get("parallel_unroll_factor"),
        "parallel_partition_factor": cnotes.get("parallel_partition_factor"),
        "parallel_pipeline_style": cnotes.get("parallel_pipeline_style"),
        "weight_region_preference": cnotes.get("weight_region_preference"),
        "activation_region_preference": cnotes.get("activation_region_preference"),
        "allow_double_buffer": cnotes.get("allow_double_buffer", mnotes.get("allow_double_buffer")),
        "axi_word_bits": comnotes.get("axi_word_bits", cnotes.get("axi_word_bits")),
        "burst_len": comnotes.get("burst_len", cnotes.get("burst_len")),
        "enable_bitpack": comnotes.get("enable_bitpack", cnotes.get("enable_bitpack")),
        "enable_compression": comnotes.get("enable_compression", cnotes.get("enable_compression")),
        "array_partition_mode": cnotes.get("array_partition_mode"),
        "mac_style": cnotes.get("mac_style"),
        "accum_strategy": cnotes.get("accum_strategy"),
        "activation_impl": cnotes.get("activation_impl"),
        "round_mode": cnotes.get("round_mode"),
        "sat_mode": cnotes.get("sat_mode"),
    }


def build_training_plan(graph, raw_cfg: Dict[str, Any], compile_plan=None, memory_plan=None, communication_plan=None) -> TrainingPlan:
    optimizer_type = str(_cfg_get(raw_cfg, "training.optimizer.type", "sgd")).lower()
    learning_rate = float(_cfg_get(raw_cfg, "training.optimizer.learning_rate", 0.01))
    loss_type = str(_cfg_get(raw_cfg, "training.loss.type", "mse")).lower()
    batch_size = int(_cfg_get(raw_cfg, "training.execution.batch_size", 1))
    epochs = int(_cfg_get(raw_cfg, "training.execution.epochs", 1))

    storage = _resolve_storage_policy(raw_cfg, compile_plan=compile_plan, memory_plan=memory_plan)
    movement_policy = _resolve_movement_policy(raw_cfg, communication_plan=communication_plan)
    planner_policy = _planner_policy_dict(compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan)

    cache_policy = _deep_merge(_default_training_cache_policy(), _cfg_get(raw_cfg, "training.cache", {}) or {})
    phase_overrides = _cfg_get(raw_cfg, "training.phase_overrides", {}) or {}
    numerics = _resolve_training_numerics(raw_cfg)
    estimator = _deep_merge(_default_estimator_cfg(), _cfg_get(raw_cfg, "training.estimator", {}) or {})

    op_sequence: List[str] = []
    op_capabilities: Dict[str, Dict[str, Any]] = {}
    parameter_trainable_ops: List[str] = []
    backward_only_ops: List[str] = []
    unsupported_ops: List[str] = []
    fully_supported_ops: List[str] = []

    plan_map = {}
    if compile_plan is not None and hasattr(compile_plan, "layer_plans"):
        for lp in compile_plan.layer_plans:
            plan_map[lp.node_name] = lp.to_dict() if hasattr(lp, "to_dict") else lp

    for idx, op in enumerate(getattr(graph, "ops", [])):
        op_type = str(getattr(op, "op_type", "") or "")
        op_name = str(getattr(op, "name", f"{op_type}_{idx}") or f"{op_type}_{idx}")
        caps = OP_TRAINING_CAPS.get(op_type, TrainingOpCaps(False, False, False, False))
        cls = _classify_caps(caps)

        op_sequence.append(op_type)
        op_capabilities[op_name] = {
            "op_type": op_type,
            "caps": caps.to_dict(),
            "classification": cls,
            "planner": plan_map.get(op_name),
        }

        if cls == "parameter_trainable":
            parameter_trainable_ops.append(op_type)
            fully_supported_ops.append(op_type)
        elif cls == "backward_only":
            backward_only_ops.append(op_type)
            fully_supported_ops.append(op_type)
        else:
            unsupported_ops.append(op_type)

    return TrainingPlan(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        loss_type=loss_type,
        batch_size=batch_size,
        epochs=epochs,
        weights_mode=storage["weights_mode"],
        weight_storage=storage["weight_storage"],
        activation_storage=storage["activation_storage"],
        gradient_storage=storage["gradient_storage"],
        optimizer_state_storage=storage["optimizer_state_storage"],
        movement_policy=movement_policy,
        numerics=numerics,
        cache_policy=cache_policy,
        phase_overrides=phase_overrides,
        estimator=estimator,
        planner_policy=planner_policy,
        op_sequence=op_sequence,
        op_capabilities=op_capabilities,
        parameter_trainable_ops=parameter_trainable_ops,
        backward_only_ops=backward_only_ops,
        unsupported_ops=unsupported_ops,
        fully_supported_ops=fully_supported_ops,
    )


def emit_training_artifacts(out_dir: Path, plan: TrainingPlan) -> Path:
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    json_path = training_dir / "training_plan.json"
    txt_path = training_dir / "summary.txt"

    write_text(json_path, json.dumps(plan.to_dict(), indent=2))

    lines: List[str] = []
    lines.append("=============== FPGAI Training Plan ===============")
    lines.append(f"optimizer_type           : {plan.optimizer_type}")
    lines.append(f"learning_rate            : {plan.learning_rate}")
    lines.append(f"loss_type                : {plan.loss_type}")
    lines.append(f"batch_size               : {plan.batch_size}")
    lines.append(f"epochs                   : {plan.epochs}")
    lines.append(f"weights_mode             : {plan.weights_mode}")
    lines.append(f"weight_storage           : {plan.weight_storage}")
    lines.append(f"activation_storage       : {plan.activation_storage}")
    lines.append(f"gradient_storage         : {plan.gradient_storage}")
    lines.append(f"optimizer_state_storage  : {plan.optimizer_state_storage}")
    lines.append(f"parameter_trainable_ops  : {sorted(set(plan.parameter_trainable_ops))}")
    lines.append(f"backward_only_ops        : {sorted(set(plan.backward_only_ops))}")
    lines.append(f"unsupported_ops          : {sorted(set(plan.unsupported_ops))}")
    lines.append(f"fully_supported_ops      : {sorted(set(plan.fully_supported_ops))}")
    lines.append("planner_policy           :")
    for k, v in sorted(plan.planner_policy.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("cache_policy             :")
    for k, v in sorted(plan.cache_policy.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("training_numerics        :")
    for k, v in sorted((plan.numerics.get('training') or {}).items()):
        lines.append(f"  - {k}: {v}")
    lines.append("movement_policy          :")
    for k, v in sorted(plan.movement_policy.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("estimator                :")
    for k, v in sorted(plan.estimator.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("===================================================")

    write_text(txt_path, "\n".join(lines))
    return json_path