from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import json
import hashlib

from fpgai.config.access import get_path
from fpgai.util.fs import write_text
from fpgai.validation.capture_schema import (
    NumericCaptureContract,
    default_training_capture_requirements,
    write_capture_contract,
)


_cfg_get = get_path



def _cfg_weight_load_interface(raw_cfg: Dict[str, Any], default: str = "embedded") -> str:
    value = _cfg_get(raw_cfg, "data_movement.weights.load.interface", None)
    if value is None:
        value = _cfg_get(raw_cfg, "data_movement.ps_pl.weights.mode", default)
    return str(value or default).lower().replace("-", "_")

def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = copy.deepcopy(base)
        for k, v in override.items():
            out[k] = _deep_merge(out.get(k), v)
        return out
    return copy.deepcopy(override if override is not None else base)




@dataclass(frozen=True)
class TrainingExecutionSchedule:
    """Canonical dataset-training execution schedule.

    Residency/movement choices are intentionally not encoded here.  This object
    only defines how dataset records are grouped into optimizer updates and
    epochs so the HLS testbench, software reference, reports, and runtime can
    share one interpretation.
    """

    batch_size: int
    epochs: int
    batch_mode: str
    shuffle: bool
    seed: int
    drop_last: bool
    sample_count: Optional[int]
    batches_per_epoch: Optional[int]
    samples_per_epoch: Optional[int]
    total_forward_backward_calls: Optional[int]
    total_optimizer_updates: Optional[int]
    explicit_train_steps: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "batch_mode": self.batch_mode,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "sample_count": self.sample_count,
            "batches_per_epoch": self.batches_per_epoch,
            "samples_per_epoch": self.samples_per_epoch,
            "total_forward_backward_calls": self.total_forward_backward_calls,
            "total_optimizer_updates": self.total_optimizer_updates,
            "explicit_train_steps": self.explicit_train_steps,
            "max_updates": self.explicit_train_steps,
            "sampling_policy": "deterministic_shuffle" if self.shuffle else "sequential",
            "partial_batch_policy": "drop" if self.drop_last else "include",
            "workload_resolution": "dataset" if self.sample_count is not None else "kernel_invocation",
            "kernel_calls_per_optimizer_step": 1,
            "forward_backward_calls_per_kernel_call": self.batch_size if self.batch_mode != "direct" else 1,
            "optimizer_updates_per_kernel_call": 1,
        }


def resolve_training_execution_schedule(
    raw_cfg: Dict[str, Any],
    *,
    sample_count: Optional[int] = None,
) -> TrainingExecutionSchedule:
    """Resolve canonical epoch/batch semantics with legacy compatibility.

    Canonical public keys are ``training.batch.*``.  Legacy
    ``training.execution.train_steps`` remains supported as an explicit optimizer
    update count, but it is never interpreted as an epoch count.
    """

    batch_size = max(1, int(_cfg_get(
        raw_cfg,
        "training.batch.size",
        _cfg_get(raw_cfg, "training.execution.batch_size", _cfg_get(raw_cfg, "training.batch_size", 1)),
    )))
    epochs = max(1, int(_cfg_get(
        raw_cfg,
        "training.batch.epochs",
        _cfg_get(raw_cfg, "training.execution.epochs", 1),
    )))
    batch_mode = str(_cfg_get(
        raw_cfg,
        "training.batch.mode",
        _cfg_get(
            raw_cfg,
            "training.accumulation.mode",
            _cfg_get(raw_cfg, "training.execution.batch_mode", "replay"),
        ),
    )).strip().lower().replace("-", "_")
    shuffle = bool(_cfg_get(
        raw_cfg,
        "training.batch.shuffle",
        _cfg_get(raw_cfg, "training.execution.shuffle", False),
    ))
    seed = int(_cfg_get(
        raw_cfg,
        "training.batch.seed",
        _cfg_get(raw_cfg, "training.execution.seed", 0),
    ))
    drop_last = bool(_cfg_get(
        raw_cfg,
        "training.batch.drop_last",
        _cfg_get(raw_cfg, "training.execution.drop_last", False),
    ))
    explicit = _cfg_get(
        raw_cfg,
        "training.batch.max_updates",
        _cfg_get(raw_cfg, "training.execution.train_steps", None),
    )
    explicit_train_steps = None if explicit is None else max(1, int(explicit))

    normalized_count = None if sample_count is None else max(0, int(sample_count))
    batches_per_epoch: Optional[int] = None
    samples_per_epoch: Optional[int] = None
    total_calls: Optional[int] = None
    total_updates: Optional[int] = None
    if normalized_count is not None:
        if normalized_count <= 0:
            raise ValueError("Dataset training requires sample_count > 0.")
        if drop_last:
            batches_per_epoch = normalized_count // batch_size
            if batches_per_epoch <= 0:
                raise ValueError(
                    "training.batch.drop_last=true would produce zero batches: "
                    f"sample_count={normalized_count}, batch_size={batch_size}."
                )
            samples_per_epoch = batches_per_epoch * batch_size
        else:
            batches_per_epoch = (normalized_count + batch_size - 1) // batch_size
            samples_per_epoch = normalized_count

        natural_updates = epochs * batches_per_epoch
        if explicit_train_steps is not None:
            total_updates = explicit_train_steps
            # Explicit train_steps is a legacy update-count override.  It is
            # intentionally not multiplied by epochs.
            full_epochs, remainder = divmod(total_updates, batches_per_epoch)
            total_calls = full_epochs * samples_per_epoch
            if remainder:
                if drop_last:
                    total_calls += remainder * batch_size
                else:
                    for batch_index in range(remainder):
                        start = batch_index * batch_size
                        total_calls += min(batch_size, normalized_count - start)
        else:
            total_updates = natural_updates
            total_calls = epochs * samples_per_epoch

    return TrainingExecutionSchedule(
        batch_size=batch_size,
        epochs=epochs,
        batch_mode=batch_mode,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        sample_count=normalized_count,
        batches_per_epoch=batches_per_epoch,
        samples_per_epoch=samples_per_epoch,
        total_forward_backward_calls=total_calls,
        total_optimizer_updates=total_updates,
        explicit_train_steps=explicit_train_steps,
    )


def training_record_order(
    sample_count: int,
    *,
    epoch_index: int,
    shuffle: bool,
    seed: int,
) -> List[int]:
    """Return the deterministic record order shared with the CSim testbench.

    The small LCG/Fisher-Yates implementation is deliberate: it is easy to
    reproduce in generated C++ without depending on implementation-specific
    standard-library random engines.
    """

    count = max(0, int(sample_count))
    order = list(range(count))
    if not shuffle or count <= 1:
        return order
    state = (int(seed) ^ (((int(epoch_index) + 1) * 0x9E3779B9) & 0xFFFFFFFF)) & 0xFFFFFFFF
    for index in range(count - 1, 0, -1):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        swap_index = state % (index + 1)
        order[index], order[swap_index] = order[swap_index], order[index]
    return order


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


# Semantic/reference training coverage is tracked separately from on-device
# HLS/VHDL coverage. This prevents a correct Python backward from being
# misreported as hardware-training support.
OP_TRAINING_REFERENCE_CAPS: Dict[str, TrainingOpCaps] = {
    "Dense": TrainingOpCaps(True, True, True, True),
    "Linear": TrainingOpCaps(True, True, True, True),
    "Conv": TrainingOpCaps(True, True, True, True),
    "Conv2D": TrainingOpCaps(True, True, True, True),
    "DepthwiseConv2D": TrainingOpCaps(True, True, True, True),
    "PointwiseConv2D": TrainingOpCaps(True, True, True, True),
    "BatchNormalization": TrainingOpCaps(True, True, True, True),
    "BatchNorm": TrainingOpCaps(True, True, True, True),
    "Relu": TrainingOpCaps(True, True, False, False),
    "LeakyRelu": TrainingOpCaps(True, True, False, False),
    "Sigmoid": TrainingOpCaps(True, True, False, False),
    "SiLU": TrainingOpCaps(True, True, False, False),
    "Softmax": TrainingOpCaps(True, True, False, False),
    "ReduceSum": TrainingOpCaps(True, True, False, False),
    "Add": TrainingOpCaps(True, True, False, False),
    "Mul": TrainingOpCaps(True, True, False, False),
    "MatMul": TrainingOpCaps(True, True, True, True),
    "Transpose": TrainingOpCaps(True, True, False, False),
    "LayerNormalization": TrainingOpCaps(True, True, True, True),
    "RMSNorm": TrainingOpCaps(True, True, True, True),
    "CausalMask": TrainingOpCaps(True, True, False, False),
    "RotaryEmbedding": TrainingOpCaps(True, True, False, False),
    "MultiHeadAttention": TrainingOpCaps(True, True, False, False),
    "MaxPool": TrainingOpCaps(True, True, False, False),
    "AvgPool": TrainingOpCaps(True, True, False, False),
    "AveragePool": TrainingOpCaps(True, True, False, False),
    "GlobalAveragePool": TrainingOpCaps(True, True, False, False),
    "Flatten": TrainingOpCaps(True, True, False, False),
    "Reshape": TrainingOpCaps(True, True, False, False),
    "Concat": TrainingOpCaps(True, True, False, False),
    "Slice": TrainingOpCaps(True, True, False, False),
    "Resize": TrainingOpCaps(True, True, False, False),
    "Gather": TrainingOpCaps(True, True, True, True),
    "Identity": TrainingOpCaps(True, True, False, False),
    "Cast": TrainingOpCaps(True, True, False, False),
    "Squeeze": TrainingOpCaps(True, True, False, False),
    "Unsqueeze": TrainingOpCaps(True, True, False, False),
}

OP_TRAINING_CAPS: Dict[str, TrainingOpCaps] = {
    "Dense": TrainingOpCaps(True, True, True, True),
    "Linear": TrainingOpCaps(True, True, True, True),
    "Conv": TrainingOpCaps(True, True, True, True),
    "Conv2D": TrainingOpCaps(True, True, True, True),
    "DepthwiseConv2D": TrainingOpCaps(True, True, True, True),
    "PointwiseConv2D": TrainingOpCaps(True, True, True, True),
    "BatchNormalization": TrainingOpCaps(True, True, True, True),
    "BatchNorm": TrainingOpCaps(True, True, True, True),
    "Relu": TrainingOpCaps(True, True, False, False),
    "LeakyRelu": TrainingOpCaps(True, True, False, False),
    "Sigmoid": TrainingOpCaps(True, True, False, False),
    "SiLU": TrainingOpCaps(True, True, False, False),
    "Softmax": TrainingOpCaps(True, True, False, False),
    "ReduceSum": TrainingOpCaps(True, True, False, False),
    "Add": TrainingOpCaps(True, True, False, False),
    "Mul": TrainingOpCaps(True, True, False, False),
    "MatMul": TrainingOpCaps(True, True, True, True),
    "Transpose": TrainingOpCaps(True, True, False, False),
    "LayerNormalization": TrainingOpCaps(True, True, True, True),
    "RMSNorm": TrainingOpCaps(True, True, True, True),
    "CausalMask": TrainingOpCaps(True, True, False, False),
    "RotaryEmbedding": TrainingOpCaps(True, True, False, False),
    "MultiHeadAttention": TrainingOpCaps(True, True, False, False),
    "MaxPool": TrainingOpCaps(True, True, False, False),
    "AvgPool": TrainingOpCaps(True, True, False, False),
    "AveragePool": TrainingOpCaps(True, True, False, False),
    "GlobalAveragePool": TrainingOpCaps(True, True, False, False),
    "Flatten": TrainingOpCaps(True, True, False, False),
    "Reshape": TrainingOpCaps(True, True, False, False),
    "Concat": TrainingOpCaps(True, True, False, False),
    "Slice": TrainingOpCaps(True, True, False, False),
    "Resize": TrainingOpCaps(True, True, False, False),
    "Gather": TrainingOpCaps(True, True, True, True),
    "Identity": TrainingOpCaps(True, True, False, False),
    "Cast": TrainingOpCaps(True, True, False, False),
    "Squeeze": TrainingOpCaps(True, True, False, False),
    "Unsqueeze": TrainingOpCaps(True, True, False, False),
}


@dataclass(frozen=True)
class TrainingPlan:
    optimizer_type: str
    learning_rate: float
    loss_type: str
    batch_size: int
    epochs: int
    execution_schedule: Dict[str, Any]
    gradient_mechanism: Dict[str, Any]
    implementation_stack: Dict[str, Any]

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
            "execution_schedule": self.execution_schedule,
            "gradient_mechanism": self.gradient_mechanism,
            "implementation_stack": self.implementation_stack,
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
            "data_movement.weights.load.interface",
            _cfg_get(
                raw_cfg,
                "data_movement.ps_pl.weights.mode",
                notes.get("global_weights_mode_requested", "embedded"),
            ),
        )
    ).lower().replace("-", "_")

    weight_pref = notes.get("weight_region_preference") or ["BRAM"]
    act_pref = notes.get("activation_region_preference") or ["BRAM"]

    weight_storage = str(_cfg_get(raw_cfg, "memory.weight_storage", _cfg_get(raw_cfg, "training.storage.weights", weight_pref[0]))).lower()
    activation_storage = str(_cfg_get(raw_cfg, "memory.activation_storage", _cfg_get(raw_cfg, "training.storage.activations", act_pref[0]))).lower()
    gradient_storage = str(_cfg_get(raw_cfg, "memory.gradient_storage", _cfg_get(raw_cfg, "training.storage.gradients", activation_storage))).lower()
    optimizer_state_storage = str(_cfg_get(raw_cfg, "memory.optimizer_state_storage", _cfg_get(raw_cfg, "training.storage.optimizer_state", weight_storage))).lower()

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



def _resolve_implementation_stack(raw_cfg: Dict[str, Any], gradient_mechanism: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize implementation provenance without requiring the future plugin runtime.

    The contract is intentionally stable across built-in and community implementations.
    It records selections now; later registry/ABI work will validate and instantiate them.
    """
    def selected(path: str, default: str) -> str:
        value = _cfg_get(raw_cfg, path, default)
        return str(value or default).strip()

    operator_impls = _cfg_get(raw_cfg, "implementations.operators", {}) or {}
    if not isinstance(operator_impls, dict):
        operator_impls = {}

    model_family = selected("implementations.model_family", "fpgai.imported_graph")
    model_impl = selected("implementations.model", "fpgai.graph_lowering")
    memory_policy = selected("implementations.memory_policy", "fpgai.memory.default")
    streaming_policy = selected("implementations.streaming_policy", "fpgai.streaming.default")
    transport_policy = selected("implementations.transport_policy", "fpgai.transport.default")
    numerical_policy = selected("implementations.numerical_policy", "fpgai.numeric.configured")
    backend = selected("implementations.backend", "fpgai.backend.hls_cpp")
    toolchain = selected("implementations.toolchain", "vitis_hls")
    board = selected("targets.platform.board", selected("targets.board", "unspecified"))

    return {
        "schema_version": 1,
        "model_family": model_family,
        "model_implementation": model_impl,
        "operator_implementations": {str(k): str(v) for k, v in sorted(operator_impls.items())},
        "memory_policy": memory_policy,
        "streaming_policy": streaming_policy,
        "transport_policy": transport_policy,
        "training_mechanism": str(gradient_mechanism.get("computation", "full_buffer")),
        "numerical_policy": numerical_policy,
        "backend": backend,
        "toolchain": toolchain,
        "board": board,
        "extension_resolution_status": "built_in_or_declared_selection",
        "registry_validation_status": "not_available_until_extension_abi_support",
    }


def _implementation_stack_fingerprint(stack: Dict[str, Any]) -> str:
    canonical = json.dumps(stack, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def build_training_plan(graph, raw_cfg: Dict[str, Any], compile_plan=None, memory_plan=None, communication_plan=None) -> TrainingPlan:
    optimizer_type = str(_cfg_get(raw_cfg, "training.optimizer.type", "sgd")).lower()
    learning_rate = float(_cfg_get(raw_cfg, "training.optimizer.learning_rate", 0.01))
    loss_type = str(_cfg_get(raw_cfg, "training.loss.type", "mse")).lower()
    execution_schedule = resolve_training_execution_schedule(raw_cfg)
    batch_size = execution_schedule.batch_size
    epochs = execution_schedule.epochs

    storage = _resolve_storage_policy(raw_cfg, compile_plan=compile_plan, memory_plan=memory_plan)
    movement_policy = _resolve_movement_policy(raw_cfg, communication_plan=communication_plan)
    planner_policy = _planner_policy_dict(compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan)

    cache_policy = _deep_merge(_default_training_cache_policy(), _cfg_get(raw_cfg, "training.cache", {}) or {})
    phase_overrides = _cfg_get(raw_cfg, "training.phase_overrides", {}) or {}
    numerics = _resolve_training_numerics(raw_cfg)
    estimator = _deep_merge(_default_estimator_cfg(), _cfg_get(raw_cfg, "training.estimator", {}) or {})

    gradient_computation = str(_cfg_get(raw_cfg, "training.gradients.computation", "full_buffer") or "full_buffer").strip().lower().replace("-", "_")
    gradient_materialization_default = "streamed" if gradient_computation == "fused_update" else ("tiled" if gradient_computation == "tiled_accumulate" else "full")
    gradient_materialization = str(_cfg_get(raw_cfg, "training.gradients.materialization", gradient_materialization_default) or gradient_materialization_default).strip().lower().replace("-", "_")
    gradient_export_policy = str(_cfg_get(raw_cfg, "training.gradients.export_policy", "recompute" if gradient_computation == "fused_update" else "materialized") or "disabled").strip().lower().replace("-", "_")
    parameter_gradient_storage = "none" if gradient_computation == "fused_update" else storage["gradient_storage"]
    gradient_mechanism = {
        "computation": gradient_computation,
        "materialization": gradient_materialization,
        "parameter_gradient_storage": parameter_gradient_storage,
        "configured_gradient_region": storage["gradient_storage"],
        "export_policy": gradient_export_policy,
        "complete_parameter_gradient_buffer": gradient_computation == "full_buffer",
        "gradient_tile_buffer": gradient_computation == "tiled_accumulate",
        "direct_optimizer_consumption": gradient_computation == "fused_update",
        "persistent_optimizer_state_required": optimizer_type in {"adam", "momentum"},
        "persistent_optimizer_state_storage": storage["optimizer_state_storage"] if optimizer_type in {"adam", "momentum"} else "none",
    }
    implementation_stack = _resolve_implementation_stack(raw_cfg, gradient_mechanism)

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
        execution_schedule=execution_schedule.to_dict(),
        gradient_mechanism=gradient_mechanism,
        implementation_stack=implementation_stack,
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



def _equivalence_workload_contract(plan: TrainingPlan) -> Dict[str, Any]:
    """Return the mechanism-independent contract used for fair comparisons."""
    contract = {
        "optimizer_type": plan.optimizer_type,
        "learning_rate": plan.learning_rate,
        "loss_type": plan.loss_type,
        "execution_schedule": plan.execution_schedule,
        "training_numerics": plan.numerics.get("training") or {},
        "parameter_trainable_ops": sorted(set(plan.parameter_trainable_ops)),
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return {
        "contract": contract,
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }

def _emit_gradient_mechanism_equivalence(training_dir: Path, plan: TrainingPlan) -> None:
    workload = _equivalence_workload_contract(plan)
    capture_requirements = default_training_capture_requirements(
        optimizer_type=plan.optimizer_type,
        export_gradients=str(plan.gradient_mechanism.get("export_policy", "none")) != "none",
    )
    capture_contract = NumericCaptureContract(
        workload_fingerprint_sha256=workload["fingerprint_sha256"],
        implementation_stack_fingerprint_sha256=_implementation_stack_fingerprint(plan.implementation_stack),
        producer_kind="hls_csim",
        producer_id=str(plan.implementation_stack.get("backend", "fpgai.backend.hls_cpp")),
        captures=capture_requirements,
        metadata={
            "training_mechanism": plan.gradient_mechanism.get("computation"),
            "optimizer_type": plan.optimizer_type,
            "loss_type": plan.loss_type,
            "validation_adapter_status": "built_in_schema_ready_capture_pending",
        },
    )
    capture_path = write_capture_contract(training_dir / "numeric_capture_contract.json", capture_contract)
    payload = {
        "artifact_kind": "fpgai_gradient_mechanism_equivalence",
        "schema_version": 2,
        "current_mechanism": plan.gradient_mechanism.get("computation"),
        "implementation_stack": plan.implementation_stack,
        "implementation_stack_fingerprint_sha256": _implementation_stack_fingerprint(plan.implementation_stack),
        "supported_comparison_mechanisms": ["full_buffer", "tiled_accumulate", "fused_update"],
        "workload_contract": workload["contract"],
        "workload_fingerprint_sha256": workload["fingerprint_sha256"],
        "contract_status": "resolved",
        "numeric_equivalence_status": "capture_pending",
        "numeric_reference": "python_training_reference",
        "numeric_capture_contract": str(capture_path),
        "numeric_capture_schema_version": 1,
        "required_comparisons": {
            "pre_update_loss_abs_error": None,
            "post_update_loss_abs_error": None,
            "weights_max_abs_error": None,
            "biases_max_abs_error": None,
            "adam_m_max_abs_error": None,
            "adam_v_max_abs_error": None,
            "optimizer_step_match": None,
            "exported_gradients_max_abs_error": None,
        },
        "claim_status": "architectural_result_preliminary_until_numeric_equivalence_passes",
        "comparison_rule": "Only artifacts with identical workload_fingerprint_sha256 are directly comparable.",
    }
    write_text(training_dir / "gradient_mechanism_equivalence.json", json.dumps(payload, indent=2) + "\n")
    lines = [
        "======= FPGAI Gradient Mechanism Equivalence =======",
        f"current_mechanism          : {payload['current_mechanism']}",
        f"workload_fingerprint      : {payload['workload_fingerprint_sha256']}",
        f"implementation_fingerprint: {payload['implementation_stack_fingerprint_sha256']}",
        f"contract_status           : {payload['contract_status']}",
        f"numeric_equivalence_status: {payload['numeric_equivalence_status']}",
        f"claim_status              : {payload['claim_status']}",
        "comparison_rule          : identical workload fingerprints required",
        "====================================================",
    ]
    write_text(training_dir / "gradient_mechanism_equivalence.txt", "\n".join(lines) + "\n")

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
    lines.append("execution_schedule       :")
    for k, v in sorted(plan.execution_schedule.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("gradient_mechanism       :")
    for k, v in sorted(plan.gradient_mechanism.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("implementation_stack     :")
    for k, v in sorted(plan.implementation_stack.items()):
        lines.append(f"  - {k}: {v}")
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
    _emit_gradient_mechanism_equivalence(training_dir, plan)
    return json_path
