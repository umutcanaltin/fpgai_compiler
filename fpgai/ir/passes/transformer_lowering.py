from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from fpgai.ir import Graph
from fpgai.ir.contracts import ImplementationCandidate
from fpgai.ir.ops import Op as CanonicalOp


@dataclass(frozen=True)
class TransformerExecutionPlan:
    schema: str
    model_dimension: int
    num_heads: int
    num_kv_heads: int
    head_dimension: int
    sequence_length: int | None
    execution_mode: str
    projection_ops: tuple[str, ...]
    rotary_ops: tuple[str, ...]
    attention_ops: tuple[str, ...]
    kv_cache_tensors: tuple[str, ...]
    weight_storage: str
    kv_cache_storage: str
    kv_cache_capacity: int
    reuse_group: str
    phase_order: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "model_dimension": self.model_dimension,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dimension": self.head_dimension,
            "sequence_length": self.sequence_length,
            "execution_mode": self.execution_mode,
            "projection_ops": list(self.projection_ops),
            "rotary_ops": list(self.rotary_ops),
            "attention_ops": list(self.attention_ops),
            "kv_cache_tensors": list(self.kv_cache_tensors),
            "weight_storage": self.weight_storage,
            "kv_cache_storage": self.kv_cache_storage,
            "kv_cache_capacity": self.kv_cache_capacity,
            "reuse_group": self.reuse_group,
            "phase_order": list(self.phase_order),
        }


def _shape(graph: Graph, name: str) -> tuple[int, ...] | None:
    spec = graph.get_tensor(name)
    if spec is None:
        return None
    try:
        shape = tuple(int(x) for x in spec.shape)
    except Exception:
        return None
    return shape if shape and all(x > 0 for x in shape) else None


def configure_kv_cache_state(
    graph: Graph,
    *,
    key_cache: str,
    value_cache: str,
    capacity: int,
    sequence_axis: int = -2,
    storage: str = "auto",
    overflow_policy: str = "saturate",
    owner: str | None = None,
    state_group: str | None = None,
) -> None:
    """Attach explicit persistent mutable state contracts to KV-cache tensors."""
    overflow = str(overflow_policy).strip().lower().replace("-", "_")
    if overflow != "saturate":
        raise ValueError("IRLLM007: current KV-cache HLS runtime supports overflow_policy=saturate only")
    pair_group = str(state_group or f"kv:{key_cache}:{value_cache}")
    for name, kind in ((key_cache, "kv_key_cache"), (value_cache, "kv_value_cache")):
        tensor = graph.get_tensor(name)
        if tensor is None:
            raise KeyError(f"IRLLM001: unknown KV-cache tensor {name!r}")
        tensor.semantics.state.kind = kind
        tensor.semantics.state.mutable = True
        tensor.semantics.state.persistent_across_invocations = True
        tensor.semantics.state.update_policy = "append"
        tensor.semantics.state.sequence_axis = int(sequence_axis)
        tensor.semantics.state.capacity = int(capacity)
        tensor.semantics.state.overflow_policy = overflow
        tensor.semantics.state.owner = str(owner) if owner is not None else None
        tensor.semantics.state.state_group = pair_group
        storage_value = str(storage).strip().lower().replace("-", "_")
        if storage_value not in {"", "auto", "unspecified", "default"}:
            tensor.semantics.memory.storage = storage_value
            tensor.semantics.memory.residency = "external" if storage_value in {"ddr", "host", "external"} else "on_chip"
        tensor.semantics.memory.lifetime = "runtime_session"
        tensor.semantics.tags = tuple(tensor.semantics.tags) + (kind, "transformer_state")


def internalize_explicit_group_query_attention_state(
    graph: Graph,
    *,
    max_sequence_length: int,
    cache_storage: str = "auto",
    overflow_policy: str = "saturate",
) -> Dict[str, Any]:
    """Lower explicit-cache ONNX GroupQueryAttention into FPGAI persistent state.

    This pass is source/operator driven. It recognizes the normalized
    ``GroupQueryAttention`` contract used by ONNX Runtime contrib exports and
    replaces external past/present cache ports with the same persistent-state,
    RoPE, cache-update/read and cached-MHA primitives used by native FPGAI
    decoder graphs. No model-family identifier participates in the lowering.
    """
    if int(max_sequence_length) <= 0:
        raise ValueError("IRLLM020: GroupQueryAttention state capacity must be positive")

    rewritten: list[Any] = []
    removed_inputs: set[str] = set()
    removed_outputs: set[str] = set()
    layers: list[Dict[str, Any]] = []

    for op_index, op in enumerate(list(graph.ops)):
        if op.op_type != "GroupQueryAttention":
            rewritten.append(op)
            continue
        if len(op.inputs) < 7 or len(op.outputs) < 1:
            raise ValueError(
                f"IRLLM021: GroupQueryAttention {op.name!r} requires query/key/value, "
                "past K/V, sequence-length and total-length inputs"
            )

        query, key, value, past_key, past_value = op.inputs[:5]
        q_shape = _shape(graph, query)
        k_shape = _shape(graph, key)
        v_shape = _shape(graph, value)
        if not q_shape or not k_shape or not v_shape:
            raise ValueError(f"IRLLM022: GroupQueryAttention {op.name!r} requires static Q/K/V shapes")
        q_model = int(q_shape[-1])
        kv_model = int(k_shape[-1])
        if int(v_shape[-1]) != kv_model:
            raise ValueError(f"IRLLM023: GroupQueryAttention {op.name!r} K/V widths must match")

        num_heads = int(op.attrs.get("num_heads", 0) or 0)
        num_kv_heads = int(op.attrs.get("num_kv_heads", 0) or 0)
        if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(f"IRLLM024: GroupQueryAttention {op.name!r} requires valid query/KV head counts")
        if q_model % num_heads != 0 or kv_model % num_kv_heads != 0:
            raise ValueError(f"IRLLM025: GroupQueryAttention {op.name!r} model widths must divide by head counts")
        head_dim = q_model // num_heads
        if kv_model // num_kv_heads != head_dim:
            raise ValueError(f"IRLLM026: GroupQueryAttention {op.name!r} Q and KV head dimensions must match")

        prefix = str(op.name or f"gqa_{op_index}")
        k_state = f"{prefix}__key_cache"
        v_state = f"{prefix}__value_cache"
        position = f"{prefix}__position"
        k_update = f"{prefix}__key_update"
        v_update = f"{prefix}__value_update"
        k_read = f"{prefix}__key_read"
        v_read = f"{prefix}__value_read"
        valid_length = f"{prefix}__valid_length"

        dtype = graph.get_tensor(key).dtype if graph.get_tensor(key) is not None else "float32"
        for name, shape, dt in (
            (k_state, (1, int(max_sequence_length), kv_model), dtype),
            (v_state, (1, int(max_sequence_length), kv_model), graph.get_tensor(value).dtype),
            (position, (1,), "int32"),
            (valid_length, (1,), "int32"),
            (k_update, (1, int(max_sequence_length), kv_model), dtype),
            (v_update, (1, int(max_sequence_length), kv_model), graph.get_tensor(value).dtype),
            (k_read, (1, int(max_sequence_length), kv_model), dtype),
            (v_read, (1, int(max_sequence_length), kv_model), graph.get_tensor(value).dtype),
        ):
            if graph.get_tensor(name) is None:
                graph.add_tensor(name, shape, dt)

        group = f"kv.{prefix}"
        configure_kv_cache_state(
            graph, key_cache=k_state, value_cache=v_state,
            capacity=int(max_sequence_length), sequence_axis=1,
            storage=cache_storage, overflow_policy=overflow_policy,
            owner=prefix, state_group=group,
        )

        rewritten.append(CanonicalOp(
            name=f"{prefix}__state_length", op_type="PersistentStateLength",
            inputs=[k_state], outputs=[position], attrs={},
        ))

        q_for_attn, k_for_cache = query, key
        if bool(op.attrs.get("do_rotary", False)):
            if len(op.inputs) < 9:
                raise ValueError(f"IRLLM027: GroupQueryAttention {op.name!r} fused RoPE requires cosine and sine cache inputs")
            cos_name, sin_name = op.inputs[7], op.inputs[8]
            q_rope = f"{prefix}__query_rope"
            k_rope = f"{prefix}__key_rope"
            if graph.get_tensor(q_rope) is None:
                graph.add_tensor(q_rope, q_shape, graph.get_tensor(query).dtype)
            if graph.get_tensor(k_rope) is None:
                graph.add_tensor(k_rope, k_shape, graph.get_tensor(key).dtype)
            common = {
                "rotary_dim": int(op.attrs.get("rotary_dim", head_dim) or head_dim),
                "interleaved": bool(op.attrs.get("rotary_interleaved", False)),
            }
            rewritten.append(CanonicalOp(
                name=f"{prefix}__query_rope_op", op_type="RotaryEmbedding",
                inputs=[query, cos_name, sin_name, position], outputs=[q_rope],
                attrs={**common, "num_heads": num_heads},
            ))
            rewritten.append(CanonicalOp(
                name=f"{prefix}__key_rope_op", op_type="RotaryEmbedding",
                inputs=[key, cos_name, sin_name, position], outputs=[k_rope],
                attrs={**common, "num_heads": num_kv_heads},
            ))
            q_for_attn, k_for_cache = q_rope, k_rope

        rewritten.extend([
            CanonicalOp(
                name=f"{prefix}__append_key", op_type="KVCacheUpdate",
                inputs=[k_state, k_for_cache], outputs=[k_update],
                attrs={"sequence_axis": 1, "capacity": int(max_sequence_length), "update_policy": "append"},
            ),
            CanonicalOp(
                name=f"{prefix}__append_value", op_type="KVCacheUpdate",
                inputs=[v_state, value], outputs=[v_update],
                attrs={"sequence_axis": 1, "capacity": int(max_sequence_length), "update_policy": "append"},
            ),
            CanonicalOp(name=f"{prefix}__valid_length_op", op_type="PersistentStateLength", inputs=[k_state], outputs=[valid_length], attrs={}),
            CanonicalOp(name=f"{prefix}__read_key", op_type="PersistentStateRead", inputs=[k_state], outputs=[k_read], attrs={}),
            CanonicalOp(name=f"{prefix}__read_value", op_type="PersistentStateRead", inputs=[v_state], outputs=[v_read], attrs={}),
            CanonicalOp(
                name=f"{prefix}__cached_attention", op_type="MultiHeadAttention",
                inputs=[q_for_attn, k_read, v_read, valid_length], outputs=[op.outputs[0]],
                attrs={
                    "num_heads": num_heads, "num_kv_heads": num_kv_heads,
                    "causal": bool(op.attrs.get("causal", True)),
                    "scale": float(op.attrs.get("scale", 0.0) or 0.0),
                    "execution_mode": str(op.attrs.get("execution_mode", "serialized")),
                },
            ),
        ])

        removed_inputs.update({past_key, past_value, op.inputs[5], op.inputs[6]})
        if len(op.outputs) > 1:
            removed_outputs.update(op.outputs[1:])
        layers.append({
            "index": len(layers), "owner": prefix, "state_group": group,
            "key": k_state, "value": v_state, "storage": str(cache_storage),
            "overflow_policy": str(overflow_policy),
        })

    if not layers:
        return {"schema": "fpgai.explicit-gqa-state-internalization/v1", "layer_count": 0, "layers": []}

    graph.ops = rewritten
    consumers = {name for item in graph.ops for name in item.inputs}
    graph.inputs = [name for name in graph.inputs if name not in removed_inputs or name in consumers]
    graph.outputs = [name for name in graph.outputs if name not in removed_outputs]
    graph.semantics.runtime_contract.update({
        "persistent_kv_cache_backend_required": True,
        "max_sequence_length": int(max_sequence_length),
        "kv_cache_storage": str(cache_storage),
        "explicit_cache_ports_internalized": True,
    })
    plan = {
        "schema": "fpgai.explicit-gqa-state-internalization/v1",
        "layer_count": len(layers), "max_sequence_length": int(max_sequence_length),
        "cache_storage": str(cache_storage), "layers": layers,
    }
    graph.metadata["explicit_gqa_state_internalization"] = plan
    return plan



@dataclass(frozen=True)
class TokenDecodingPlan:
    schema: str
    max_sequence_length: int
    token_sequence_length: int
    position_offset: int
    key_cache: str
    value_cache: str
    cache_storage: str
    cache_update_policy: str
    cache_overflow_policy: str
    runtime_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "max_sequence_length": self.max_sequence_length,
            "token_sequence_length": self.token_sequence_length,
            "position_offset": self.position_offset,
            "key_cache": self.key_cache,
            "value_cache": self.value_cache,
            "cache_storage": self.cache_storage,
            "cache_update_policy": self.cache_update_policy,
            "cache_overflow_policy": self.cache_overflow_policy,
            "runtime_mode": self.runtime_mode,
        }


def plan_token_decoding(
    graph: Graph,
    *,
    key_cache: str,
    value_cache: str,
    max_sequence_length: int,
    token_sequence_length: int = 1,
    position_offset: int = 0,
    cache_storage: str = "auto",
    overflow_policy: str = "saturate",
) -> TokenDecodingPlan:
    """Attach explicit token-by-token decoding/runtime semantics.

    This is a compiler/runtime contract, not a claim that persistent external-memory
    cache mutation is already implemented by the stateless HLS top. The contract
    makes cache ownership, append semantics and position handling explicit so a
    runtime/backend implementation can negotiate support deterministically.
    """
    if max_sequence_length <= 0 or token_sequence_length <= 0:
        raise ValueError("IRLLM005: sequence lengths must be positive")
    if position_offset < 0 or position_offset + token_sequence_length > max_sequence_length:
        raise ValueError("IRLLM006: token decode position range exceeds max_sequence_length")
    configure_kv_cache_state(
        graph, key_cache=key_cache, value_cache=value_cache, capacity=max_sequence_length,
        storage=cache_storage, overflow_policy=overflow_policy
    )
    for op in graph.ops:
        if op.op_type == "RotaryEmbedding":
            op.attrs.setdefault("position_offset", int(position_offset))
            op.semantics.schedule["position_offset"] = int(position_offset)
            op.semantics.schedule["runtime_mode"] = "token_decode"
        if op.op_type == "MultiHeadAttention":
            op.semantics.schedule["runtime_mode"] = "token_decode"
            op.semantics.schedule["kv_cache_read"] = True

    graph.semantics.runtime_contract.update({
        "execution_mode": "token_decode",
        "token_sequence_length": int(token_sequence_length),
        "position_offset": int(position_offset),
        "max_sequence_length": int(max_sequence_length),
        "kv_cache": {
            "key": key_cache,
            "value": value_cache,
            "storage": cache_storage,
            "update_policy": "append",
            "overflow_policy": str(overflow_policy),
            "persistent_across_invocations": True,
        },
        "persistent_kv_cache_backend_required": True,
    })
    plan = TokenDecodingPlan(
        schema="fpgai.token-decoding-plan/v1",
        max_sequence_length=int(max_sequence_length),
        token_sequence_length=int(token_sequence_length),
        position_offset=int(position_offset),
        key_cache=str(key_cache),
        value_cache=str(value_cache),
        cache_storage=str(cache_storage),
        cache_update_policy="append",
        cache_overflow_policy=str(overflow_policy),
        runtime_mode="token_decode",
    )
    graph.metadata["token_decoding_plan"] = plan.to_dict()
    return plan


def plan_layered_token_decoding(
    graph: Graph,
    *,
    layer_caches: Sequence[Dict[str, str]],
    max_sequence_length: int,
    token_sequence_length: int = 1,
    cache_storage: str = "auto",
    overflow_policy: str = "saturate",
) -> Dict[str, Any]:
    """Configure persistent K/V ownership for a multi-layer decoder.

    Each layer owns an independent K/V pair while all pairs share one explicit
    runtime-session contract. This is model-agnostic: callers provide tensor names
    from the imported graph rather than a model family identifier.
    """
    if max_sequence_length <= 0 or token_sequence_length <= 0:
        raise ValueError("IRLLM008: layered decode sequence lengths must be positive")
    layers: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(layer_caches):
        key = str(item.get("key", ""))
        value = str(item.get("value", ""))
        owner = str(item.get("owner") or f"transformer.layer.{index}")
        if not key or not value:
            raise ValueError("IRLLM009: each layered decode entry requires key and value tensor names")
        if key in seen or value in seen:
            raise ValueError("IRLLM010: persistent cache tensors cannot be owned by multiple layers")
        seen.update({key, value})
        group = str(item.get("state_group") or f"kv.layer.{index}")
        configure_kv_cache_state(
            graph, key_cache=key, value_cache=value, capacity=max_sequence_length,
            storage=str(item.get("storage") or cache_storage),
            overflow_policy=str(item.get("overflow_policy") or overflow_policy),
            owner=owner, state_group=group,
        )
        layers.append({
            "index": int(index), "owner": owner, "state_group": group,
            "key": key, "value": value,
            "storage": str(item.get("storage") or cache_storage),
            "overflow_policy": str(item.get("overflow_policy") or overflow_policy),
        })
    contract = {
        "schema": "fpgai.layered-token-decoding-plan/v1",
        "runtime_mode": "token_decode",
        "token_sequence_length": int(token_sequence_length),
        "max_sequence_length": int(max_sequence_length),
        "layer_count": len(layers),
        "layers": layers,
        "cursor_policy": "independent_per_state_group",
    }
    graph.semantics.runtime_contract["layered_kv_cache"] = contract
    graph.metadata["layered_token_decoding_plan"] = contract
    return contract



@dataclass(frozen=True)
class AutoregressiveRuntimePlan:
    schema: str
    default_mode: str
    supported_modes: tuple[str, ...]
    max_sequence_length: int
    prefill_sequence_length: int
    decode_sequence_length: int
    layer_count: int
    cache_storage: str
    reset_state_on_prefill: bool
    position_source: str
    tied_parameter_groups: tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "default_mode": self.default_mode,
            "supported_modes": list(self.supported_modes),
            "max_sequence_length": self.max_sequence_length,
            "prefill_sequence_length": self.prefill_sequence_length,
            "decode_sequence_length": self.decode_sequence_length,
            "layer_count": self.layer_count,
            "cache_storage": self.cache_storage,
            "reset_state_on_prefill": self.reset_state_on_prefill,
            "position_source": self.position_source,
            "tied_parameter_groups": [dict(x) for x in self.tied_parameter_groups],
        }


def _validate_tied_parameter_groups(graph: Graph, groups: Sequence[Dict[str, Any]]) -> tuple[Dict[str, Any], ...]:
    resolved: list[Dict[str, Any]] = []
    for index, raw in enumerate(groups):
        name = str(raw.get("name") or f"tied_parameter_group_{index}")
        members = list(raw.get("members") or [])
        if len(members) < 2:
            raise ValueError("IRLLM011: tied parameter groups require at least two members")
        resolved_members: list[Dict[str, str]] = []
        canonical_name = str(members[0].get("tensor") or "")
        canonical_value = None
        canonical_nbytes = 0
        if canonical_name in graph.constants:
            import numpy as np
            canonical_value = np.asarray(graph.constants[canonical_name])
            canonical_nbytes = int(canonical_value.nbytes)
        duplicate_bytes = 0
        for item in members:
            tensor_name = str(item.get("tensor") or "")
            view = str(item.get("view") or "native").strip().lower()
            if graph.get_tensor(tensor_name) is None:
                raise KeyError(f"IRLLM012: unknown tied parameter tensor {tensor_name!r}")
            if tensor_name not in graph.constants:
                raise ValueError(f"IRLLM013: tied parameter tensor {tensor_name!r} must be a compile-time parameter/constant")
            if view not in {"native", "transpose"}:
                raise ValueError("IRLLM014: tied parameter view must be native or transpose")
            tensor = graph.get_tensor(tensor_name)
            if canonical_value is not None:
                import numpy as np
                member_value = np.asarray(graph.constants[tensor_name])
                expected_value = canonical_value if view == "native" else canonical_value.T
                if member_value.shape != expected_value.shape or not np.array_equal(member_value, expected_value):
                    raise ValueError(
                        f"IRLLM018: tied parameter {tensor_name!r} values do not match the declared {view} view of physical owner {canonical_name!r}"
                    )
                if tensor_name != canonical_name:
                    duplicate_bytes += int(member_value.nbytes)
            tensor.semantics.tags = tuple(tensor.semantics.tags) + (f"physical_owner:{canonical_name}", "tied_parameter", f"tied_group:{name}")
            resolved_members.append({"tensor": tensor_name, "view": view})
        resolved.append({
            "name": name,
            "physical_owner": canonical_name,
            "members": resolved_members,
            "physical_parameter_bytes": canonical_nbytes,
            "deduplicated_parameter_bytes": duplicate_bytes,
        })
    return tuple(resolved)


def plan_autoregressive_runtime(
    graph: Graph,
    *,
    layer_caches: Sequence[Dict[str, str]],
    max_sequence_length: int,
    prefill_sequence_length: int,
    decode_sequence_length: int = 1,
    cache_storage: str = "auto",
    overflow_policy: str = "saturate",
    reset_state_on_prefill: bool = True,
    tied_parameter_groups: Sequence[Dict[str, Any]] = (),
) -> AutoregressiveRuntimePlan:
    """Attach one explicit prefill/decode runtime-session contract.

    The contract is source/model agnostic: callers identify state tensors and optional
    tied parameters from the imported graph. It records lifecycle semantics only;
    board-specific state transfer remains a runtime-backend responsibility.
    """
    if max_sequence_length <= 0:
        raise ValueError("IRLLM015: max_sequence_length must be positive")
    if prefill_sequence_length <= 0 or prefill_sequence_length > max_sequence_length:
        raise ValueError("IRLLM016: prefill_sequence_length must be within cache capacity")
    if decode_sequence_length <= 0 or decode_sequence_length > max_sequence_length:
        raise ValueError("IRLLM017: decode_sequence_length must be within cache capacity")
    layered = plan_layered_token_decoding(
        graph,
        layer_caches=layer_caches,
        max_sequence_length=max_sequence_length,
        token_sequence_length=decode_sequence_length,
        cache_storage=cache_storage,
        overflow_policy=overflow_policy,
    )
    tied = _validate_tied_parameter_groups(graph, tied_parameter_groups)
    contract = {
        "schema": "fpgai.autoregressive-runtime-plan/v1",
        "default_mode": "prefill",
        "supported_modes": ["prefill", "decode"],
        "max_sequence_length": int(max_sequence_length),
        "prefill_sequence_length": int(prefill_sequence_length),
        "decode_sequence_length": int(decode_sequence_length),
        "layer_count": int(layered.get("layer_count", 0)),
        "cache_storage": str(cache_storage),
        "reset_state_on_prefill": bool(reset_state_on_prefill),
        "position_source": "persistent_state_cursor",
        "cache_cursor_policy": str(layered.get("cursor_policy", "independent_per_state_group")),
        "tied_parameter_groups": [dict(x) for x in tied],
    }
    graph.semantics.runtime_contract["autoregressive_session"] = contract
    graph.metadata["autoregressive_runtime_plan"] = contract
    for op in graph.ops:
        if op.op_type in {"RotaryEmbedding", "MultiHeadAttention", "KVCacheUpdate", "PersistentStateRead", "PersistentStateLength"}:
            op.semantics.schedule.setdefault("runtime_modes", ["prefill", "decode"])
    return AutoregressiveRuntimePlan(
        schema=contract["schema"],
        default_mode=contract["default_mode"],
        supported_modes=tuple(contract["supported_modes"]),
        max_sequence_length=int(max_sequence_length),
        prefill_sequence_length=int(prefill_sequence_length),
        decode_sequence_length=int(decode_sequence_length),
        layer_count=int(contract["layer_count"]),
        cache_storage=str(cache_storage),
        reset_state_on_prefill=bool(reset_state_on_prefill),
        position_source=str(contract["position_source"]),
        tied_parameter_groups=tied,
    )

def plan_transformer_execution(
    graph: Graph,
    *,
    model_dimension: int,
    num_heads: int,
    max_sequence_length: int,
    num_kv_heads: int | None = None,
    execution_mode: str = "auto",
    weight_storage: str = "auto",
    kv_cache_storage: str = "auto",
    score_buffer_storage: str = "auto",
    projection_roles: Sequence[str] = ("q", "k", "v", "o"),
) -> List[TransformerExecutionPlan]:
    """Attach LLM execution semantics without forcing a single backend implementation.

    The plan records legal implementation candidates and optional reuse groups. It does not force
    serialized execution or any BRAM/URAM/DDR placement: explicit user configuration wins when
    legal, while ``auto`` remains unresolved until implementation selection.
    """
    if model_dimension <= 0 or num_heads <= 0 or model_dimension % num_heads:
        raise ValueError("IRLLM002: model_dimension must be positive and divisible by num_heads")
    if max_sequence_length <= 0:
        raise ValueError("IRLLM003: max_sequence_length must be positive")
    kv_heads = int(num_heads if num_kv_heads is None else num_kv_heads)
    if kv_heads <= 0 or num_heads % kv_heads:
        raise ValueError("IRLLM005: num_heads must be divisible by num_kv_heads")
    if execution_mode not in {"auto", "unspecified", "serialized", "phase_shared", "parallel"}:
        raise ValueError("IRLLM004: execution_mode must be auto, serialized, phase_shared, or parallel")

    head_dimension = model_dimension // num_heads
    role_set = {str(x) for x in projection_roles}
    projection_ops = [
        op for op in graph.ops
        if str((op.attrs or {}).get("projection_role", "")) in role_set
        and op.op_type in {"Dense", "MatMul"}
    ]
    rotary_ops = [op for op in graph.ops if op.op_type == "RotaryEmbedding"]
    attention_ops = [op for op in graph.ops if op.op_type == "MultiHeadAttention"]
    cache_tensors = [
        name for name, tensor in graph.tensors.items()
        if tensor.semantics.state.kind in {"kv_key_cache", "kv_value_cache"}
    ]

    # infer sequence length from first attention/projection tensor when possible
    sequence_length = None
    candidates = attention_ops + projection_ops
    for op in candidates:
        if not op.inputs:
            continue
        shape = _shape(graph, str(op.inputs[0]))
        if shape and len(shape) >= 2:
            sequence_length = shape[-2]
            break

    reuse_group = "transformer_compute_engine_0"
    for op in projection_ops:
        op.semantics.schedule.update({
            "transformer_role": f"projection_{op.attrs.get('projection_role')}",
            "reuse_group": reuse_group,
        })
        if execution_mode not in {"auto", "unspecified"}:
            op.semantics.schedule["execution_mode"] = execution_mode
        if weight_storage not in {"auto", "unspecified"}:
            op.semantics.schedule["weight_storage"] = weight_storage
        op.semantics.implementation_candidates = tuple(op.semantics.implementation_candidates) + (
            ImplementationCandidate(backend="hls", implementation_id="fpgai.hls.linear.serialized", status="candidate"),
            ImplementationCandidate(backend="vhdl", implementation_id="fpgai.vhdl.linear.streaming", status="candidate"),
        )
        for inp in op.inputs[1:]:
            if inp in graph.constants and graph.get_tensor(str(inp)) is not None:
                if weight_storage not in {"auto", "unspecified"}:
                    graph.tensors[str(inp)].semantics.memory.storage = weight_storage
                    graph.tensors[str(inp)].semantics.memory.residency = "external" if weight_storage in {"ddr", "host", "external"} else "on_chip"
                graph.tensors[str(inp)].semantics.tags = tuple(graph.tensors[str(inp)].semantics.tags) + ("transformer_weight",)

    for op in rotary_ops:
        op.semantics.schedule.update({
            "transformer_role": "rotary_position_encoding",
            "head_dimension": head_dimension,
        })
        if execution_mode not in {"auto", "unspecified"}:
            op.semantics.schedule["execution_mode"] = execution_mode
        op.semantics.implementation_candidates = tuple(op.semantics.implementation_candidates) + (
            ImplementationCandidate(backend="hls", implementation_id="fpgai.hls.rope.pairwise", status="candidate"),
            ImplementationCandidate(backend="vhdl", implementation_id="fpgai.vhdl.rope.pairwise", status="candidate"),
        )

    for op in attention_ops:
        op.attrs.setdefault("num_heads", int(num_heads))
        op.attrs.setdefault("num_kv_heads", int(kv_heads))
        if execution_mode not in {"auto", "unspecified"}:
            op.attrs.setdefault("execution_mode", execution_mode)
        op.semantics.schedule.update({
            "transformer_role": "multi_head_attention",
            "reuse_group": reuse_group,
            "num_heads": int(num_heads),
            "num_kv_heads": int(kv_heads),
            "head_dimension": int(head_dimension),
            "max_sequence_length": int(max_sequence_length),
        })
        if execution_mode not in {"auto", "unspecified"}:
            op.semantics.schedule["execution_mode"] = execution_mode
            op.semantics.buffering["head_materialization"] = "serialized" if execution_mode != "parallel" else "parallel"
        if score_buffer_storage not in {"auto", "unspecified"}:
            op.semantics.buffering["score_buffer"] = score_buffer_storage
        op.semantics.implementation_candidates = tuple(op.semantics.implementation_candidates) + (
            ImplementationCandidate(
                backend="hls",
                implementation_id="fpgai.hls.mha.serialized",
                status="candidate",
                constraints={"static_shapes": True, "head_reuse": True},
            ),
            ImplementationCandidate(backend="vhdl", implementation_id="fpgai.vhdl.mha.streaming", status="candidate"),
        )

    graph.semantics.runtime_contract.update({
        "transformer_execution": execution_mode,
        "kv_cache_capacity": int(max_sequence_length),
        "persistent_state": bool(cache_tensors),
    })
    if kv_cache_storage not in {"auto", "unspecified"}:
        graph.semantics.runtime_contract["kv_cache_storage"] = kv_cache_storage
    graph.metadata["transformer_execution_policy"] = {
        "execution_mode": execution_mode,
        "reuse_group": reuse_group,
        "weight_storage": weight_storage,
        "kv_cache_storage": kv_cache_storage,
        "score_buffer_storage": score_buffer_storage,
        "selection_policy": "user_request_or_backend_default",
    }

    if not (projection_ops or rotary_ops or attention_ops or cache_tensors):
        return []

    plan = TransformerExecutionPlan(
        schema="fpgai.transformer-execution-plan/v1",
        model_dimension=int(model_dimension),
        num_heads=int(num_heads),
        num_kv_heads=int(kv_heads),
        head_dimension=int(head_dimension),
        sequence_length=sequence_length,
        execution_mode=execution_mode,
        projection_ops=tuple(op.name for op in projection_ops),
        rotary_ops=tuple(op.name for op in rotary_ops),
        attention_ops=tuple(op.name for op in attention_ops),
        kv_cache_tensors=tuple(cache_tensors),
        weight_storage=str(weight_storage),
        kv_cache_storage=str(kv_cache_storage),
        kv_cache_capacity=int(max_sequence_length),
        reuse_group=reuse_group,
        phase_order=("q_projection", "k_projection", "v_projection", "rope", "attention_heads", "o_projection"),
    )
    graph.metadata["transformer_execution_plans"] = [plan.to_dict()]
    return [plan]
