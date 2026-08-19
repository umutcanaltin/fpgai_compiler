from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Dict, Iterable

from fpgai.ir.graph import Graph
from fpgai.ir.ops import Op

CompositeExpander = Callable[[Graph, Op], list[Op]]


@dataclass(frozen=True)
class CompositeLayerSpec:
    op_type: str
    expander: CompositeExpander
    description: str = ""
    provider: str = "fpgai"
    version: str = "1"


_COMPOSITES: Dict[str, CompositeLayerSpec] = {}
_DISCOVERED = False


def register_composite_layer(spec: CompositeLayerSpec, *, replace: bool = False) -> None:
    key = str(spec.op_type)
    if key in _COMPOSITES and not replace:
        raise ValueError(f"COMPOSITE001: composite layer {key!r} already registered")
    _COMPOSITES[key] = spec


def _tensor_like(graph: Graph, name: str, source: str) -> None:
    if graph.get_tensor(name) is not None:
        return
    src = graph.get_tensor(source)
    if src is None:
        graph.add_tensor(name, ())
    else:
        graph.add_tensor(name, tuple(src.shape), src.dtype)



def _tensor_matmul_result(graph: Graph, name: str, left: str, right: str) -> None:
    if graph.get_tensor(name) is not None:
        return
    lhs = graph.get_tensor(left)
    rhs = graph.get_tensor(right)
    if lhs is not None and rhs is not None:
        lshape = tuple(int(x) for x in lhs.shape)
        rshape = tuple(int(x) for x in rhs.shape)
        if len(lshape) >= 1 and len(rshape) == 2 and lshape[-1] == rshape[0]:
            graph.add_tensor(name, lshape[:-1] + (rshape[1],), lhs.dtype)
            return
    graph.add_tensor(name, ())


def _expand_gated_mlp(graph: Graph, op: Op) -> list[Op]:
    if len(op.inputs) != 4 or len(op.outputs) != 1:
        raise ValueError("COMPOSITE201: GatedMLP expects inputs [x,w_gate,w_up,w_down] and one output")
    x, w_gate, w_up, w_down = op.inputs
    out = op.outputs[0]
    prefix = op.name or "gated_mlp"
    gate = f"{prefix}__gate"
    up = f"{prefix}__up"
    gate_act = f"{prefix}__gate_act"
    gated = f"{prefix}__gated"
    _tensor_matmul_result(graph, gate, x, w_gate)
    _tensor_matmul_result(graph, up, x, w_up)
    _tensor_like(graph, gate_act, gate)
    _tensor_like(graph, gated, gate)
    _tensor_matmul_result(graph, out, gated, w_down)
    common = {"expanded_from": "GatedMLP", "composite_name": prefix}
    return [
        Op(f"{prefix}__gate_projection", "MatMul", [x, w_gate], [gate], {**common, "projection_role": "ffn_gate"}),
        Op(f"{prefix}__up_projection", "MatMul", [x, w_up], [up], {**common, "projection_role": "ffn_up"}),
        Op(f"{prefix}__silu", "SiLU", [gate], [gate_act], dict(common)),
        Op(f"{prefix}__gate_mul", "Mul", [gate_act, up], [gated], dict(common)),
        Op(f"{prefix}__down_projection", "MatMul", [gated, w_down], [out], {**common, "projection_role": "ffn_down"}),
    ]


def _expand_transformer_block(graph: Graph, op: Op) -> list[Op]:
    if len(op.inputs) < 8 or len(op.outputs) != 1:
        raise ValueError(
            "COMPOSITE101: TransformerBlock expects at least inputs "
            "[x,wq,wk,wv,wo,norm_scale,rope_cos,rope_sin] and one output"
        )
    x, wq, wk, wv, wo, norm_scale, rope_cos, rope_sin = op.inputs[:8]
    out = op.outputs[0]
    heads = int(op.attrs.get("heads", op.attrs.get("num_heads", 1)))
    causal = bool(op.attrs.get("causal", True))
    epsilon = float(op.attrs.get("epsilon", 1e-5))
    position_offset = int(op.attrs.get("position_offset", 0))
    prefix = op.name or "transformer_block"
    common = {"expanded_from": "TransformerBlock", "composite_name": prefix}

    # Backward-compatible compact attention block: the first eight inputs retain
    # the previously validated expansion and remain useful as a composite layer.
    if len(op.inputs) < 12:
        names = {k: f"{prefix}__{k}" for k in ("q","k","v","qr","kr","attn","proj")}
        for key in ("q", "k", "v"):
            _tensor_like(graph, names[key], x)
        _tensor_like(graph, names["qr"], names["q"])
        _tensor_like(graph, names["kr"], names["k"])
        _tensor_like(graph, names["attn"], x)
        _tensor_like(graph, names["proj"], x)
        _tensor_like(graph, out, x)
        return [
            Op(f"{prefix}__q_projection", "MatMul", [x, wq], [names["q"]], {**common, "projection_role": "q"}),
            Op(f"{prefix}__k_projection", "MatMul", [x, wk], [names["k"]], {**common, "projection_role": "k"}),
            Op(f"{prefix}__v_projection", "MatMul", [x, wv], [names["v"]], {**common, "projection_role": "v"}),
            Op(f"{prefix}__q_rope", "RotaryEmbedding", [names["q"], rope_cos, rope_sin], [names["qr"]], {**common, "position_offset": position_offset}),
            Op(f"{prefix}__k_rope", "RotaryEmbedding", [names["k"], rope_cos, rope_sin], [names["kr"]], {**common, "position_offset": position_offset}),
            Op(
                f"{prefix}__attention", "MultiHeadAttention",
                [names["qr"], names["kr"], names["v"]], [names["attn"]],
                {**common, "num_heads": heads, "heads": heads, "causal": causal, "execution_mode": op.attrs.get("execution_mode", "auto")},
            ),
            Op(f"{prefix}__o_projection", "MatMul", [names["attn"], wo], [names["proj"]], {**common, "projection_role": "o"}),
            Op(f"{prefix}__rms_norm", "RMSNorm", [names["proj"], norm_scale], [out], {**common, "axis": -1, "epsilon": epsilon}),
        ]

    # Full layerwise Transformer block with pre-norm attention, residuals and a
    # gated/SwiGLU-style feed-forward path. It still expands entirely into
    # ordinary FPGAI operators before backend selection.
    w_gate, w_up, w_down, ffn_norm_scale = op.inputs[8:12]
    names = {k: f"{prefix}__{k}" for k in (
        "attn_norm","q","k","v","qr","kr","attn","proj","attn_residual",
        "ffn_norm","gate","up","gate_act","gated","down"
    )}
    _tensor_like(graph, names["attn_norm"], x)
    _tensor_matmul_result(graph, names["q"], names["attn_norm"], wq)
    _tensor_matmul_result(graph, names["k"], names["attn_norm"], wk)
    _tensor_matmul_result(graph, names["v"], names["attn_norm"], wv)
    _tensor_like(graph, names["qr"], names["q"])
    _tensor_like(graph, names["kr"], names["k"])
    _tensor_like(graph, names["attn"], names["q"])
    _tensor_matmul_result(graph, names["proj"], names["attn"], wo)
    _tensor_like(graph, names["attn_residual"], x)
    _tensor_like(graph, names["ffn_norm"], x)
    _tensor_matmul_result(graph, names["gate"], names["ffn_norm"], w_gate)
    _tensor_matmul_result(graph, names["up"], names["ffn_norm"], w_up)
    _tensor_like(graph, names["gate_act"], names["gate"])
    _tensor_like(graph, names["gated"], names["gate"])
    _tensor_matmul_result(graph, names["down"], names["gated"], w_down)
    _tensor_like(graph, out, x)
    return [
        Op(f"{prefix}__attn_norm", "RMSNorm", [x, norm_scale], [names["attn_norm"]], {**common, "axis": -1, "epsilon": epsilon}),
        Op(f"{prefix}__q_projection", "MatMul", [names["attn_norm"], wq], [names["q"]], {**common, "projection_role": "q"}),
        Op(f"{prefix}__k_projection", "MatMul", [names["attn_norm"], wk], [names["k"]], {**common, "projection_role": "k"}),
        Op(f"{prefix}__v_projection", "MatMul", [names["attn_norm"], wv], [names["v"]], {**common, "projection_role": "v"}),
        Op(f"{prefix}__q_rope", "RotaryEmbedding", [names["q"], rope_cos, rope_sin], [names["qr"]], {**common, "position_offset": position_offset}),
        Op(f"{prefix}__k_rope", "RotaryEmbedding", [names["k"], rope_cos, rope_sin], [names["kr"]], {**common, "position_offset": position_offset}),
        Op(f"{prefix}__attention", "MultiHeadAttention", [names["qr"], names["kr"], names["v"]], [names["attn"]], {**common, "num_heads": heads, "heads": heads, "causal": causal, "execution_mode": op.attrs.get("execution_mode", "auto")}),
        Op(f"{prefix}__o_projection", "MatMul", [names["attn"], wo], [names["proj"]], {**common, "projection_role": "o"}),
        Op(f"{prefix}__attn_residual", "Add", [x, names["proj"]], [names["attn_residual"]], dict(common)),
        Op(f"{prefix}__ffn_norm", "RMSNorm", [names["attn_residual"], ffn_norm_scale], [names["ffn_norm"]], {**common, "axis": -1, "epsilon": epsilon}),
        Op(f"{prefix}__gate_projection", "MatMul", [names["ffn_norm"], w_gate], [names["gate"]], {**common, "projection_role": "ffn_gate"}),
        Op(f"{prefix}__up_projection", "MatMul", [names["ffn_norm"], w_up], [names["up"]], {**common, "projection_role": "ffn_up"}),
        Op(f"{prefix}__silu", "SiLU", [names["gate"]], [names["gate_act"]], dict(common)),
        Op(f"{prefix}__gate_mul", "Mul", [names["gate_act"], names["up"]], [names["gated"]], dict(common)),
        Op(f"{prefix}__down_projection", "MatMul", [names["gated"], w_down], [names["down"]], {**common, "projection_role": "ffn_down"}),
        Op(f"{prefix}__ffn_residual", "Add", [names["attn_residual"], names["down"]], [out], dict(common)),
    ]


def _register_builtins() -> None:
    if "GatedMLP" not in _COMPOSITES:
        register_composite_layer(CompositeLayerSpec(
            "GatedMLP",
            _expand_gated_mlp,
            "Composite gated/SwiGLU MLP expanded to MatMul, SiLU, Mul and MatMul before backend lowering.",
        ))
    if "TransformerBlock" not in _COMPOSITES:
        register_composite_layer(CompositeLayerSpec(
            "TransformerBlock",
            _expand_transformer_block,
            "Composite Transformer block expanded to ordinary FPGAI IR operators before backend lowering.",
        ))


def discover_composite_layers() -> None:
    global _DISCOVERED
    _register_builtins()
    if _DISCOVERED:
        return
    _DISCOVERED = True
    try:
        eps = metadata.entry_points()
        selected = eps.select(group="fpgai.composite_layers") if hasattr(eps, "select") else eps.get("fpgai.composite_layers", ())
    except Exception:
        selected = ()
    for ep in selected:
        provider = ep.load()
        value = provider() if callable(provider) and not isinstance(provider, CompositeLayerSpec) else provider
        specs: Iterable[CompositeLayerSpec] = (value,) if isinstance(value, CompositeLayerSpec) else tuple(value)
        for spec in specs:
            register_composite_layer(spec)


def composite_layer_registry() -> Dict[str, CompositeLayerSpec]:
    discover_composite_layers()
    return dict(_COMPOSITES)


def expand_composite_layers(graph: Graph) -> Graph:
    discover_composite_layers()
    rewritten: list[Op] = []
    count = 0
    for op in graph.ops:
        spec = _COMPOSITES.get(op.op_type)
        if spec is None:
            rewritten.append(op)
            continue
        expanded = list(spec.expander(graph, op))
        provider_info = {
            "provider": str(spec.provider),
            "version": str(spec.version),
            "composite_op_type": str(spec.op_type),
        }
        for expanded_op in expanded:
            expanded_op.attrs.setdefault("_fpgai_composite_provider", provider_info)
        rewritten.extend(expanded)
        count += 1
    graph.ops = rewritten
    if count:
        graph.metadata.setdefault("composite_expansion", {})
        graph.metadata["composite_expansion"].update({
            "schema": "fpgai.composite-expansion/v1",
            "expanded_count": count,
        })
    return graph
