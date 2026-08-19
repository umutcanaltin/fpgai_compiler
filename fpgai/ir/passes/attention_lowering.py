from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from fpgai.ir import Graph
from fpgai.ir.contracts import ImplementationCandidate


@dataclass(frozen=True)
class AttentionLoweringPlan:
    score_matmul: str
    softmax: str
    value_matmul: str
    sequence_length: int | None
    head_dimension: int | None
    tile_m: int
    tile_n: int
    tile_k: int
    accumulator_dtype: str
    score_buffer_storage: str
    implementation_policy: str
    causal: bool = False
    mask_op: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "fpgai.attention-lowering-plan/v1",
            "score_matmul": self.score_matmul,
            "softmax": self.softmax,
            "value_matmul": self.value_matmul,
            "sequence_length": self.sequence_length,
            "head_dimension": self.head_dimension,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "accumulator_dtype": self.accumulator_dtype,
            "score_buffer_storage": self.score_buffer_storage,
            "implementation_policy": self.implementation_policy,
            "causal": self.causal,
            "mask_op": self.mask_op,
        }


def _shape(graph: Graph, tensor: str) -> tuple[int, ...] | None:
    spec = graph.get_tensor(tensor)
    if spec is None:
        return None
    shape = tuple(int(x) for x in spec.shape)
    return shape if shape and all(x > 0 for x in shape) else None


def plan_attention_lowering(
    graph: Graph,
    *,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    accumulator_dtype: str = "float32",
    score_buffer_storage: str = "bram",
) -> List[AttentionLoweringPlan]:
    """Recognize MatMul -> [scale] -> Softmax -> MatMul attention cores and attach hardware contracts.

    This pass is intentionally backend-neutral. It turns logical attention structure into
    explicit FPGAI implementation candidates, tiling, accumulation and buffering contracts;
    HLS/VHDL implementations remain selectable later.
    """
    producers = {out: op for op in graph.ops for out in op.outputs}
    consumers: Dict[str, list] = {}
    for op in graph.ops:
        for inp in op.inputs:
            consumers.setdefault(inp, []).append(op)

    plans: List[AttentionLoweringPlan] = []
    for softmax in [op for op in graph.ops if op.op_type == "Softmax"]:
        if not softmax.inputs or not softmax.outputs:
            continue
        score_tensor = softmax.inputs[0]
        score_src = producers.get(score_tensor)
        mask_op = score_src if score_src and score_src.op_type == "CausalMask" else None
        if mask_op and mask_op.inputs:
            score_src = producers.get(mask_op.inputs[0])
        if score_src and score_src.op_type == "Mul" and score_src.inputs:
            # Attention scaling may insert Mul between score MatMul and Softmax.
            matmul_input = next((x for x in score_src.inputs if producers.get(x) and producers[x].op_type == "MatMul"), None)
            score_mm = producers.get(matmul_input) if matmul_input else None
        else:
            score_mm = score_src if score_src and score_src.op_type == "MatMul" else None
        if score_mm is None:
            continue
        probability = softmax.outputs[0]
        value_mm = next((op for op in consumers.get(probability, []) if op.op_type == "MatMul"), None)
        if value_mm is None:
            continue
        q_shape = _shape(graph, score_mm.inputs[0]) if score_mm.inputs else None
        score_shape = _shape(graph, softmax.outputs[0])
        seq = score_shape[-1] if score_shape and len(score_shape) >= 2 else None
        head_dim = q_shape[-1] if q_shape else None
        plan = AttentionLoweringPlan(
            score_matmul=score_mm.name,
            softmax=softmax.name,
            value_matmul=value_mm.name,
            sequence_length=seq,
            head_dimension=head_dim,
            tile_m=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
            accumulator_dtype=str(accumulator_dtype),
            score_buffer_storage=str(score_buffer_storage),
            implementation_policy="implementation_selectable",
            causal=mask_op is not None,
            mask_op=mask_op.name if mask_op is not None else None,
        )
        plans.append(plan)
        for op, role in ((score_mm, "attention_score_matmul"), (softmax, "attention_softmax"), (value_mm, "attention_value_matmul")):
            op.semantics.schedule.update({
                "attention_role": role,
                "tile_m": int(tile_m), "tile_n": int(tile_n), "tile_k": int(tile_k),
                "accumulator_dtype": str(accumulator_dtype),
            })
            op.semantics.implementation_candidates = tuple(op.semantics.implementation_candidates) + (
                ImplementationCandidate(backend="hls", status="candidate"),
                ImplementationCandidate(backend="vhdl", status="candidate"),
            )
        if mask_op is not None:
            mask_op.semantics.schedule.update({"attention_role": "attention_causal_mask"})
            mask_op.semantics.implementation_candidates = tuple(mask_op.semantics.implementation_candidates) + (
                ImplementationCandidate(backend="hls", status="candidate"),
                ImplementationCandidate(backend="vhdl", status="candidate"),
            )
        graph.tensors[probability].semantics.memory.storage = str(score_buffer_storage)
        graph.tensors[probability].semantics.memory.residency = "on_chip"
        graph.tensors[probability].semantics.tags = tuple(graph.tensors[probability].semantics.tags) + ("attention_probability_buffer",)
        softmax.semantics.buffering.update({
            "score_probability_buffer": str(score_buffer_storage),
            "materialization": "explicit",
        })
    graph.metadata["attention_lowering_plans"] = [p.to_dict() for p in plans]
    return plans
