from __future__ import annotations

from typing import List
from copy import deepcopy

from .graph import Graph
from .ops import Op


def validate_allowlist(g: Graph, allowed: List[str]) -> None:
    bad = [op.op_type for op in g.ops if op.op_type not in allowed]
    if bad:
        raise ValueError(f"Disallowed ops found: {sorted(set(bad))}. Allowed: {allowed}")


def assign_stable_names(g: Graph) -> Graph:
    g2 = deepcopy(g)
    counters = {}
    for op in g2.ops:
        k = op.op_type.lower()
        i = counters.get(k, 0)
        counters[k] = i + 1
        op.attrs["src_name"] = op.name
        op.name = f"{k}{i}" if k != "dense" else f"dense{i}"
    return g2


def insert_activations(g: Graph, *, kind: str, alpha: float = 0.1, except_last: bool = True) -> Graph:
    # Simple v1: insert after each Dense except last Dense
    if kind.lower() not in ("leakyrelu", "relu", "none"):
        raise ValueError(f"Unknown activation kind: {kind}")

    if kind.lower() == "none":
        return g

    g2 = deepcopy(g)
    dense_idxs = [i for i, op in enumerate(g2.ops) if op.op_type == "Dense"]
    if not dense_idxs:
        return g2

    last_dense_idx = dense_idxs[-1] if except_last else None

    new_ops = []
    for i, op in enumerate(g2.ops):
        new_ops.append(op)
        if op.op_type != "Dense":
            continue
        if except_last and i == last_dense_idx:
            continue

        out = op.outputs[0]
        act_out = f"{out}_{kind.lower()}"

        if kind.lower() == "leakyrelu":
            aop = Op(
                name=f"{op.name}_leakyrelu",
                op_type="LeakyRelu",
                inputs=[out],
                outputs=[act_out],
                attrs={"alpha": float(alpha)},
            )
        else:
            aop = Op(
                name=f"{op.name}_relu",
                op_type="Relu",
                inputs=[out],
                outputs=[act_out],
                attrs={},
            )

        # redirect following ops that consume this tensor (simple, later you’ll do SSA properly)
        for nxt in g2.ops[i + 1 :]:
            nxt.inputs = [act_out if x == out else x for x in nxt.inputs]

        new_ops.append(aop)

    g2.ops = new_ops
    return g2
