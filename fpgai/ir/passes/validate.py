from __future__ import annotations
from typing import List, Set
from fpgai.ir.graph import Graph

def validate_allowlist(g: Graph, allowlist: List[str]) -> None:
    allowed: Set[str] = set(allowlist)
    bad = []
    for op in g.ops:
        if op.op_type not in allowed:
            bad.append(f"{op.op_type} ({op.name})")
    if bad:
        raise ValueError("Graph contains ops not in allowlist:\n  - " + "\n  - ".join(bad))
