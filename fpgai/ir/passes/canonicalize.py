from __future__ import annotations
from fpgai.ir.graph import Graph

def canonicalize(g: Graph) -> Graph:
    # your importer already canonicalizes; keep this as a hook for later.
    return g
