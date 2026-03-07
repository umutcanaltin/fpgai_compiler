from __future__ import annotations
from fpgai.ir.graph import Graph

def infer_shapes(g: Graph) -> Graph:
    # assumes importer already calls infer; keep here for pipeline control
    return g
