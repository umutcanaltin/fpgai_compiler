from __future__ import annotations

from fpgai.ir.graph import Graph


def annotate_dense_features(g: Graph) -> None:
    for op in g.ops:
        if op.op_type != "Dense":
            continue

        w_name = op.attrs.get("weight")
        if not isinstance(w_name, str) or w_name not in g.params:
            continue

        W = g.params[w_name]
        if W is None or len(getattr(W, "shape", ())) != 2:
            continue

        s0, s1 = int(W.shape[0]), int(W.shape[1])
        layout = str(op.attrs.get("layout", "out_in")).lower()

        if layout in ("out_in", "outin", "out-in"):
            out_f, in_f = s0, s1
            layout = "out_in"
        elif layout in ("in_out", "inout", "in-out"):
            in_f, out_f = s0, s1
            layout = "in_out"
        else:
            out_f, in_f = s0, s1
            layout = "out_in"

        op.attrs["in_features"] = int(in_f)
        op.attrs["out_features"] = int(out_f)
        op.attrs["layout"] = layout
