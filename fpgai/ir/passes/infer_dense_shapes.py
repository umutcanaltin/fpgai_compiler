from __future__ import annotations

from fpgai.ir.graph import Graph


def infer_dense_shapes(g: Graph) -> Graph:
    """
    Populate Dense op attrs:
      - in_features
      - out_features
      - has_bias
    Uses weight initializer shape.
    """
    for op in g.ops:
        if op.op_type != "Dense":
            continue

        w_name = op.attrs.get("weight")
        if not w_name or w_name not in g.params:
            continue

        W = g.params[w_name]
        # common conventions:
        # - PyTorch Linear weight is [out_features, in_features]
        # - Some exports may use [in_features, out_features]
        # We'll detect using bias shape when available, otherwise assume torch layout.
        b_name = op.attrs.get("bias")
        B = g.params.get(b_name) if b_name else None

        out_features = None
        in_features = None

        if W.ndim == 2:
            r, c = int(W.shape[0]), int(W.shape[1])
            if B is not None and B.ndim == 1:
                # bias length matches out_features
                if int(B.shape[0]) == r:
                    out_features, in_features = r, c
                elif int(B.shape[0]) == c:
                    out_features, in_features = c, r
                else:
                    # fallback to torch layout
                    out_features, in_features = r, c
            else:
                # fallback to torch layout
                out_features, in_features = r, c

        op.attrs["out_features"] = out_features
        op.attrs["in_features"] = in_features
        op.attrs["has_bias"] = bool(b_name)

    return g
