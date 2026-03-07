from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from fpgai.ir.graph import Graph
from fpgai.ir.ops import Op


def _shape(x) -> Optional[Tuple[int, ...]]:
    if x is None or not hasattr(x, "shape"):
        return None
    return tuple(int(d) for d in x.shape)


def _is_w_shape(sh: Optional[Tuple[int, ...]]) -> bool:
    return sh is not None and len(sh) == 2


def _is_b_shape(sh: Optional[Tuple[int, ...]]) -> bool:
    return sh is not None and len(sh) == 1


def annotate_dense_params(graph: Graph) -> Graph:
    """
    Make Dense ops robust:
      - ensure attrs['weight'] and attrs['bias'] point to initializer tensors
      - fix accidental swap (weight<->bias) based on tensor ranks (2D vs 1D)
      - infer in_features/out_features/layout from weight tensor and input tensor spec

    This is intentionally conservative and won't guess if both candidates are ambiguous.
    """
    for op in graph.ops:
        if op.op_type != "Dense":
            continue

        w_name = op.attrs.get("weight")
        b_name = op.attrs.get("bias")

        W = graph.params.get(w_name) if isinstance(w_name, str) else None
        B = graph.params.get(b_name) if isinstance(b_name, str) else None

        w_sh = _shape(W)
        b_sh = _shape(B)

        # If weight is missing/invalid but bias looks like weight, swap
        if not _is_w_shape(w_sh) and _is_w_shape(b_sh):
            op.attrs["weight"], op.attrs["bias"] = b_name, w_name
            w_name, b_name = b_name, w_name
            W, B = B, W
            w_sh, b_sh = b_sh, w_sh

        # If bias is missing/invalid but weight looks like bias and we have nothing else, warn later
        # If no bias provided, it's fine, but we should not accidentally treat something else as bias.

        # Validate weight
        if not (isinstance(w_name, str) and w_name in graph.params and _is_w_shape(w_sh)):
            raise ValueError(
                f"Dense {op.name}: invalid weight binding. "
                f"weight={w_name!r} shape={w_sh} (must be 2D initializer)"
            )

        # Bias is optional, but if present it must be 1D
        if b_name is not None:
            if not (isinstance(b_name, str) and b_name in graph.params and _is_b_shape(b_sh)):
                raise ValueError(
                    f"Dense {op.name}: invalid bias binding. "
                    f"bias={b_name!r} shape={b_sh} (must be 1D initializer or omit bias)"
                )

        # Infer features/layout
        s0, s1 = int(W.shape[0]), int(W.shape[1])

        # Default assumption (PyTorch Linear exported): (out_features, in_features)
        layout = str(op.attrs.get("layout", "out_in")).lower()

        # If we can use input tensor spec to decide, do it
        x_name = op.inputs[0] if op.inputs else None
        x_spec = graph.get_tensor(x_name) if x_name else None
        x_last = int(x_spec.shape[-1]) if (x_spec and x_spec.shape) else None

        if x_last is not None:
            if x_last == s1:
                layout = "out_in"
                out_f, in_f = s0, s1
            elif x_last == s0:
                layout = "in_out"
                in_f, out_f = s0, s1
            else:
                # fallback
                layout = "out_in"
                out_f, in_f = s0, s1
        else:
            layout = "out_in"
            out_f, in_f = s0, s1

        op.attrs["layout"] = layout
        op.attrs["in_features"] = int(in_f)
        op.attrs["out_features"] = int(out_f)

    return graph
