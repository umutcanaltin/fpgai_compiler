from __future__ import annotations

from typing import Dict, List, Optional, Set, TYPE_CHECKING
import onnx

from fpgai.ir import Graph
from fpgai.ir.ops import Op, make_name
from fpgai.ir.types import TensorSpec
from fpgai.ir.passes.infer_shapes import infer_shapes as infer_ir_shapes

from .parsing import (
    shape_from_value_info,
    dtype_from_value_info,
    attr_to_py,
    collect_initializers,
)
from .canonicalize import canonicalize_op
from .patterns import fuse_matmul_add_to_dense
from .annotate import annotate_dense_features
from .external_import import try_import_external_node

if TYPE_CHECKING:
    from fpgai.operators.external import ExternalOperatorContext


def import_onnx(
    path: str,
    *,
    name: Optional[str] = None,
    canonicalize: bool = True,
    infer_shapes: bool = True,
    insert_missing_activations: bool = False,
    external_operator_context: "ExternalOperatorContext | None" = None,
) -> Graph:
    model = onnx.load(path)
    if infer_shapes:
        # Populate intermediate ValueInfo records before constructing the FPGAI
        # tensor table. ONNX shape inference is intentionally best-effort here:
        # models may contain custom-domain operators without installed schemas.
        # Standard operators preceding those custom nodes can still be inferred,
        # after which approved external callbacks infer their own outputs.
        try:
            model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        except Exception:
            # Preserve the historic importer behavior for models that ONNX cannot
            # infer. Existing graph input/output/value_info metadata remains usable.
            pass
    g = Graph(name=name or (model.graph.name if model.graph.name else "onnx_graph"))

    # Initializers (Weights/Biases)
    # Store them in g.constants so the backend can access the numpy arrays
    raw_inits = collect_initializers(model)
    g.constants = raw_inits  # <--- CRITICAL FIX
    g.params = raw_inits     # Keep params alias for backward compatibility

    init_names: Set[str] = set(g.constants.keys())

    # Inputs (exclude initializers)
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        g.inputs.append(inp.name)
        g.add_tensor(inp.name, shape_from_value_info(inp), dtype_from_value_info(inp))

    # Outputs
    for out in model.graph.output:
        g.outputs.append(out.name)
        g.add_tensor(out.name, shape_from_value_info(out), dtype_from_value_info(out))

    # Optional intermediates
    if infer_shapes:
        for vi in model.graph.value_info:
            if vi.name not in g.tensors:
                g.add_tensor(vi.name, shape_from_value_info(vi), dtype_from_value_info(vi))

    # Resolve the model opset per ONNX domain. Empty domain is ai.onnx.
    opsets = {(item.domain or "ai.onnx"): int(item.version) for item in model.opset_import}

    # Nodes -> ops (raw)
    raw_ops: List[Op] = []
    for idx, node in enumerate(model.graph.node):
        op_type = node.op_type
        op_name = make_name(op_type, idx, node.name if node.name else None)
        inputs = list(node.input)
        outputs = list(node.output)

        attrs: Dict[str, object] = {}
        for a in node.attribute:
            attrs[a.name] = attr_to_py(a)

        external_op = None
        if external_operator_context is not None:
            domain = node.domain or "ai.onnx"
            external_op = try_import_external_node(
                graph=g,
                node=node,
                op_name=op_name,
                attrs=attrs,
                domain=domain,
                opset=opsets.get(domain, 1),
                context=external_operator_context,
            )
        raw_ops.append(external_op or Op(name=op_name, op_type=op_type, inputs=inputs, outputs=outputs, attrs=attrs))

    if not canonicalize:
        g.ops = raw_ops
        return infer_ir_shapes(g) if infer_shapes else g

    # Canonicalize per-op
    ops = [canonicalize_op(op) for op in raw_ops]
    
    # Fuse patterns (using the constants keys)
    ops = fuse_matmul_add_to_dense(ops, params=set(g.constants.keys()))
    g.ops = ops

    # Annotate Dense features from weight shapes
    annotate_dense_features(g)

    # Run the conservative FPGAI IR shape pass on the final canonical graph.
    # External callbacks may have resolved custom-op outputs only after ONNX's
    # own shape inference stopped at the custom domain; downstream built-ins
    # such as Add must then be propagated from those newly known tensors.
    return infer_ir_shapes(g) if infer_shapes else g