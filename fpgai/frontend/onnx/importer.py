from __future__ import annotations

from typing import Dict, List, Optional, Set, TYPE_CHECKING
import numpy as np
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
    shape_overrides: Optional[Dict[str, object]] = None,
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
    g.constants = raw_inits
    g.params = raw_inits     # Keep params alias for backward compatibility

    # Initializers are tensors semantically, even though they are not runtime
    # graph inputs. Register them in the canonical tensor table so downstream
    # passes (quantization, type propagation, reporting) can attach metadata to
    # weights and biases without maintaining a second tensor namespace.
    for init_name, init_value in raw_inits.items():
        init_array = np.asarray(init_value)
        g.add_tensor(init_name, tuple(int(dim) for dim in init_array.shape), str(init_array.dtype))

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
        domain = node.domain or "ai.onnx"
        attrs["_onnx_domain"] = domain
        attrs["_onnx_opset"] = int(opsets.get(domain, 1))

        # ONNX Constant nodes are compile-time tensors, not executable hardware
        # operators. Materialize their numeric payload into the canonical constant
        # table so allowlist validation and backend lowering see ordinary static
        # tensor inputs, just like graph initializers.
        if op_type == "Constant" and outputs:
            numeric_keys = ("value", "value_float", "value_int", "value_floats", "value_ints")
            payload = next((attrs[key] for key in numeric_keys if key in attrs), None)
            if payload is not None:
                array = np.asarray(payload)
                if array.dtype.kind in {"b", "i", "u", "f", "c"}:
                    output = str(outputs[0])
                    g.constants[output] = array
                    g.params = g.constants
                    existing = g.get_tensor(output)
                    dtype = str(array.dtype) if array.dtype is not None else str(getattr(existing, "dtype", "float32"))
                    g.add_tensor(output, tuple(int(dim) for dim in array.shape), dtype)
                    continue

        external_op = None
        if external_operator_context is not None:
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
    
    # Lower static ONNX Split into ordinary Slice operators so the backend
    # remains layerwise and does not need a model- or multi-output-specific kernel.
    lowered_ops: List[Op] = []
    for op in ops:
        if op.op_type != "Split":
            lowered_ops.append(op)
            continue
        attrs = dict(op.attrs or {})
        axis = int(attrs.get("axis", 0))
        split_sizes = attrs.get("split")
        if split_sizes is None and len(op.inputs) > 1 and op.inputs[1] in g.constants:
            split_sizes = [int(x) for x in np.asarray(g.constants[op.inputs[1]]).reshape(-1).tolist()]
        if split_sizes is None:
            sizes = []
            for output in op.outputs:
                spec = g.get_tensor(output)
                if spec is None or not getattr(spec, "shape", None):
                    sizes = []
                    break
                rank = len(spec.shape); resolved_axis = axis + rank if axis < 0 else axis
                if resolved_axis < 0 or resolved_axis >= rank:
                    sizes = []
                    break
                sizes.append(int(spec.shape[resolved_axis]))
            split_sizes = sizes or None
        if split_sizes is None or len(split_sizes) != len(op.outputs):
            lowered_ops.append(op)
            continue
        cursor = 0
        for output_index, (output, size) in enumerate(zip(op.outputs, split_sizes)):
            size = int(size)
            lowered_ops.append(Op(
                name=f"{op.name}__slice_{output_index}",
                op_type="Slice",
                inputs=[op.inputs[0]],
                outputs=[output],
                attrs={"starts": [cursor], "ends": [cursor + size], "axes": [axis], "steps": [1], "canonicalized_from": "Split"},
            ))
            cursor += size
    ops = lowered_ops

    # Fuse patterns (using the constants keys)
    ops = fuse_matmul_add_to_dense(ops, params=set(g.constants.keys()))
    g.ops = ops

    # Real exported graphs commonly retain symbolic sequence/cache dimensions.
    # FPGA hardware needs bounded static extents, so allow the user/config to
    # resolve those tensor shapes explicitly without changing the source model.
    # This is tensor-name driven and works for arbitrary models/frontends.
    if shape_overrides:
        applied: Dict[str, list[int]] = {}
        for tensor_name, raw_shape in dict(shape_overrides).items():
            if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
                raise ValueError(f"ONNXSHAPE001: shape override for {tensor_name!r} must be a non-empty list/tuple")
            shape = tuple(int(x) for x in raw_shape)
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"ONNXSHAPE002: shape override for {tensor_name!r} must contain positive static dimensions")
            existing = g.get_tensor(str(tensor_name))
            dtype = str(getattr(existing, "dtype", "float32")) if existing is not None else "float32"
            semantics = getattr(existing, "semantics", None) if existing is not None else None
            quantization = getattr(existing, "quantization", None) if existing is not None else None
            g.add_tensor(str(tensor_name), shape, dtype, semantics=semantics, quantization=quantization)
            applied[str(tensor_name)] = list(shape)
        g.metadata["shape_overrides"] = applied

    # Annotate Dense features from weight shapes
    annotate_dense_features(g)

    # Run the conservative FPGAI IR shape pass on the final canonical graph.
    # External callbacks may have resolved custom-op outputs only after ONNX's
    # own shape inference stopped at the custom domain; downstream built-ins
    # such as Add must then be propagated from those newly known tensors.
    return infer_ir_shapes(g) if infer_shapes else g