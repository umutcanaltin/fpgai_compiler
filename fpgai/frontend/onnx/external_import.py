from __future__ import annotations
from typing import Any, Mapping
from fpgai.ir.ops import Op
from fpgai.operators.external import (
    ExternalOperatorContext, OnnxImportContext, ShapeInferenceContext, TypeInferenceContext,
    TensorDescriptor,
)

def _tensor_map(graph) -> dict[str,TensorDescriptor]:
    return {name:TensorDescriptor(name,tuple(spec.shape),str(spec.dtype)) for name,spec in graph.tensors.items()}

def try_import_external_node(*, graph, node, op_name: str, attrs: Mapping[str,Any], domain: str, opset: int, context: ExternalOperatorContext) -> Op|None:
    selected=context.binding_registry.resolve(domain,node.op_type,opset)
    if selected is None: return None
    definition=selected.definition
    imported=definition.onnx_import(OnnxImportContext(
        domain=domain,op_type=node.op_type,opset=opset,node_name=op_name,
        inputs=tuple(node.input),outputs=tuple(node.output),attributes=dict(attrs),tensors=_tensor_map(graph),
    ))
    if not imported.canonical_op_type or not imported.outputs: raise ValueError("External ONNX importer returned an invalid operator")
    input_specs=tuple(_tensor_map(graph)[name] for name in imported.inputs if name in graph.tensors)
    shapes=None; dtypes=None
    if definition.shape_inference and len(input_specs)==len(imported.inputs):
        shapes=definition.shape_inference(ShapeInferenceContext(imported.attributes,input_specs)).output_shapes
    if definition.type_inference and len(input_specs)==len(imported.inputs):
        dtypes=definition.type_inference(TypeInferenceContext(imported.attributes,tuple(x.dtype for x in input_specs))).output_dtypes
    if shapes is not None and len(shapes)!=len(imported.outputs): raise ValueError("Shape inference output count mismatch")
    if dtypes is not None and len(dtypes)!=len(imported.outputs): raise ValueError("Type inference output count mismatch")
    for index,name in enumerate(imported.outputs):
        inferred_shape=tuple(shapes[index]) if shapes is not None else None
        inferred_dtype=str(dtypes[index]) if dtypes is not None else None
        known=graph.get_tensor(name)
        if known is not None:
            if inferred_shape is not None and tuple(known.shape) and tuple(known.shape)!=inferred_shape:
                raise ValueError(f"External shape inference disagrees for tensor {name}")
            continue
        if inferred_shape is not None:
            graph.add_tensor(name,inferred_shape,inferred_dtype or "float32")
    provenance={"package_id":selected.package_id,"package_version":selected.package_version,"operator_id":definition.contract.operator_id,"operator_semantics_version":definition.contract.version,"manifest_sha256":selected.manifest_hash,"capabilities":definition.contract.capabilities.to_dict(),"category":definition.contract.category}
    normalized=dict(imported.attributes); normalized["_fpgai_external_operator"]=provenance
    return Op(name=op_name,op_type=imported.canonical_op_type,inputs=list(imported.inputs),outputs=list(imported.outputs),attrs=normalized)
