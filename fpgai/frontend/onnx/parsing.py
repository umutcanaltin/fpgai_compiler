from __future__ import annotations

from typing import Dict, List, Tuple
import onnx
from onnx import numpy_helper

_ONNX_DTYPE_MAP = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32",
    7: "int64", 9: "bool", 10: "float16", 11: "float64", 12: "uint32", 13: "uint64",
}


def shape_from_value_info(vi: onnx.ValueInfoProto) -> Tuple[int, ...]:
    dims: List[int] = []
    for dim in vi.type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value) if dim.HasField("dim_value") else 1)
    return tuple(dims)


def dtype_from_value_info(vi: onnx.ValueInfoProto) -> str:
    et = int(vi.type.tensor_type.elem_type)
    return _ONNX_DTYPE_MAP.get(et, "float32")


def attr_to_py(a: onnx.AttributeProto):
    if a.type == onnx.AttributeProto.INT:
        return int(a.i)
    if a.type == onnx.AttributeProto.FLOAT:
        return float(a.f)
    if a.type == onnx.AttributeProto.STRING:
        return a.s.decode("utf-8", errors="ignore")
    if a.type == onnx.AttributeProto.INTS:
        return [int(x) for x in a.ints]
    if a.type == onnx.AttributeProto.FLOATS:
        return [float(x) for x in a.floats]
    if a.type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(a.t)
    return a


def collect_initializers(model: onnx.ModelProto) -> Dict[str, object]:
    out = {}
    for init in model.graph.initializer:
        out[init.name] = numpy_helper.to_array(init)
    return out
