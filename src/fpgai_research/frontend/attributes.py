"""Preserve ONNX semantics while accepting explicitly emitted default attributes."""
from __future__ import annotations

def canonical_attributes(op: str, attrs: dict) -> dict:
    result=dict(attrs)
    if op not in ("Conv","MaxPool"):
        return result
    allowed=({"strides","pads","group","dilations","kernel_shape","auto_pad"}
             if op=="Conv" else
             {"kernel_shape","strides","pads","auto_pad","ceil_mode","dilations","storage_order"})
    unknown=set(result)-allowed
    if unknown:
        raise ValueError(f"Unsupported {op} attributes: {sorted(unknown)}")
    auto_pad=result.pop("auto_pad","NOTSET")
    if isinstance(auto_pad,bytes):
        auto_pad=auto_pad.decode("ascii")
    if auto_pad!="NOTSET":
        raise ValueError(f"{op} automatic padding {auto_pad!r} is unsupported")
    if op=="MaxPool":
        for name,expected in (("ceil_mode",0),("storage_order",0)):
            if int(result.pop(name,expected))!=expected:
                raise ValueError(f"MaxPool {name} must be {expected}")
        if tuple(result.pop("dilations",(1,1)))!=(1,1):
            raise ValueError("Dilated MaxPool unsupported")
    for key in ("kernel_shape","strides","pads","dilations"):
        if key in result:
            result[key]=tuple(result[key])
    return result
