"""Strict bounded static ONNX Squeeze/Unsqueeze shape contracts."""
from __future__ import annotations

def _normalize(axes,rank,*,allow_empty=False):
    if not isinstance(axes,(tuple,list)) or len(axes)>8 or (not allow_empty and not axes):
        raise ValueError("ONNX axes require a bounded constant integer vector")
    if any(type(axis) is not int for axis in axes):
        raise ValueError("ONNX axes must be integers")
    values=[]
    for axis in axes:
        index=axis+rank if axis<0 else axis
        if not 0<=index<rank or index in values:
            raise ValueError("ONNX axes out of range or duplicated")
        values.append(index)
    return tuple(values)

def squeeze_shape(shape: tuple[int,...],axes: tuple[int,...]|None=None)->tuple[int,...]:
    selected={i for i,d in enumerate(shape) if d==1} if axes is None else set(_normalize(axes,len(shape),allow_empty=True))
    if any(shape[i]!=1 for i in selected):
        raise ValueError("Squeeze axes must select singleton input dimensions")
    return tuple(dim for i,dim in enumerate(shape) if i not in selected)

def unsqueeze_shape(shape: tuple[int,...],axes: tuple[int,...])->tuple[int,...]:
    rank=len(shape)+len(axes)
    if rank>8:
        raise ValueError("Unsqueeze result rank exceeds supported static bound")
    selected=set(_normalize(axes,rank))
    iterator=iter(shape)
    return tuple(1 if i in selected else next(iterator) for i in range(rank))
