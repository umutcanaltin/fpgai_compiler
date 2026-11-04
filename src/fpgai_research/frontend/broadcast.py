"""Static ONNX multidirectional broadcasting for elementwise operations.

Only ranks 0..4 and positive static dimensions are represented. All indices
are compile-time stride expressions; this is not a physical timing claim.
"""
from __future__ import annotations
from math import prod

def broadcast_shape(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(first,tuple) or not isinstance(second,tuple):
        raise ValueError("Broadcast requires tuple shape contracts")
    if max(len(first),len(second)) > 4 or any(type(d) is not int or d < 1 for d in (*first,*second)):
        raise ValueError("Broadcast supports only positive static rank<=4 operands")
    out=[]
    rank=max(len(first),len(second))
    for a,b in zip((1,)*(rank-len(first))+first,(1,)*(rank-len(second))+second):
        if a != b and a != 1 and b != 1:
            raise ValueError(f"Incompatible ONNX broadcast dimensions: {a} and {b}")
        out.append(max(a,b))
    return tuple(out)

def broadcast_flat_index(shape: tuple[int,...],result_shape: tuple[int,...],*,variable: str="i") -> str:
    """C++ row-major input offset from a flattened broadcast output offset."""
    if broadcast_shape(shape,result_shape) != result_shape:
        raise ValueError("Operand cannot broadcast to expected output shape")
    aligned=(1,)*(len(result_shape)-len(shape))+shape
    terms=[]
    for axis,size in enumerate(aligned):
        if size==1:
            continue
        output_stride=prod(result_shape[axis+1:])
        input_stride=prod(aligned[axis+1:])
        coordinate=f"({variable}%{size})" if output_stride==1 else f"(({variable}/{output_stride})%{size})"
        terms.append(coordinate if input_stride==1 else f"({coordinate}*{input_stride})")
    return " + ".join(terms) if terms else "0"
