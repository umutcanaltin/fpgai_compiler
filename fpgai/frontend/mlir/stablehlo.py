from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from fpgai.ir import Graph, annotate_default_hardware_semantics
from fpgai.ir.passes.infer_shapes import infer_shapes
from .canonicalize import canonicalize_stablehlo


class StableHLOImportError(ValueError):
    pass


_TYPE_RE = re.compile(r"tensor<([^>]*)>")
_FUNC_RE = re.compile(r"func\.func(?:\s+public)?\s+@([A-Za-z0-9_.$-]+)\s*\((.*?)\)\s*(?:->\s*([^\{]+))?\{", re.S)
_ARG_RE = re.compile(r"(%[A-Za-z0-9_.$-]+)\s*:\s*(tensor<[^>]+>)")
_GENERIC_OP_RE = re.compile(
    r"(?P<results>%[A-Za-z0-9_.$-]+(?:\s*,\s*%[A-Za-z0-9_.$-]+)*)\s*=\s*"
    r'"stablehlo\.(?P<op>[A-Za-z0-9_]+)"\((?P<operands>[^)]*)\)'
    r"(?P<attrs>\s*\{.*?\})?\s*:\s*\((?P<intypes>.*?)\)\s*->\s*(?P<outtypes>\([^\n]*\)|tensor<[^>]+>)",
    re.S,
)
_PRETTY_OP_RE = re.compile(
    r"(?P<result>%[A-Za-z0-9_.$-]+)\s*=\s*stablehlo\.(?P<op>[A-Za-z0-9_]+)\s+"
    r"(?P<body>.*?)\s*:\s*(?P<types>.+?)\s*$"
)
_REDUCE_PRETTY_RE = re.compile(
    r"(?P<result>%[A-Za-z0-9_.$-]+)\s*=\s*stablehlo\.reduce\(?(?P<input>%[A-Za-z0-9_.$-]+)\s+init:\s*(?P<init>%[A-Za-z0-9_.$-]+)\)?\s*"
    r"applies\s+stablehlo\.(?P<reducer>maximum|add)\s+across\s+dimensions\s*=\s*\[(?P<dims>[^\]]*)\]\s*:\s*"
    r"\((?P<intype>tensor<[^>]+>),\s*tensor<[^>]+>\)\s*->\s*(?P<outtype>tensor<[^>]+>)",
    re.S,
)
_RETURN_GENERIC_RE = re.compile(r'"func\.return"\((?P<values>[^)]*)\)')
_RETURN_PRETTY_RE = re.compile(r"(?:func\.)?return\s+(?P<values>%[^:\n]+)")


def _read(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    candidate = str(source)
    if "\n" not in candidate and len(candidate) < 4096:
        path = Path(candidate)
        if path.exists():
            return path.read_text(encoding="utf-8")
    return candidate


def _tensor_type(value: str) -> tuple[tuple[int, ...], str]:
    match = _TYPE_RE.fullmatch(value.strip()) or _TYPE_RE.search(value)
    if not match:
        raise StableHLOImportError(f"MLIRSTABLE001: unsupported tensor type {value!r}")
    body = match.group(1)
    parts = body.split("x")
    elem = parts[-1]
    dims = parts[:-1]
    shape: list[int] = []
    for dim in dims:
        dim = dim.strip()
        if dim == "?":
            shape.append(-1)
        elif dim:
            shape.append(int(dim))
    dtype = {
        "f16": "float16", "f32": "float32", "f64": "float64",
        "i1": "bool", "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
        "ui8": "uint8", "ui16": "uint16", "ui32": "uint32",
    }.get(elem, elem)
    return tuple(shape), dtype


def _result_types(text: str) -> list[str]:
    return re.findall(r"tensor<[^>]+>", text)


def _parse_dense_literal(attrs: str) -> tuple[Any, str] | None:
    m = re.search(r"value\s*=\s*dense<(.+?)>\s*:\s*(tensor<[^>]+>)", attrs, re.S)
    if not m:
        return None
    raw = m.group(1).strip()
    ty = m.group(2)
    normalized = raw.replace("true", "True").replace("false", "False")
    try:
        value = ast.literal_eval(normalized)
    except Exception:
        try:
            value = float(raw) if any(c in raw for c in ".eE") else int(raw, 0)
        except Exception as exc:
            raise StableHLOImportError(f"MLIRSTABLE002: unsupported dense constant {raw!r}") from exc
    return value, ty


def _parse_perm(attrs_or_body: str) -> list[int] | None:
    for pattern in (
        r"permutation\s*=\s*dense<\[([^\]]+)\]>",
        r"permutation\s*=\s*\[([^\]]+)\]",
        r"dims\s*=\s*\[([^\]]+)\]",
    ):
        m = re.search(pattern, attrs_or_body)
        if m:
            return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    return None


def _map_op(op: str) -> str | None:
    return {
        "add": "Add",
        "multiply": "Mul",
        "dot": "MatMul",
        "dot_general": "MatMul",
        "transpose": "Transpose",
        "reshape": "Reshape",
        "maximum": "Maximum",
        "reduce": None,
        "convert": "Cast",
        "broadcast_in_dim": "Broadcast",
        "subtract": "Sub",
        "divide": "Div",
        "exponential": "Exp",
        "rsqrt": "Rsqrt",
        "sqrt": "Sqrt",
    }.get(op)


def _matmul_like_dot_general(attrs: str) -> bool:
    # Conservative first-phase acceptance: common last-dimension x penultimate-dimension
    # contraction, optionally with matching batch dimensions. If attributes are absent,
    # preserve the op as MatMul because the result type still provides shape information.
    if "dot_dimension_numbers" not in attrs and "contracting_dims" not in attrs:
        return True
    lhs = re.search(r"lhs_contracting_dimensions\s*=\s*\[([^\]]*)\]", attrs)
    rhs = re.search(r"rhs_contracting_dimensions\s*=\s*\[([^\]]*)\]", attrs)
    if lhs and rhs:
        return len([x for x in lhs.group(1).split(",") if x.strip()]) == 1 and len([x for x in rhs.group(1).split(",") if x.strip()]) == 1
    return True


def _add_tensor_from_type(graph: Graph, name: str, type_text: str) -> None:
    shape, dtype = _tensor_type(type_text)
    graph.add_tensor(name, shape, dtype)


def _canonical_name(value_id: str) -> str:
    return value_id.lstrip("%")


def _parse_function(text: str) -> tuple[str, list[tuple[str, str]], str]:
    m = _FUNC_RE.search(text)
    if not m:
        raise StableHLOImportError("MLIRSTABLE003: no func.func entry point found")
    args = [(a, t) for a, t in _ARG_RE.findall(m.group(2))]
    return m.group(1), args, m.group(3) or ""


def import_stablehlo_mlir(
    source: str | Path,
    *,
    pipeline_mode: str = "inference",
    target_board: str | None = None,
    source_framework: str | None = None,
) -> Graph:
    """Import a conservative textual StableHLO subset into FPGAI IR v2.

    This is an interoperability frontend, not a claim of complete StableHLO coverage.
    Unsupported StableHLO operations fail explicitly. Result tensor types from MLIR are
    retained so shape information is not guessed.
    """
    text = _read(source)
    func_name, args, _ = _parse_function(text)
    graph = Graph(func_name)
    graph.semantics.source_ir = "stablehlo"
    graph.semantics.source_metadata = {
        "framework": source_framework or "unknown",
        "frontend": "fpgai.frontend.mlir.stablehlo",
        "coverage": "supported_subset",
    }
    value_to_tensor: Dict[str, str] = {}
    for value_id, type_text in args:
        name = _canonical_name(value_id)
        _add_tensor_from_type(graph, name, type_text)
        graph.inputs.append(name)
        value_to_tensor[value_id] = name

    consumed_spans: list[tuple[int, int]] = []
    op_index = 0
    for match in _GENERIC_OP_RE.finditer(text):
        consumed_spans.append(match.span())
        stable_op = match.group("op")
        results = [x.strip() for x in match.group("results").split(",")]
        operands = [x.strip() for x in match.group("operands").split(",") if x.strip()]
        attrs = match.group("attrs") or ""
        outtypes = _result_types(match.group("outtypes"))
        if stable_op == "constant":
            parsed = _parse_dense_literal(attrs)
            if parsed is None or len(results) != 1:
                raise StableHLOImportError("MLIRSTABLE004: stablehlo.constant requires a dense value")
            value, type_text = parsed
            name = _canonical_name(results[0])
            _add_tensor_from_type(graph, name, type_text)
            dtype = graph.tensors[name].dtype
            graph.constants[name] = np.asarray(value, dtype=dtype)
            value_to_tensor[results[0]] = name
            continue
        mapped = _map_op(stable_op)
        if mapped is None:
            raise StableHLOImportError(f"MLIRSTABLE005: unsupported StableHLO op stablehlo.{stable_op}")
        if stable_op == "dot_general" and not _matmul_like_dot_general(attrs):
            raise StableHLOImportError("MLIRSTABLE006: dot_general dimensions are outside the current MatMul-compatible subset")
        input_names = [value_to_tensor.get(v, _canonical_name(v)) for v in operands]
        output_names = [_canonical_name(v) for v in results]
        op_attrs: Dict[str, Any] = {"stablehlo_op": stable_op}
        if mapped == "Transpose":
            perm = _parse_perm(attrs)
            if perm is not None:
                op_attrs["perm"] = perm
        op = graph.add_op(mapped, input_names, output_names, name=f"{mapped.lower()}_{op_index}", attrs=op_attrs)
        op.semantics.tags = tuple(op.semantics.tags) + ("imported_from_stablehlo",)
        for idx, result in enumerate(results):
            name = output_names[idx]
            value_to_tensor[result] = name
            if idx < len(outtypes):
                _add_tensor_from_type(graph, name, outtypes[idx])
        op_index += 1

    # StableHLO pretty reduction syntax contains an explicit reducer region and therefore
    # does not match the simple single-result pretty-op expression below.
    for match in _REDUCE_PRETTY_RE.finditer(text):
        reducer = match.group("reducer")
        result = match.group("result")
        input_id = match.group("input")
        output = _canonical_name(result)
        dims = [int(x.strip()) for x in match.group("dims").split(",") if x.strip()]
        mapped = "ReduceMax" if reducer == "maximum" else "ReduceSum"
        graph.add_op(
            mapped,
            [value_to_tensor.get(input_id, _canonical_name(input_id))],
            [output],
            name=f"{mapped.lower()}_{op_index}",
            attrs={"dimensions": dims, "keepdims": True, "stablehlo_op": "reduce", "stablehlo_reducer": reducer},
        )
        _add_tensor_from_type(graph, output, match.group("outtype"))
        value_to_tensor[result] = output
        op_index += 1

    # Pretty syntax support for common single-result StableHLO ops.  Real JAX
    # exports often use functional type syntax such as
    #   : (tensor<...>) -> tensor<...>
    # rather than only `: tensor<...>`, so take the final tensor type as the result.
    lines = text.splitlines()
    for line in lines:
        if '"stablehlo.' in line:
            continue
        m = _PRETTY_OP_RE.search(line.strip())
        if not m:
            continue
        stable_op = m.group("op")
        if stable_op == "reduce":
            continue  # handled by _REDUCE_PRETTY_RE above
        result_types = _result_types(m.group("types"))
        if not result_types:
            continue
        result_type = result_types[-1]
        if stable_op == "constant":
            cm = re.search(r"dense<(.+?)>\s*", m.group("body"))
            if not cm:
                raise StableHLOImportError("MLIRSTABLE007: unsupported pretty constant")
            raw = cm.group(1).strip()
            try:
                value = ast.literal_eval(raw.replace("true", "True").replace("false", "False"))
            except Exception:
                try:
                    value = float(raw) if any(c in raw for c in ".eE") else int(raw, 0)
                except Exception:
                    # Hex bit-pattern float constants (for example -inf in JAX softmax)
                    # are retained as integer payloads. Canonicalization removes such
                    # implementation constants when they are not part of logical FPGAI IR.
                    value = 0
            result = m.group("result")
            name = _canonical_name(result)
            _add_tensor_from_type(graph, name, result_type)
            graph.constants[name] = np.asarray(value, dtype=graph.tensors[name].dtype)
            value_to_tensor[result] = name
            continue
        mapped = _map_op(stable_op)
        if mapped is None:
            raise StableHLOImportError(f"MLIRSTABLE008: unsupported StableHLO op stablehlo.{stable_op}")
        body = m.group("body")
        operands = re.findall(r"%[A-Za-z0-9_.$-]+", body)
        result = m.group("result")
        output = _canonical_name(result)
        attrs: Dict[str, Any] = {"stablehlo_op": stable_op}
        if mapped == "Transpose":
            perm = _parse_perm(body)
            if perm is not None:
                attrs["perm"] = perm
        if mapped == "Broadcast":
            dims = re.search(r"dims\s*=\s*\[([^\]]*)\]", body)
            if dims:
                attrs["broadcast_dimensions"] = [int(x.strip()) for x in dims.group(1).split(",") if x.strip()]
        if stable_op == "dot_general" and not _matmul_like_dot_general(body):
            raise StableHLOImportError("MLIRSTABLE006: dot_general dimensions are outside the current MatMul-compatible subset")
        op = graph.add_op(mapped, [value_to_tensor.get(v, _canonical_name(v)) for v in operands], [output], name=f"{mapped.lower()}_{op_index}", attrs=attrs)
        op.semantics.tags = tuple(op.semantics.tags) + ("imported_from_stablehlo",)
        _add_tensor_from_type(graph, output, result_type)
        value_to_tensor[result] = output
        op_index += 1

    ret = _RETURN_GENERIC_RE.search(text) or _RETURN_PRETTY_RE.search(text)
    if ret:
        values = re.findall(r"%[A-Za-z0-9_.$-]+", ret.group("values"))
        graph.outputs = [value_to_tensor.get(v, _canonical_name(v)) for v in values]
    if not graph.outputs and graph.ops:
        graph.outputs = list(graph.ops[-1].outputs)
    if not graph.ops:
        raise StableHLOImportError("MLIRSTABLE009: no supported StableHLO operations found")

    canonicalize_stablehlo(graph)
    infer_shapes(graph)
    annotate_default_hardware_semantics(graph, pipeline_mode=pipeline_mode, target_board=target_board)
    graph.semantics.source_ir = "stablehlo"
    graph.semantics.source_metadata.update({
        "framework": source_framework or "unknown",
        "frontend": "fpgai.frontend.mlir.stablehlo",
        "coverage": "supported_subset",
    })
    return graph
