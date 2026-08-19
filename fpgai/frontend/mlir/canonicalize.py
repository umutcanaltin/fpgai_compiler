from __future__ import annotations

from typing import Dict

from fpgai.ir import Graph
from fpgai.ir.ops import Op


def _producer_map(graph: Graph) -> Dict[str, Op]:
    return {str(out): op for op in graph.ops for out in op.outputs}


def _strip_broadcasts(tensor: str, producers: Dict[str, Op]) -> tuple[str, list[Op]]:
    current = str(tensor)
    path: list[Op] = []
    while True:
        op = producers.get(current)
        if op is None or op.op_type != "Broadcast" or len(op.inputs) != 1:
            return current, path
        path.append(op)
        current = str(op.inputs[0])


def _find_reduce_max(tensor: str, producers: Dict[str, Op]) -> tuple[Op | None, list[Op]]:
    base, path = _strip_broadcasts(tensor, producers)
    op = producers.get(base)
    if op is not None and op.op_type == "ReduceMax":
        return op, path
    # JAX softmax guards the reduced maximum with maximum(-inf, reduced_max).
    if op is not None and op.op_type == "Maximum":
        for inp in op.inputs:
            candidate_base, candidate_path = _strip_broadcasts(str(inp), producers)
            candidate = producers.get(candidate_base)
            if candidate is not None and candidate.op_type == "ReduceMax":
                return candidate, path + [op] + candidate_path
    return None, path


def _axis_from_reduce(op: Op, rank: int | None) -> int | None:
    dims = (op.attrs or {}).get("dimensions")
    if not isinstance(dims, (list, tuple)) or len(dims) != 1:
        return None
    axis = int(dims[0])
    if rank is not None and axis < 0:
        axis += rank
    return axis


def eliminate_dead_ops(graph: Graph) -> int:
    """Remove operations not required to produce graph outputs.

    Constants/tensor specs are intentionally retained because they are harmless metadata and
    may be useful for diagnostics/provenance. The execution graph itself becomes canonical.
    """
    producers = _producer_map(graph)
    needed_ops: set[int] = set()
    stack = [str(x) for x in graph.outputs]
    constants = getattr(graph, "constants", {}) or {}
    while stack:
        tensor = stack.pop()
        if tensor in constants:
            continue
        op = producers.get(tensor)
        if op is None or id(op) in needed_ops:
            continue
        needed_ops.add(id(op))
        stack.extend(str(x) for x in op.inputs)
    before = len(graph.ops)
    graph.ops = [op for op in graph.ops if id(op) in needed_ops]
    return before - len(graph.ops)


def canonicalize_stablehlo_softmax(graph: Graph) -> int:
    """Recover logical Softmax from framework StableHLO decompositions.

    Recognizes the numerically-stable form used by JAX/PyTorch-style lowerings:
      reduce_max(x) -> [guard maximum] -> broadcast* -> sub -> exp
      exp -> reduce_sum -> broadcast* -> div
    """
    producers = _producer_map(graph)
    count = 0
    replacements: dict[int, Op] = {}

    for div in graph.ops:
        if div.op_type != "Div" or len(div.inputs) != 2 or len(div.outputs) != 1:
            continue
        exp = producers.get(str(div.inputs[0]))
        sum_base, _ = _strip_broadcasts(str(div.inputs[1]), producers)
        reduce_sum = producers.get(sum_base)
        if exp is None or exp.op_type != "Exp" or reduce_sum is None or reduce_sum.op_type != "ReduceSum":
            continue
        if not exp.inputs or not reduce_sum.inputs or str(reduce_sum.inputs[0]) != str(exp.outputs[0]):
            continue
        sub = producers.get(str(exp.inputs[0]))
        if sub is None or sub.op_type != "Sub" or len(sub.inputs) != 2:
            continue
        x = str(sub.inputs[0])
        reduce_max, _ = _find_reduce_max(str(sub.inputs[1]), producers)
        if reduce_max is None or not reduce_max.inputs or str(reduce_max.inputs[0]) != x:
            continue

        spec = graph.get_tensor(x)
        rank = len(tuple(spec.shape)) if spec is not None else None
        axis_max = _axis_from_reduce(reduce_max, rank)
        axis_sum = _axis_from_reduce(reduce_sum, rank)
        if axis_max is None or axis_sum is None or axis_max != axis_sum:
            continue

        softmax = Op(
            name=f"softmax_canonical_{count}",
            op_type="Softmax",
            inputs=[x],
            outputs=list(div.outputs),
            attrs={"axis": int(axis_max), "canonicalized_from": "stablehlo_decomposed_softmax"},
        )
        softmax.semantics.tags = ("canonicalized_from_stablehlo", "stablehlo_softmax_pattern")
        replacements[id(div)] = softmax
        count += 1

    if count:
        graph.ops = [replacements.get(id(op), op) for op in graph.ops]
        removed = eliminate_dead_ops(graph)
        graph.metadata.setdefault("canonicalizations", []).append({
            "pass": "stablehlo_softmax",
            "count": count,
            "dead_ops_removed": removed,
            "schema": "fpgai.mlir-canonicalization/v1",
        })
    return count



def _constant_scalar(graph: Graph, tensor: str, producers: Dict[str, Op]) -> float | None:
    base, _ = _strip_broadcasts(str(tensor), producers)
    value = (getattr(graph, "constants", {}) or {}).get(base)
    if value is None:
        return None
    try:
        import numpy as np
        arr = np.asarray(value).reshape(-1)
        if arr.size != 1:
            return None
        return float(arr[0])
    except Exception:
        return None


def _source_tensor(tensor: str, producers: Dict[str, Op]) -> str:
    base, _ = _strip_broadcasts(str(tensor), producers)
    return base


def _mean_pattern(graph: Graph, tensor: str, producers: Dict[str, Op]) -> tuple[Op | None, int | None]:
    """Return (ReduceSum, axis) for broadcast(reduce_sum(x))/constant mean forms."""
    base, _ = _strip_broadcasts(str(tensor), producers)
    op = producers.get(base)
    if op is None or op.op_type != "Div" or len(op.inputs) != 2:
        return None, None
    reduce_base, _ = _strip_broadcasts(str(op.inputs[0]), producers)
    reduce = producers.get(reduce_base)
    denom = _constant_scalar(graph, str(op.inputs[1]), producers)
    if reduce is None or reduce.op_type != "ReduceSum" or denom is None or denom <= 0:
        return None, None
    src = str(reduce.inputs[0]) if reduce.inputs else ""
    spec = graph.get_tensor(src)
    rank = len(tuple(spec.shape)) if spec is not None else None
    axis = _axis_from_reduce(reduce, rank)
    if axis is None:
        return None, None
    return reduce, axis


def canonicalize_stablehlo_rmsnorm(graph: Graph) -> int:
    """Recover RMSNorm from the common StableHLO decomposition emitted by JAX-like frontends."""
    producers = _producer_map(graph)
    replacements: dict[int, Op] = {}
    count = 0
    for final_mul in graph.ops:
        if final_mul.op_type != "Mul" or len(final_mul.inputs) != 2 or len(final_mul.outputs) != 1:
            continue
        for norm_side, scale_side in ((final_mul.inputs[0], final_mul.inputs[1]), (final_mul.inputs[1], final_mul.inputs[0])):
            norm_mul = producers.get(str(norm_side))
            if norm_mul is None or norm_mul.op_type != "Mul" or len(norm_mul.inputs) != 2:
                continue
            scale_source = _source_tensor(str(scale_side), producers)
            if graph.get_tensor(scale_source) is None:
                continue
            for x_side, rsqrt_side in ((norm_mul.inputs[0], norm_mul.inputs[1]), (norm_mul.inputs[1], norm_mul.inputs[0])):
                x = str(x_side)
                rsqrt_base = _source_tensor(str(rsqrt_side), producers)
                rsqrt = producers.get(rsqrt_base)
                if rsqrt is None or rsqrt.op_type != "Rsqrt" or len(rsqrt.inputs) != 1:
                    continue
                add = producers.get(str(rsqrt.inputs[0]))
                if add is None or add.op_type != "Add" or len(add.inputs) != 2:
                    continue
                mean_tensor = None; eps = None
                for mean_side, eps_side in ((add.inputs[0], add.inputs[1]), (add.inputs[1], add.inputs[0])):
                    candidate_eps = _constant_scalar(graph, str(eps_side), producers)
                    reduce, axis = _mean_pattern(graph, str(mean_side), producers)
                    if reduce is not None and candidate_eps is not None:
                        mean_tensor = str(mean_side); eps = candidate_eps; break
                if mean_tensor is None or eps is None:
                    continue
                reduce, axis = _mean_pattern(graph, mean_tensor, producers)
                square = producers.get(str(reduce.inputs[0])) if reduce and reduce.inputs else None
                if square is None or square.op_type != "Mul" or len(square.inputs) != 2 or str(square.inputs[0]) != x or str(square.inputs[1]) != x:
                    continue
                op = Op(
                    name=f"rmsnorm_canonical_{count}", op_type="RMSNorm",
                    inputs=[x, scale_source], outputs=list(final_mul.outputs),
                    attrs={"axis": int(axis), "epsilon": float(eps), "canonicalized_from": "stablehlo_decomposed_rmsnorm"},
                )
                op.semantics.tags = ("canonicalized_from_stablehlo", "stablehlo_rmsnorm_pattern")
                replacements[id(final_mul)] = op
                count += 1
                break
            if id(final_mul) in replacements:
                break
    if count:
        graph.ops = [replacements.get(id(op), op) for op in graph.ops]
        removed = eliminate_dead_ops(graph)
        graph.metadata.setdefault("canonicalizations", []).append({
            "pass": "stablehlo_rmsnorm", "count": count, "dead_ops_removed": removed,
            "schema": "fpgai.mlir-canonicalization/v1",
        })
    return count


def canonicalize_stablehlo_layernorm(graph: Graph) -> int:
    """Recover LayerNormalization from a standard mean/variance StableHLO decomposition."""
    producers = _producer_map(graph)
    replacements: dict[int, Op] = {}
    count = 0
    for final_add in graph.ops:
        if final_add.op_type != "Add" or len(final_add.inputs) != 2 or len(final_add.outputs) != 1:
            continue
        for scaled_side, bias_side in ((final_add.inputs[0], final_add.inputs[1]), (final_add.inputs[1], final_add.inputs[0])):
            scaled = producers.get(str(scaled_side))
            if scaled is None or scaled.op_type != "Mul" or len(scaled.inputs) != 2:
                continue
            bias_source = _source_tensor(str(bias_side), producers)
            if graph.get_tensor(bias_source) is None:
                continue
            for normalized_side, gamma_side in ((scaled.inputs[0], scaled.inputs[1]), (scaled.inputs[1], scaled.inputs[0])):
                normalized = producers.get(str(normalized_side))
                gamma_source = _source_tensor(str(gamma_side), producers)
                if normalized is None or normalized.op_type != "Mul" or len(normalized.inputs) != 2 or graph.get_tensor(gamma_source) is None:
                    continue
                for centered_side, rsqrt_side in ((normalized.inputs[0], normalized.inputs[1]), (normalized.inputs[1], normalized.inputs[0])):
                    centered = producers.get(str(centered_side))
                    rsqrt = producers.get(_source_tensor(str(rsqrt_side), producers))
                    if centered is None or centered.op_type != "Sub" or len(centered.inputs) != 2 or rsqrt is None or rsqrt.op_type != "Rsqrt":
                        continue
                    x = str(centered.inputs[0])
                    mean_reduce, mean_axis = _mean_pattern(graph, str(centered.inputs[1]), producers)
                    if mean_reduce is None or not mean_reduce.inputs or str(mean_reduce.inputs[0]) != x:
                        continue
                    add_eps = producers.get(str(rsqrt.inputs[0])) if rsqrt.inputs else None
                    if add_eps is None or add_eps.op_type != "Add" or len(add_eps.inputs) != 2:
                        continue
                    var_tensor = None; eps = None
                    for var_side, eps_side in ((add_eps.inputs[0], add_eps.inputs[1]), (add_eps.inputs[1], add_eps.inputs[0])):
                        candidate_eps = _constant_scalar(graph, str(eps_side), producers)
                        var_reduce, var_axis = _mean_pattern(graph, str(var_side), producers)
                        if var_reduce is not None and candidate_eps is not None and var_axis == mean_axis:
                            var_tensor = str(var_side); eps = candidate_eps; break
                    if var_tensor is None or eps is None:
                        continue
                    var_reduce, _ = _mean_pattern(graph, var_tensor, producers)
                    square = producers.get(str(var_reduce.inputs[0])) if var_reduce and var_reduce.inputs else None
                    if square is None or square.op_type != "Mul" or len(square.inputs) != 2:
                        continue
                    if str(square.inputs[0]) != str(centered.outputs[0]) or str(square.inputs[1]) != str(centered.outputs[0]):
                        continue
                    op = Op(
                        name=f"layernorm_canonical_{count}", op_type="LayerNormalization",
                        inputs=[x, gamma_source, bias_source], outputs=list(final_add.outputs),
                        attrs={"axis": int(mean_axis), "epsilon": float(eps), "canonicalized_from": "stablehlo_decomposed_layernorm"},
                    )
                    op.semantics.tags = ("canonicalized_from_stablehlo", "stablehlo_layernorm_pattern")
                    replacements[id(final_add)] = op
                    count += 1
                    break
                if id(final_add) in replacements:
                    break
            if id(final_add) in replacements:
                break
    if count:
        graph.ops = [replacements.get(id(op), op) for op in graph.ops]
        removed = eliminate_dead_ops(graph)
        graph.metadata.setdefault("canonicalizations", []).append({
            "pass": "stablehlo_layernorm", "count": count, "dead_ops_removed": removed,
            "schema": "fpgai.mlir-canonicalization/v1",
        })
    return count



def canonicalize_scalar_constants(graph: Graph) -> int:
    """Fold scalar arithmetic emitted around framework attention scaling."""
    import math
    import numpy as np
    folded = 0
    changed = True
    while changed:
        changed = False
        for op in graph.ops:
            if len(op.outputs) != 1:
                continue
            out = str(op.outputs[0])
            values = []
            for name in op.inputs:
                value = (getattr(graph, "constants", {}) or {}).get(str(name))
                if value is None:
                    values = []; break
                arr = np.asarray(value).reshape(-1)
                if arr.size != 1:
                    values = []; break
                values.append(float(arr[0]))
            if op.op_type in {"Sqrt", "Rsqrt", "Cast"} and len(values) == 1:
                if op.op_type == "Sqrt": result = math.sqrt(values[0])
                elif op.op_type == "Rsqrt": result = 1.0 / math.sqrt(values[0])
                else: result = values[0]
            elif op.op_type in {"Mul", "Div", "Add", "Sub", "Maximum"} and len(values) == 2:
                if op.op_type == "Mul": result = values[0] * values[1]
                elif op.op_type == "Div": result = values[0] / values[1]
                elif op.op_type == "Add": result = values[0] + values[1]
                elif op.op_type == "Sub": result = values[0] - values[1]
                else: result = max(values[0], values[1])
            else:
                continue
            dtype = getattr(graph.get_tensor(out), "dtype", "float32")
            graph.constants[out] = np.asarray(result, dtype=dtype)
            folded += 1; changed = True
        if changed:
            # Keep folding through users, but operations themselves are removed by DCE after rewrites.
            changed = False
    return folded


def canonicalize_scalar_broadcast_elementwise(graph: Graph) -> int:
    """Replace broadcast(scalar_constant) inputs to elementwise ops with the scalar constant itself."""
    producers = _producer_map(graph)
    changed = 0
    for op in graph.ops:
        if op.op_type not in {"Mul", "Div", "Add", "Sub"}:
            continue
        new_inputs = list(op.inputs)
        for i, name in enumerate(op.inputs):
            base, path = _strip_broadcasts(str(name), producers)
            if not path or base not in (getattr(graph, "constants", {}) or {}):
                continue
            try:
                import numpy as np
                if np.asarray(graph.constants[base]).size != 1:
                    continue
            except Exception:
                continue
            new_inputs[i] = base
            changed += 1
        op.inputs = new_inputs
    return changed


def _consumer_map(graph: Graph) -> Dict[str, list[Op]]:
    consumers: Dict[str, list[Op]] = {}
    for op in graph.ops:
        for inp in op.inputs:
            consumers.setdefault(str(inp), []).append(op)
    return consumers


def _strip_reshape_views(tensor: str, producers: Dict[str, Op]) -> tuple[str, list[Op]]:
    """Strip only shape-only StableHLO reshape views.

    Transpose is deliberately *not* stripped because it is a semantic operation for
    attention (notably K -> K^T).
    """
    current = str(tensor)
    path: list[Op] = []
    while True:
        op = producers.get(current)
        if op is None or op.op_type != "Reshape" or len(op.inputs) != 1:
            return current, path
        path.append(op)
        current = str(op.inputs[0])


def _batch1_matmul_shape(graph: Graph, left: str, right: str) -> tuple[int, int, int] | None:
    left_spec = graph.get_tensor(left)
    right_spec = graph.get_tensor(right)
    if left_spec is None or right_spec is None:
        return None
    try:
        ls = tuple(int(x) for x in left_spec.shape)
        rs = tuple(int(x) for x in right_spec.shape)
    except Exception:
        return None
    if len(ls) != 3 or len(rs) != 3 or ls[0] != 1 or rs[0] != 1:
        return None
    if ls[-1] != rs[-2]:
        return None
    return (1, ls[-2], rs[-1])


def _find_batch1_matmul_output_view(
    graph: Graph,
    matmul: Op,
    expected_shape: tuple[int, int, int],
    consumers: Dict[str, list[Op]],
) -> Op | None:
    """Find the final shape/layout wrapper restoring a logical [1,M,N] result.

    JAX StableHLO has used several equivalent dot_general layouts across releases,
    including MatMul->Transpose, MatMul->Reshape->Transpose and
    MatMul->Transpose->Reshape.  Walk a single-use view chain rather than keying the
    canonicalizer to one exporter version.
    """
    if len(matmul.outputs) != 1:
        return None
    current_tensor = str(matmul.outputs[0])
    current_op: Op = matmul
    for _ in range(4):
        spec = graph.get_tensor(current_tensor)
        if spec is not None:
            try:
                if tuple(int(x) for x in spec.shape) == expected_shape:
                    return current_op
            except Exception:
                pass
        users = consumers.get(current_tensor, [])
        if len(users) != 1:
            return None
        user = users[0]
        if len(user.outputs) != 1:
            return None
        if user.op_type in {"Reshape", "Transpose"}:
            pass
        elif user.op_type == "Broadcast":
            # JAX 0.11 portable StableHLO restores a flattened batch-1
            # dot_general result with broadcast_in_dim, e.g. [M,N] ->
            # [1,M,N] using dimensions [1,2]. Accept only this exact
            # batch restoration; arbitrary broadcasting is semantic.
            src_spec = graph.get_tensor(current_tensor)
            dst_spec = graph.get_tensor(str(user.outputs[0]))
            dims = (user.attrs or {}).get("broadcast_dimensions")
            try:
                src_shape = tuple(int(x) for x in src_spec.shape) if src_spec is not None else ()
                dst_shape = tuple(int(x) for x in dst_spec.shape) if dst_spec is not None else ()
            except Exception:
                return None
            if not (
                len(src_shape) == 2
                and len(dst_shape) == 3
                and dst_shape[0] == 1
                and dst_shape[1:] == src_shape
                and list(dims or []) == [1, 2]
            ):
                return None
        else:
            return None
        current_op = user
        current_tensor = str(user.outputs[0])
    return None


def canonicalize_batch1_dot_layout(graph: Graph) -> int:
    """Collapse JAX batch-1 dot_general view plumbing into logical MatMul.

    The rewrite is structural and shape-proved.  It strips Reshape views from either
    dot operand, preserves semantic Transpose operations, and accepts a short chain
    of Reshape/Transpose views on the result.  This covers portable StableHLO emitted
    by multiple JAX releases without making FPGAI depend on a single textual layout.
    """
    producers = _producer_map(graph)
    consumers = _consumer_map(graph)
    replacements: dict[int, Op] = {}
    count = 0

    for matmul in graph.ops:
        if matmul.op_type != "MatMul" or len(matmul.inputs) != 2 or len(matmul.outputs) != 1:
            continue
        # Limit this canonicalization to imported StableHLO dot_general nodes.
        if (matmul.attrs or {}).get("stablehlo_op") != "dot_general":
            continue

        left, left_views = _strip_reshape_views(str(matmul.inputs[0]), producers)
        right, right_views = _strip_reshape_views(str(matmul.inputs[1]), producers)
        expected = _batch1_matmul_shape(graph, left, right)
        if expected is None:
            continue

        final_view = _find_batch1_matmul_output_view(graph, matmul, expected, consumers)
        if final_view is None:
            continue

        # If there is no view plumbing at all there is nothing to canonicalize.
        if not left_views and not right_views and final_view is matmul:
            continue

        new_op = Op(
            name=f"matmul_batch1_canonical_{count}",
            op_type="MatMul",
            inputs=[left, right],
            outputs=list(final_view.outputs),
            attrs={
                "canonicalized_from": "stablehlo_batch1_dot_general_layout",
                "input_reshape_views_removed": len(left_views) + len(right_views),
                "output_view_terminal": final_view.op_type,
            },
        )
        new_op.semantics.tags = ("canonicalized_from_stablehlo", "stablehlo_batch1_matmul_layout")
        replacements[id(final_view)] = new_op
        count += 1

    if count:
        graph.ops = [replacements.get(id(op), op) for op in graph.ops]
        removed = eliminate_dead_ops(graph)
        graph.metadata.setdefault("canonicalizations", []).append({
            "pass": "stablehlo_batch1_dot_layout",
            "count": count,
            "dead_ops_removed": removed,
            "schema": "fpgai.mlir-canonicalization/v1",
        })
    return count


def canonicalize_stablehlo(graph: Graph) -> Graph:
    folded = canonicalize_scalar_constants(graph)
    rewired = canonicalize_scalar_broadcast_elementwise(graph)
    canonicalize_batch1_dot_layout(graph)
    canonicalize_stablehlo_softmax(graph)
    canonicalize_stablehlo_layernorm(graph)
    canonicalize_stablehlo_rmsnorm(graph)
    if folded or rewired:
        graph.metadata.setdefault("canonicalizations", []).append({
            "pass": "stablehlo_scalar_constants", "folded": folded, "broadcast_rewrites": rewired,
            "schema": "fpgai.mlir-canonicalization/v1",
        })
    eliminate_dead_ops(graph)
    return graph
