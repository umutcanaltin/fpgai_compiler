from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import numpy as np

from fpgai.ir.graph import Graph
from fpgai.util.fs import ensure_clean_dir, write_text
from .project import HLSProject

from .emit.types_h import emit_types_h
from .emit.params_h import emit_params_h_stub
from .emit.params_cpp import emit_params_cpp
from .emit.layers_dense import emit_dense_h, emit_dense_cpp
from .emit.layers_conv import emit_conv_h, emit_conv_cpp
from .emit.layers_pool import emit_pool_h, emit_pool_cpp
from .emit.layers_activations import emit_activations_h, emit_activations_cpp
from .emit.model_inst_cpp import emit_model_inst_cpp
from .emit.top_cpp import emit_top_cpp
from .emit.top_train_cpp import emit_top_train_cpp
from .emit.weights_runtime_h import emit_weights_runtime_h
from .emit.weights_runtime_cpp import emit_weights_runtime_cpp
from .emit.csim_tcl import emit_csim_tcl
from .emit.csim_train_tcl import emit_csim_train_tcl
from fpgai.backends.hls.testbench import emit_tb_cpp
from fpgai.backends.hls.testbench_train import emit_tb_train_cpp


def _as_numpy_numeric(x):
    if x is None:
        return None
    if isinstance(x, (str, bytes)):
        return None
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.dtype.kind in ("U", "S", "O"):
        return None
    return arr.astype(np.float32, copy=False)


def _flatten_from_graph_named(graph: Graph, tensor_name: str):
    if tensor_name is None:
        return None

    if hasattr(graph, "constants") and tensor_name in graph.constants:
        arr = _as_numpy_numeric(graph.constants[tensor_name])
        if arr is not None:
            return arr.reshape(-1)

    if hasattr(graph, "params") and tensor_name in graph.params:
        arr = _as_numpy_numeric(graph.params[tensor_name])
        if arr is not None:
            return arr.reshape(-1)

    try:
        t = graph.get_tensor(tensor_name)
    except Exception:
        t = None

    if t is not None:
        data = getattr(t, "data", None)
        if data is not None:
            arr = _as_numpy_numeric(data)
            if arr is not None:
                return arr.reshape(-1)

        for attr_name in ("initializer", "value", "values"):
            if hasattr(t, attr_name):
                arr = _as_numpy_numeric(getattr(t, attr_name))
                if arr is not None:
                    return arr.reshape(-1)

    return None


def _resolve_attr_candidate(graph: Graph, v):
    arr = _as_numpy_numeric(v)
    if arr is not None and arr.size > 0:
        return arr.reshape(-1), "direct_numeric_attr"

    if isinstance(v, str):
        arr = _flatten_from_graph_named(graph, v)
        if arr is not None:
            return arr.reshape(-1), f"graph_ref('{v}')"

    return None, None


def _candidate_attr_arrays(graph, op, expected_w: int, expected_b: int):
    attrs = getattr(op, "attrs", {}) or {}
    candidates = []

    likely_weight_keys = [
        "weights",
        "weight",
        "kernel",
        "W",
        "w",
        "weight_data",
        "weights_data",
        "kernel_data",
        "weight_values",
        "weights_values",
        "kernel_values",
        "initializer",
        "params",
    ]
    likely_bias_keys = [
        "bias",
        "biases",
        "B",
        "b",
        "bias_data",
        "bias_values",
    ]

    for k in likely_weight_keys:
        if k in attrs:
            arr, src = _resolve_attr_candidate(graph, attrs[k])
            if arr is not None and arr.size > 0:
                candidates.append(("weight", k, arr, src))

    for k in likely_bias_keys:
        if k in attrs:
            arr, src = _resolve_attr_candidate(graph, attrs[k])
            if arr is not None and arr.size > 0:
                candidates.append(("bias", k, arr, src))

    for k, v in attrs.items():
        arr, src = _resolve_attr_candidate(graph, v)
        if arr is None or arr.size == 0:
            continue
        if expected_w > 0 and arr.size == expected_w:
            candidates.append(("weight", k, arr, src))
        if expected_b > 0 and arr.size == expected_b:
            candidates.append(("bias", k, arr, src))

    return candidates


def _pick_dense_from_attrs(graph, op, expected_w: int, expected_b: int):
    weight_arr = None
    bias_arr = None

    for kind, _key, arr, _src in _candidate_attr_arrays(graph, op, expected_w, expected_b):
        if kind == "weight" and weight_arr is None:
            weight_arr = arr
        elif kind == "bias" and bias_arr is None:
            bias_arr = arr

    return weight_arr, bias_arr


def _dense_expected_shapes(op):
    in_f = int(op.attrs.get("in_features") or 0)
    out_f = int(op.attrs.get("out_features") or 0)
    expected_w = out_f * in_f if in_f > 0 and out_f > 0 else 0
    expected_b = out_f if out_f > 0 else 0
    return in_f, out_f, expected_w, expected_b


def _flatten_training_weights(graph: Graph) -> list[float]:
    vals: list[float] = []

    for op in graph.ops:
        if op.op_type != "Dense":
            continue

        in_f, out_f, expected_w, expected_b = _dense_expected_shapes(op)

        w_arr = None
        b_arr = None

        if len(op.inputs) > 1:
            w_arr = _flatten_from_graph_named(graph, op.inputs[1])
        if len(op.inputs) > 2:
            b_arr = _flatten_from_graph_named(graph, op.inputs[2])

        if w_arr is None or (b_arr is None and expected_b > 0):
            wa, ba = _pick_dense_from_attrs(graph, op, expected_w, expected_b)
            if w_arr is None and wa is not None:
                w_arr = wa
            if b_arr is None and ba is not None:
                b_arr = ba

        if w_arr is None:
            raise RuntimeError(
                f"Training codegen could not resolve Dense weights for {op.name}. "
                f"op.inputs={list(getattr(op, 'inputs', []))}, "
                f"op.attrs keys={sorted(list((getattr(op, 'attrs', {}) or {}).keys()))}"
            )

        if expected_w > 0:
            if w_arr.size != expected_w:
                raise RuntimeError(
                    f"Dense weight size mismatch for {op.name}: got {w_arr.size}, expected {expected_w}"
                )
            vals.extend(w_arr.reshape(-1).astype(np.float32).tolist())
        else:
            vals.extend(np.asarray(w_arr, dtype=np.float32).reshape(-1).tolist())

        if expected_b > 0:
            if b_arr is None:
                vals.extend([0.0] * expected_b)
            else:
                if b_arr.size != expected_b:
                    raise RuntimeError(
                        f"Dense bias size mismatch for {op.name}: got {b_arr.size}, expected {expected_b}"
                    )
                vals.extend(b_arr.reshape(-1).astype(np.float32).tolist())

    return vals


def _infer_weight_words(graph: Graph) -> int:
    total = 0
    for op in graph.ops:
        if op.op_type == "Dense":
            in_f, out_f, expected_w, expected_b = _dense_expected_shapes(op)
            if expected_w > 0:
                total += expected_w
            else:
                w_name = op.attrs.get("weight") or (op.inputs[1] if len(op.inputs) > 1 else None)
                arr = None
                if w_name is not None:
                    arr = _flatten_from_graph_named(graph, w_name)
                if arr is None:
                    wa, _ = _pick_dense_from_attrs(graph, op, expected_w, expected_b)
                    arr = wa
                if arr is not None:
                    total += int(arr.size)

            if expected_b > 0:
                total += expected_b
            else:
                b_name = op.attrs.get("bias") or (op.inputs[2] if len(op.inputs) > 2 else None)
                arr = None
                if b_name is not None:
                    arr = _flatten_from_graph_named(graph, b_name)
                if arr is None:
                    _, ba = _pick_dense_from_attrs(graph, op, expected_w, expected_b)
                    arr = ba
                if arr is not None:
                    total += int(arr.size)

        elif op.op_type == "Conv":
            if len(op.inputs) > 1:
                w_name = op.inputs[1]
                if w_name in getattr(graph, "constants", {}):
                    total += int(np.asarray(graph.constants[w_name]).size)
                else:
                    t = graph.get_tensor(w_name)
                    if t is not None and getattr(t, "shape", None):
                        total += int(np.prod(t.shape))
            if len(op.inputs) > 2:
                b_name = op.inputs[2]
                if b_name in getattr(graph, "constants", {}):
                    total += int(np.asarray(graph.constants[b_name]).size)
                else:
                    t = graph.get_tensor(b_name)
                    if t is not None and getattr(t, "shape", None):
                        total += int(np.prod(t.shape))
    return total


def _infer_in_out_words(graph: Graph) -> Tuple[int, int]:
    in_words = 1
    out_words = 1

    if graph.inputs:
        x = graph.get_tensor(graph.inputs[0])
        if x is not None and getattr(x, "shape", None):
            shape = tuple(int(d) for d in x.shape)
            if len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            in_words = int(np.prod(shape)) if shape else 1

    if graph.outputs:
        y = graph.get_tensor(graph.outputs[0])
        if y is not None and getattr(y, "shape", None):
            shape = tuple(int(d) for d in y.shape)
            if len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            out_words = int(np.prod(shape)) if shape else 1

    return in_words, out_words


def emit_hls_stub(
    graph: Graph,
    out_dir: Path,
    *,
    top_name: str = "deeplearn",
    hls_options: Dict[str, Any] | None = None,
    compile_plan: Any = None,
    memory_plan: Any = None,
    communication_plan: Any = None,
) -> HLSProject:
    hls_options = hls_options or {}

    weights_mode = str(hls_options.get("weights_mode", "embedded")).lower()
    part = str(hls_options.get("part", "xck26-sfvc784-2LV-c"))
    intermediate_dump = bool(hls_options.get("intermediate_dump", False))
    pipeline_mode = str(hls_options.get("pipeline_mode", "inference")).lower()
    training_cfg = dict(hls_options.get("training_cfg", {}) or {})
    raw_cfg = dict(hls_options.get("raw_cfg", {}) or {})

    proj = HLSProject(out_dir=out_dir, top_name=top_name)

    ensure_clean_dir(proj.hls_dir, clean=True)
    ensure_clean_dir(proj.include_layers_dir, clean=True)
    ensure_clean_dir(proj.src_layers_dir, clean=True)
    ensure_clean_dir(proj.include_dir, clean=False)
    ensure_clean_dir(proj.src_dir, clean=False)

    write_text(
        proj.include_dir / "fpgai_types.h",
        emit_types_h(graph, top_name=top_name, raw_cfg=raw_cfg),
    )
    write_text(
        proj.include_dir / "fpgai_params.h",
        emit_params_h_stub(graph, weights_mode=weights_mode),
    )

    write_text(proj.include_layers_dir / "dense.h", emit_dense_h())
    write_text(proj.src_layers_dir / "dense.cpp", emit_dense_cpp())
    write_text(proj.include_layers_dir / "conv.h", emit_conv_h())
    write_text(proj.src_layers_dir / "conv.cpp", emit_conv_cpp())
    write_text(proj.include_layers_dir / "pool.h", emit_pool_h())
    write_text(proj.src_layers_dir / "pool.cpp", emit_pool_cpp())
    write_text(proj.include_layers_dir / "activations.h", emit_activations_h())
    write_text(proj.src_layers_dir / "activations.cpp", emit_activations_cpp())

    write_text(
        proj.src_layers_dir / "model_inst.cpp",
        emit_model_inst_cpp(graph, compile_plan=compile_plan),
    )

    if weights_mode == "embedded":
        write_text(proj.src_dir / "fpgai_params.cpp", emit_params_cpp(graph))
    elif weights_mode in ("stream", "ddr"):
        write_text(proj.include_dir / "weights_runtime.h", emit_weights_runtime_h(graph))
        write_text(proj.src_dir / "weights_runtime.cpp", emit_weights_runtime_cpp(graph))
    else:
        raise RuntimeError(f"Unsupported weights_mode: {weights_mode}")

    if pipeline_mode == "training_on_device":
        write_text(
            proj.src_dir / f"{top_name}.cpp",
            emit_top_train_cpp(
                graph=graph,
                top_name=top_name,
                weights_mode=weights_mode,
                training_cfg=training_cfg,
            ),
        )
    else:
        write_text(
            proj.src_dir / f"{top_name}.cpp",
            emit_top_cpp(
                graph,
                top_name=top_name,
                weights_mode=weights_mode,
                compile_plan=compile_plan,
                memory_plan=memory_plan,
                communication_plan=communication_plan,
            ),
        )

    in_words, out_words = _infer_in_out_words(graph)
    weight_words = _infer_weight_words(graph)

    if pipeline_mode == "training_on_device":
        emit_tb_train_cpp(
            tb_dir=proj.src_dir,
            graph=graph,
            top_name=top_name,
            in_words=in_words,
            out_words=out_words,
            weights_mode=weights_mode,
            weight_words=weight_words,
            preload_weights=_flatten_training_weights(graph),
            training_cfg=training_cfg,
        )
        write_text(
            proj.hls_dir / "run_hls.tcl",
            emit_csim_train_tcl(
                top_name=top_name,
                part=part,
                input_bin_path=str((out_dir / "input.bin").resolve()),
                target_bin_path=str((out_dir / "target.bin").resolve()),
                weights_mode=weights_mode,
                intermediate_dump=intermediate_dump,
            ),
        )
    else:
        emit_tb_cpp(
            tb_dir=proj.src_dir,
            top_name=top_name,
            in_words=in_words,
            out_words=out_words,
            weights_mode=weights_mode,
            weight_words=weight_words,
        )
        write_text(
            proj.hls_dir / "run_hls.tcl",
            emit_csim_tcl(
                top_name=top_name,
                part=part,
                input_bin_path=str((out_dir / "input.bin").resolve()),
                weights_mode=weights_mode,
                intermediate_dump=intermediate_dump,
            ),
        )

    meta = {
        "pipeline_mode": pipeline_mode,
        "weights_mode": weights_mode,
        "top_name": top_name,
        "weight_words": int(weight_words),
        "in_words": int(in_words),
        "out_words": int(out_words),
        "training_cfg": training_cfg,
    }
    write_text(proj.hls_dir / "codegen_meta.json", json.dumps(meta, indent=2))
    return proj