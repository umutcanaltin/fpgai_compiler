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
from .emit.weights_runtime_h import emit_weights_runtime_h
from .emit.weights_runtime_cpp import emit_weights_runtime_cpp
from .emit.csim_tcl import emit_csim_tcl
from fpgai.backends.hls.testbench import emit_tb_cpp


def _infer_weight_words(graph: Graph) -> int:
    total = 0
    for op in graph.ops:
        if op.op_type == "Conv":
            if len(op.inputs) > 1:
                w_name = op.inputs[1]
                w_t = graph.get_tensor(w_name)
                if w_t is not None and getattr(w_t, "shape", None):
                    n = 1
                    for d in w_t.shape:
                        n *= int(d)
                    total += n
                elif w_name in graph.constants:
                    n = 1
                    for d in graph.constants[w_name].shape:
                        n *= int(d)
                    total += n
            if len(op.inputs) > 2:
                b_name = op.inputs[2]
                b_t = graph.get_tensor(b_name)
                if b_t is not None and getattr(b_t, "shape", None):
                    n = 1
                    for d in b_t.shape:
                        n *= int(d)
                    total += n
                elif b_name in graph.constants:
                    n = 1
                    for d in graph.constants[b_name].shape:
                        n *= int(d)
                    total += n

        elif op.op_type == "Dense":
            in_f = int(op.attrs.get("in_features") or 0)
            out_f = int(op.attrs.get("out_features") or 0)
            if in_f > 0 and out_f > 0:
                total += in_f * out_f
                total += out_f

    return total


def _infer_in_out_words(graph: Graph) -> Tuple[int, int]:
    in_words = 1
    out_words = 1

    if graph.inputs:
        x = graph.get_tensor(graph.inputs[0])
        if x and x.shape:
            in_words = int(np.prod(x.shape))

    if graph.outputs:
        y = graph.get_tensor(graph.outputs[0])
        if y and y.shape:
            out_words = int(np.prod(y.shape))

    return in_words, out_words


def _plan_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"repr": repr(obj)}


def _extract_codegen_numerics(hls_options: Dict[str, Any], compile_plan: Any) -> Dict[str, Any]:
    numerics = hls_options.get("numerics", {}) if isinstance(hls_options, dict) else {}
    if isinstance(numerics, dict) and numerics:
        return numerics

    if compile_plan is None:
        return {}

    cp = compile_plan.to_dict() if hasattr(compile_plan, "to_dict") else compile_plan
    notes = cp.get("notes", {}) if isinstance(cp, dict) else {}

    act_bits = notes.get("act_bits")
    act_int_bits = notes.get("act_int_bits")
    weight_bits = notes.get("weight_bits")
    weight_int_bits = notes.get("weight_int_bits")
    bias_bits = notes.get("bias_bits")
    bias_int_bits = notes.get("bias_int_bits")
    accum_bits = notes.get("accum_bits")
    accum_int_bits = notes.get("accum_int_bits")
    precision_mode = str(notes.get("precision_mode", "fixed")).lower()

    if precision_mode == "float":
        return {
            "activation": {"type": "float"},
            "weight": {"type": "float"},
            "bias": {"type": "float"},
            "accum": {"type": "float"},
        }

    if act_bits is not None and weight_bits is not None:
        return {
            "activation": {"type": "ap_fixed", "total_bits": int(act_bits), "int_bits": int(act_int_bits if act_int_bits is not None else 6)},
            "weight": {"type": "ap_fixed", "total_bits": int(weight_bits), "int_bits": int(weight_int_bits if weight_int_bits is not None else 6)},
            "bias": {"type": "ap_fixed", "total_bits": int(bias_bits if bias_bits is not None else 24), "int_bits": int(bias_int_bits if bias_int_bits is not None else 10)},
            "accum": {"type": "ap_fixed", "total_bits": int(accum_bits if accum_bits is not None else 24), "int_bits": int(accum_int_bits if accum_int_bits is not None else 10)},
        }

    return {}


def _extract_codegen_parallel(hls_options: Dict[str, Any], compile_plan: Any) -> Dict[str, Any]:
    par = hls_options.get("parallel", {}) if isinstance(hls_options, dict) else {}
    if isinstance(par, dict) and par:
        return par

    if compile_plan is None:
        return {}

    cp = compile_plan.to_dict() if hasattr(compile_plan, "to_dict") else compile_plan
    notes = cp.get("notes", {}) if isinstance(cp, dict) else {}

    style = str(notes.get("parallel_pipeline_style", "balanced"))
    unroll = int(notes.get("parallel_unroll_factor", 1) or 1)

    if style == "conservative":
        return {"dense_in_unroll": 1, "conv_ic_unroll": 1}
    if style == "aggressive":
        return {"dense_in_unroll": max(1, unroll), "conv_ic_unroll": max(1, min(4, unroll))}
    return {"dense_in_unroll": max(1, min(2, unroll)), "conv_ic_unroll": max(1, min(2, unroll))}


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
    clk_mhz = float(hls_options.get("clk_mhz", 200))
    intermediate_dump = bool(hls_options.get("intermediate_dump", False))
    numerics = _extract_codegen_numerics(hls_options, compile_plan)
    parallel = _extract_codegen_parallel(hls_options, compile_plan)

    proj = HLSProject(out_dir=out_dir, top_name=top_name)

    ensure_clean_dir(proj.hls_dir, clean=True)
    ensure_clean_dir(proj.include_layers_dir, clean=True)
    ensure_clean_dir(proj.src_layers_dir, clean=True)
    ensure_clean_dir(proj.include_dir, clean=False)
    ensure_clean_dir(proj.src_dir, clean=False)

    write_text(
        proj.include_dir / "fpgai_types.h",
        emit_types_h(graph, top_name=top_name, numerics=numerics, parallel=parallel),
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
    elif weights_mode == "stream":
        write_text(proj.include_dir / "weights_runtime.h", emit_weights_runtime_h(graph))
        write_text(proj.src_dir / "weights_runtime.cpp", emit_weights_runtime_cpp(graph))
    elif weights_mode == "ddr":
        write_text(proj.include_dir / "weights_runtime.h", emit_weights_runtime_h(graph))
        write_text(proj.src_dir / "weights_runtime.cpp", emit_weights_runtime_cpp(graph))
    else:
        raise ValueError(f"Unsupported weights_mode: {weights_mode}")

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
    emit_tb_cpp(
        top_name=top_name,
        in_words=in_words,
        out_words=out_words,
        tb_dir=proj.src_dir,
        weights_mode=weights_mode,
        weight_words=weight_words,
    )

    input_bin_path = (out_dir / "input.bin").resolve()
    write_text(
        proj.hls_dir / "run_hls.tcl",
        emit_csim_tcl(
            top_name=top_name,
            part=part,
            input_bin_path=str(input_bin_path),
            weights_mode=weights_mode,
            intermediate_dump=intermediate_dump,
        ),
    )

    return proj