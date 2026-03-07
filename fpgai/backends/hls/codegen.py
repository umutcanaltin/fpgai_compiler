from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np  # <--- Added correct import

from fpgai.ir.graph import Graph
from fpgai.util.fs import ensure_clean_dir, write_text
from .project import HLSProject

# IMPORT EMITTERS
from .emit.types_h import emit_types_h
from .emit.params_h import emit_params_h_stub
from .emit.params_cpp import emit_params_cpp
from .emit.layers_dense import emit_dense_h, emit_dense_cpp
from .emit.layers_conv import emit_conv_h, emit_conv_cpp
from .emit.layers_pool import emit_pool_h, emit_pool_cpp
from .emit.layers_activations import emit_activations_h, emit_activations_cpp
from .emit.model_inst_cpp import emit_model_inst_cpp
from .emit.top_cpp import emit_top_cpp
from .emit.readme_txt import emit_readme_txt
from .emit.weights_runtime_h import emit_weights_runtime_h
from .emit.weights_runtime_cpp import emit_weights_runtime_cpp
from .emit.csim_tcl import emit_csim_tcl
from fpgai.backends.hls.testbench import emit_tb_cpp


def _safe_last_dim(shape: Optional[Tuple[int, ...]]) -> Optional[int]:
    if not shape: return None
    d = shape[-1]
    return int(d) if d else None


def _infer_in_out_words(graph: Graph) -> Tuple[int, int]:
    in_words = None
    out_words = None

    # Calculate input size (Flattened)
    if graph.inputs:
        x = graph.get_tensor(graph.inputs[0])
        if x and x.shape: 
            in_words = int(np.prod(x.shape)) 
            
    # Calculate output size (Flattened)
    if graph.outputs:
        y = graph.get_tensor(graph.outputs[0])
        if y and y.shape: 
            out_words = int(np.prod(y.shape))

    # Fallback defaults
    if not in_words: in_words = 1
    if not out_words: out_words = 1
    
    return in_words, out_words


def emit_hls_stub(
    graph: Graph, 
    out_dir: Path, 
    *, 
    top_name: str = "deeplearn", 
    hls_options: Dict[str, Any] | None = None
) -> HLSProject:
    hls_options = hls_options or {}
    weights_mode = str(hls_options.get("weights_mode", "embedded")).lower()
    part = str(hls_options.get("part", "xck26-sfvc784-2LV-c"))
    clk_mhz = float(hls_options.get("clk_mhz", 200))
    proj = HLSProject(out_dir=out_dir, top_name=top_name)

    # Clean directories
    ensure_clean_dir(proj.hls_dir, clean=True)
    ensure_clean_dir(proj.include_layers_dir, clean=True)
    ensure_clean_dir(proj.src_layers_dir, clean=True)
    ensure_clean_dir(proj.include_dir, clean=False)
    ensure_clean_dir(proj.src_dir, clean=False)

    # 1. Core headers
    write_text(proj.include_dir / "fpgai_types.h", emit_types_h(graph, top_name=top_name))
    write_text(proj.include_dir / "fpgai_params.h", emit_params_h_stub(graph))

    # 2. Layers (Dense, Conv, Pool, Activations)
    write_text(proj.include_layers_dir / "dense.h", emit_dense_h())
    write_text(proj.src_layers_dir / "dense.cpp", emit_dense_cpp())
    
    write_text(proj.include_layers_dir / "conv.h", emit_conv_h())
    write_text(proj.src_layers_dir / "conv.cpp", emit_conv_cpp())
    
    write_text(proj.include_layers_dir / "pool.h", emit_pool_h())
    write_text(proj.src_layers_dir / "pool.cpp", emit_pool_cpp())
    
    write_text(proj.include_layers_dir / "activations.h", emit_activations_h())
    write_text(proj.src_layers_dir / "activations.cpp", emit_activations_cpp())

    # 3. Model Instantiation
    write_text(proj.src_layers_dir / "model_inst.cpp", emit_model_inst_cpp(graph))

    # 4. Weights
    if weights_mode == "embedded":
        write_text(proj.src_dir / "fpgai_params.cpp", emit_params_cpp(graph))
    elif weights_mode == "stream":
        write_text(proj.include_dir / "weights_runtime.h", emit_weights_runtime_h(graph))
        write_text(proj.src_dir / "weights_runtime.cpp", emit_weights_runtime_cpp(graph))

    # 5. Top Function (Wires layers together)
    write_text(proj.src_dir / f"{top_name}.cpp", emit_top_cpp(graph, top_name=top_name, weights_mode=weights_mode))

    # 6. Testbench
    in_words, out_words = _infer_in_out_words(graph)
    emit_tb_cpp(top_name=top_name, in_words=in_words, out_words=out_words, tb_dir=proj.src_dir, weights_mode=weights_mode)

    # 7. Tcl Script
    input_bin_path = (out_dir / "input.bin").resolve()
    write_text(proj.hls_dir / "run_hls.tcl", emit_csim_tcl(top_name=top_name, part=part, input_bin_path=str(input_bin_path)))

    return proj