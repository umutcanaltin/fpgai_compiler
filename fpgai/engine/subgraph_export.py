from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fpgai.config.access import get_path
from fpgai.engine.analysis import analyze_graph
from fpgai.engine.communication import make_communication_plan
from fpgai.engine.layerwise_precision import resolve_layerwise_precision
from fpgai.engine.memory import make_memory_plan
from fpgai.engine.planner import make_compile_plan
from fpgai.engine.training_contracts import _resolve_runtime_sequence
from fpgai.util.fs import ensure_clean_dir


def export_compiler_subgraph(
    compiler: Any,
    *,
    op_names,
    out_dir: str | Path,
    artifact_format: str = "hls",
) -> Path:
    """Export an IR-owned operator/subgraph through the normal backend path.

    This is compiler orchestration, not a standalone slicer: ``Graph.extract_subgraph``
    owns graph semantics while the existing planner/memory/communication/backend
    owners resolve the exported hardware. No HLS/Vivado/bitstream tool execution is
    requested by this source-export path.
    """
    fmt = str(artifact_format).strip().lower()
    if fmt not in {"hls", "hls_cpp"}:
        raise RuntimeError(
            "EXPORT002: compiler-generated subgraph export currently supports HLS source; "
            "standalone ecosystem VHDL/RTL implementations use `fpgai ecosystem export` so their declared RTL is exported without Vivado/bitstream execution"
        )
    requested = tuple(str(name) for name in op_names)
    if not requested:
        raise ValueError("EXPORT001: at least one --op selector is required")

    raw = compiler.cfg.raw
    export_root = Path(out_dir).expanduser().resolve()
    ensure_clean_dir(export_root, clean=True)
    act_kind, act_alpha, act_except_last = compiler._read_activation_insert_cfg(raw)
    if str(compiler.cfg.pipeline.mode).lower() == "training_on_device":
        compiler._reject_unsupported_training_weight_storage(raw)
    graph = compiler._import_and_prepare_graph(
        act_kind=act_kind, act_alpha=act_alpha, act_except_last=act_except_last
    )
    resolve_layerwise_precision(graph, raw)
    subgraph = graph.extract_subgraph(requested, name=f"{graph.name}_export")

    descriptors = analyze_graph(subgraph)
    compile_plan = make_compile_plan(compiler.cfg, descriptors)
    memory_plan = make_memory_plan(subgraph, descriptors, compile_plan)
    compiler._annotate_memory_movement_semantics(compile_plan, memory_plan, raw)
    communication_plan = make_communication_plan(compiler.cfg, memory_plan)
    weights_mode = compiler._resolve_hls_weights_mode(raw)
    runtime_sequence = _resolve_runtime_sequence(
        raw,
        pipeline_mode=str(compiler.cfg.pipeline.mode),
        memory_semantics_mode=str(memory_plan.notes.get("memory_semantics_mode", weights_mode)),
    )
    compiler._emit_ir_artifacts(
        export_root, subgraph, descriptors, compile_plan, memory_plan, communication_plan,
        runtime_sequence=runtime_sequence, training_plan=None,
    )
    top_name = str(get_path(raw, "pipeline.outputs.top_kernel_name", "deeplearn"))
    hls_dir = compiler._emit_hls(
        export_root,
        subgraph,
        top_name=top_name,
        weights_mode=weights_mode,
        compile_plan=compile_plan,
        memory_plan=memory_plan,
        communication_plan=communication_plan,
        build_stages={
            "cpp": True,
            "host_cpp": False,
            "testbench": True,
            "hls_project": True,
            "hls_synthesis": False,
            "vivado_project": False,
            "vivado_implementation": False,
            "bitstream": False,
            "runtime_package": False,
            "reports": True,
        },
    )

    block_reference = {
        "status": "unavailable",
        "reason": "functional FPGAI IR reference execution was not available for the selected source graph",
    }
    try:
        from fpgai.benchmark.graph_reference import deterministic_graph_inputs, execute_graph_reference_trace

        reference_inputs = deterministic_graph_inputs(graph)
        trace = execute_graph_reference_trace(graph, reference_inputs)
        ref_root = export_root / "validation" / "reference"
        ref_root.mkdir(parents=True, exist_ok=True)
        input_artifacts = {}
        output_artifacts = {}
        for tensor in subgraph.inputs:
            if tensor not in trace:
                raise RuntimeError(f"reference trace does not contain exported boundary input {tensor!r}")
            arr = np.asarray(trace[tensor], dtype=np.float32)
            safe = tensor.replace("/", "_")
            npy = ref_root / f"input_{safe}.npy"
            bin_path = ref_root / f"input_{safe}.bin"
            np.save(npy, arr)
            arr.reshape(-1).tofile(bin_path)
            input_artifacts[tensor] = {"shape": list(arr.shape), "npy": str(npy), "bin": str(bin_path)}
        for tensor in subgraph.outputs:
            if tensor not in trace:
                raise RuntimeError(f"reference trace does not contain exported boundary output {tensor!r}")
            arr = np.asarray(trace[tensor], dtype=np.float32)
            safe = tensor.replace("/", "_")
            npy = ref_root / f"output_{safe}.npy"
            bin_path = ref_root / f"output_{safe}.bin"
            np.save(npy, arr)
            arr.reshape(-1).tofile(bin_path)
            output_artifacts[tensor] = {"shape": list(arr.shape), "npy": str(npy), "bin": str(bin_path)}
        block_reference = {
            "status": "reference_captured",
            "reference_domain": "functional_fpgai_ir",
            "inputs": input_artifacts,
            "outputs": output_artifacts,
            "generated_comparison": "pending_until_csim_or_runtime",
        }
    except Exception as exc:
        block_reference = {"status": "unavailable", "reason": str(exc)}

    manifest = {
        "schema": "fpgai.subgraph-export/v1",
        "status": "exported",
        "pipeline_mode": str(compiler.cfg.pipeline.mode),
        "format": "hls_cpp",
        "source_model": str(compiler.cfg.model.path),
        "source_graph": graph.name,
        "selected_ops": list(requested),
        "subgraph_inputs": list(subgraph.inputs),
        "subgraph_outputs": list(subgraph.outputs),
        "hls_dir": str(hls_dir),
        "resolved_ir": str(export_root / "ir" / "resolved_ir.json"),
        "tool_execution": {"vitis_hls": False, "vivado": False, "bitstream": False},
        "numeric_validation": {
            "status": "reference_captured" if block_reference.get("status") == "reference_captured" else "reference_unavailable",
            "reference": block_reference,
            "claim_allowed": False,
            "policy": (
                "Standalone export captures functional FPGAI-IR boundary references without synthesis. "
                "Numeric correctness is claimed only after generated CSim/RTL/runtime output is compared with these references."
            ),
        },
        "usage": {"platform_scope": "research", "production_path": "morfics"},
    }
    (export_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return export_root
