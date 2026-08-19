from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fpgai.config.access import get_path
from fpgai.engine.analysis import analyze_graph
from fpgai.engine.communication import make_communication_plan
from fpgai.engine.layerwise_precision import resolve_layerwise_precision
from fpgai.engine.memory import make_memory_plan
from fpgai.engine.planner import make_compile_plan
from fpgai.engine.training_contracts import _resolve_runtime_sequence
from fpgai.util.fs import ensure_clean_dir


def _tuple_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError("EXPORT010: expected a string or list of strings")


def _external_operator_id(op: Any) -> tuple[str, int]:
    attrs = getattr(op, "attrs", {}) or {}
    provenance = attrs.get("_fpgai_external_operator")
    if isinstance(provenance, Mapping):
        operator_id = str(provenance.get("operator_id", "")).strip()
        version = int(provenance.get("operator_semantics_version", 1) or 1)
        if operator_id:
            return operator_id, version
    builtin = {
        "Dense": "fpgai.operator.dense",
        "Conv": "fpgai.operator.conv2d",
        "Conv2D": "fpgai.operator.conv2d",
    }.get(str(getattr(op, "op_type", "")))
    if builtin:
        return builtin, 1
    return f"fpgai.operator.{str(getattr(op, 'op_type', '')).lower()}", 1


def _resolve_export_contract(
    *,
    raw: Mapping[str, Any],
    op: Any,
    backend: str,
    language: str,
    required: bool = True,
):
    """Resolve one selected FPGAI op to a compatible Ecosystem implementation.

    Resolution reuses normal discovery/selection contracts.  ``required=False`` is
    used by HLS block export so built-in FPGAI HLS lowering remains the fallback for
    built-in operators when no Ecosystem override was selected.  External semantic
    operators never silently fall back to a built-in implementation.
    """
    from fpgai.discovery import DiscoveryRequest, discover_packages
    from fpgai.implementations import (
        CompatibilityRequest,
        ImplementationSelectionRequest,
        implementation_contract_from_manifest,
        select_implementation,
    )

    ecosystem = raw.get("ecosystem", {}) if isinstance(raw.get("ecosystem", {}), Mapping) else {}
    project_root = Path(str(ecosystem.get("project_root", "."))).expanduser().resolve()
    directories = []
    for item in _tuple_strings(ecosystem.get("package_directories", ())):
        candidate = Path(item).expanduser()
        directories.append((project_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    discovery = discover_packages(DiscoveryRequest(
        project_root=project_root,
        configured_directories=tuple(directories),
        include_builtin=bool(ecosystem.get("include_builtin", True)),
        strict=bool(ecosystem.get("strict_discovery", True)),
    ))
    if not discovery.ok:
        raise RuntimeError(f"EXPORT011: ecosystem package discovery failed while resolving the {backend.upper()} implementation")

    impl_root = raw.get("implementations", {}) if isinstance(raw.get("implementations", {}), Mapping) else {}
    enabled = _tuple_strings(impl_root.get("enable", ()))
    contracts = []
    for package_id in enabled:
        entries = [entry for entry in discovery.catalogue.find_by_package_id(package_id) if entry.asset_type == "implementation"]
        if not entries:
            continue
        entry = sorted(entries, key=lambda item: (item.priority, item.version), reverse=True)[0]
        if entry.source_path is None:
            continue
        contracts.append(implementation_contract_from_manifest(entry.source_path, manifest_hash=entry.manifest_hash))

    operator_id, semantics_version = _external_operator_id(op)
    operator_preferences = impl_root.get("operators", {}) if isinstance(impl_root.get("operators", {}), Mapping) else {}
    node_preferences = impl_root.get("nodes", {}) if isinstance(impl_root.get("nodes", {}), Mapping) else {}
    selection_cfg: dict[str, Any] = {}
    candidate = operator_preferences.get(operator_id, operator_preferences.get(str(getattr(op, "op_type", "")), {}))
    if isinstance(candidate, Mapping):
        selection_cfg.update(candidate)
    node_candidate = node_preferences.get(str(getattr(op, "name", "")))
    if isinstance(node_candidate, Mapping):
        selection_cfg.update(node_candidate)

    target_board = str(get_path(raw, "targets.platform.board", get_path(raw, "targets.board", "")) or "")
    precision_value = get_path(raw, "numerics.defaults.activation", "fp32")
    precision = "fp32" if isinstance(precision_value, Mapping) else str(precision_value or "fp32")
    if precision == "float":
        precision = "fp32"
    training = str(get_path(raw, "pipeline.mode", "inference")).strip().lower() == "training_on_device"
    request = ImplementationSelectionRequest(
        operator_id=operator_id,
        compatibility=CompatibilityRequest(
            mode="training" if training else "inference",
            backend=str(backend),
            language=str(language),
            board=target_board or None,
            precision=precision,
            require_backward_input=training,
            operator_semantics_version=semantics_version,
        ),
        preferred_packages=_tuple_strings(selection_cfg.get("preferred", ())),
        allow_fallback=bool(selection_cfg.get("allow_fallback", True)),
        policy=str(selection_cfg.get("policy", impl_root.get("selection_policy", "balanced"))),
    )
    selection = select_implementation(tuple(contracts), request)
    if selection.selected is None:
        if not required:
            return None
        detail_items = list(selection.errors)
        for decision in selection.candidates:
            if decision.reasons:
                detail_items.append(f"{decision.contract.package_id}:" + ",".join(decision.reasons))
        reasons = ", ".join(detail_items) if detail_items else f"no compatible {backend.upper()} implementation"
        raise RuntimeError(
            f"EXPORT012: no compatible {backend.upper()} implementation for operator {operator_id!r} "
            f"(node={getattr(op, 'name', '')!r}); {reasons}"
        )
    return selection.selected


def _tensor_words(graph: Any, tensor_name: str) -> int:
    spec = graph.get_tensor(str(tensor_name))
    if spec is None:
        raise RuntimeError(f"EXPORT015: tensor metadata missing for {tensor_name!r}")
    words = 1
    for dim in tuple(spec.shape):
        value = int(dim)
        if value <= 0:
            raise RuntimeError(f"EXPORT015: export requires static positive tensor shapes; {tensor_name!r} has {spec.shape!r}")
        words *= value
    return words


def _has_external_semantic_provenance(op: Any) -> bool:
    attrs = getattr(op, "attrs", {}) or {}
    return isinstance(attrs.get("_fpgai_external_operator"), Mapping)


def _selection_requested_for_op(raw: Mapping[str, Any], op: Any) -> bool:
    impl_root = raw.get("implementations", {}) if isinstance(raw.get("implementations", {}), Mapping) else {}
    operator_id, _version = _external_operator_id(op)
    operators = impl_root.get("operators", {}) if isinstance(impl_root.get("operators", {}), Mapping) else {}
    nodes = impl_root.get("nodes", {}) if isinstance(impl_root.get("nodes", {}), Mapping) else {}
    return (
        operator_id in operators
        or str(getattr(op, "op_type", "")) in operators
        or str(getattr(op, "name", "")) in nodes
    )


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
    if fmt not in {"hls", "hls_cpp", "vhdl"}:
        raise RuntimeError("EXPORT002: block export format must be hls/hls_cpp or vhdl")
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
    hls_dir = None
    vhdl_result = None
    selected_implementation = None
    if fmt in {"hls", "hls_cpp"}:
        top_name = str(get_path(raw, "pipeline.outputs.top_kernel_name", "deeplearn"))
        use_external_hls = False
        if len(subgraph.ops) == 1:
            op = subgraph.ops[0]
            external_semantics = _has_external_semantic_provenance(op)
            selection_requested = _selection_requested_for_op(raw, op)
            selected_implementation = _resolve_export_contract(
                raw=raw, op=op, backend="vitis_hls", language="hls_cpp",
                required=external_semantics or selection_requested,
            )
            use_external_hls = selected_implementation is not None
        elif any(_has_external_semantic_provenance(op) for op in subgraph.ops):
            raise RuntimeError(
                "EXPORT016: standalone HLS export of a multi-operator subgraph containing Ecosystem semantic operators "
                "requires a composed implementation contract; select one operator or export the compiled mixed graph"
            )

        if use_external_hls:
            from fpgai.implementations.hls_integration import (
                ExternalHLSProjectRequest,
                emit_external_hls_operator_project,
            )
            op = subgraph.ops[0]
            architecture = dict(op.semantics.schedule.get("architecture", {}) or {})
            part = str(get_path(raw, "targets.platform.part", "xck26-sfvc784-2LV-c") or "xck26-sfvc784-2LV-c")
            clock_mhz = float(get_path(raw, "targets.platform.clocks.0.target_mhz", 200.0) or 200.0)
            from fpgai.implementations.hls_integration import HLSFlatArrayABI, HLSTensorPortsABI, parse_hls_abi
            abi = parse_hls_abi(selected_implementation)
            runtime_inputs = [name for name in op.inputs if name not in subgraph.constants]
            runtime_outputs = list(op.outputs)
            if isinstance(abi, HLSFlatArrayABI):
                if len(runtime_inputs) != 1 or len(runtime_outputs) != 1:
                    raise RuntimeError(
                        "EXPORT018: flat_array_v1 standalone HLS export requires exactly one runtime input and one output"
                    )
                input_port_words = {"input": _tensor_words(subgraph, runtime_inputs[0])}
                output_port_words = {"output": _tensor_words(subgraph, runtime_outputs[0])}
            elif isinstance(abi, HLSTensorPortsABI):
                if len(runtime_inputs) != len(abi.inputs) or len(runtime_outputs) != len(abi.outputs):
                    raise RuntimeError(
                        f"EXPORT018: tensor_ports_v1 ABI expects {len(abi.inputs)} input(s)/{len(abi.outputs)} output(s), "
                        f"but FPGAI operator {op.name!r} exposes {len(runtime_inputs)} runtime input(s)/{len(runtime_outputs)} output(s)"
                    )
                input_port_words = {port.name: _tensor_words(subgraph, tensor) for port, tensor in zip(abi.inputs, runtime_inputs)}
                output_port_words = {port.name: _tensor_words(subgraph, tensor) for port, tensor in zip(abi.outputs, runtime_outputs)}
            else:  # pragma: no cover - parse_hls_abi owns the supported ABI set
                raise RuntimeError("EXPORT018: unsupported external HLS ABI")
            hls_result = emit_external_hls_operator_project(ExternalHLSProjectRequest(
                out_dir=export_root,
                contract=selected_implementation,
                operator_name=str(op.op_type),
                operator_attributes=dict(getattr(op, "attrs", {}) or {}),
                input_words=sum(input_port_words.values()),
                output_words=sum(output_port_words.values()),
                input_port_words=input_port_words,
                output_port_words=output_port_words,
                top_name=f"fpgai_export_{str(op.name).replace('/', '_')}",
                part=part,
                clock_period_ns=1000.0 / clock_mhz,
                architecture=architecture,
            ))
            if not hls_result.ok:
                issues = "; ".join(f"{item.code}: {item.message}" for item in hls_result.issues)
                raise RuntimeError(f"EXPORT017: Ecosystem HLS source export failed: {issues}")
            hls_dir = hls_result.hls_dir
        else:
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
    else:
        if len(subgraph.ops) != 1:
            raise RuntimeError(
                "EXPORT013: VHDL block export currently requires exactly one selected FPGAI operator; "
                "multi-operator VHDL composition must be represented by an implementation package or mixed-backend graph contract"
            )
        from fpgai.implementations.vhdl_integration import (
            ExternalVHDLProjectRequest,
            emit_external_vhdl_operator_project,
        )
        selected_implementation = _resolve_export_contract(raw=raw, op=subgraph.ops[0], backend="vhdl", language="vhdl", required=True)
        architecture = dict(subgraph.ops[0].semantics.schedule.get("architecture", {}) or {})
        part = str(get_path(raw, "targets.platform.part", "xck26-sfvc784-2LV-c") or "xck26-sfvc784-2LV-c")
        clock_mhz = float(get_path(raw, "targets.platform.clocks.0.target_mhz", 200.0) or 200.0)
        vhdl_result = emit_external_vhdl_operator_project(ExternalVHDLProjectRequest(
            out_dir=export_root,
            contract=selected_implementation,
            wrapper_top=f"fpgai_export_{str(subgraph.ops[0].name).replace('/', '_')}",
            part=part,
            clock_period_ns=1000.0 / clock_mhz,
            architecture=architecture,
        ))
        if not vhdl_result.ok:
            issues = "; ".join(f"{item.code}: {item.message}" for item in vhdl_result.issues)
            raise RuntimeError(f"EXPORT014: VHDL source export failed: {issues}")

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
        "format": "hls_cpp" if fmt in {"hls", "hls_cpp"} else "vhdl",
        "source_model": str(compiler.cfg.model.path),
        "source_graph": graph.name,
        "selected_ops": list(requested),
        "subgraph_inputs": list(subgraph.inputs),
        "subgraph_outputs": list(subgraph.outputs),
        "hls_dir": None if hls_dir is None else str(hls_dir),
        "vhdl_dir": None if vhdl_result is None or vhdl_result.rtl_dir is None else str(vhdl_result.rtl_dir.parent),
        "selected_implementation": None if selected_implementation is None else selected_implementation.to_dict(),
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
