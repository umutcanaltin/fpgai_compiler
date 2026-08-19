from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fpgai.config.access import get_path
from fpgai.contracts import PackageLock, write_package_lock
from fpgai.contracts.package_lock import LockedPackage
from fpgai.discovery import DiscoveryRequest, discover_packages
from fpgai.discovery.discovery_report import write_discovery_report
from fpgai.implementations import (
    CompatibilityRequest,
    ImplementationSelectionRequest,
    implementation_contract_from_manifest,
    select_implementation,
)
from fpgai.implementations.selection_reports import write_implementation_selection_report
from fpgai.operators.external import OperatorLoadRequest, load_operator_packages
from fpgai.implementations.hls_composition import build_hls_composition_plan, write_composition_report
from fpgai.backends.hls.codegen import emit_hls_stub


@dataclass(frozen=True)
class ExternalEcosystemCompileResult:
    handled: bool
    graph: Any | None = None
    hls_dir: Path | None = None
    hls_run: Any | None = None
    artifacts: Mapping[str, Any] | None = None


def _tuple_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError("Expected a string or list of strings")


def _shape_words(graph: Any, tensor_name: str) -> int:
    spec = graph.get_tensor(tensor_name)
    if spec is None or not getattr(spec, "shape", None):
        raise RuntimeError(f"Missing static tensor shape for {tensor_name!r}")
    total = 1
    for dim in spec.shape:
        value = int(dim)
        if value <= 0:
            raise RuntimeError(f"Dynamic or invalid tensor dimension for {tensor_name!r}: {dim!r}")
        total *= value
    return total


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _selected_entries(catalogue: Any, package_ids: tuple[str, ...]) -> list[Any]:
    entries = []
    for package_id in package_ids:
        candidates = catalogue.find_by_package_id(package_id)
        if not candidates:
            raise RuntimeError(f"Configured package {package_id!r} was not discovered")
        entries.append(sorted(candidates, key=lambda item: (item.priority, item.version), reverse=True)[0])
    return entries


def compile_external_hls_if_configured(compiler: Any, *, out_dir: Path, build_stages: Mapping[str, bool]) -> ExternalEcosystemCompileResult:
    raw = compiler.cfg.raw
    ecosystem = get_path(raw, "ecosystem", None)
    if not isinstance(ecosystem, Mapping) or not bool(ecosystem.get("enabled", False)):
        return ExternalEcosystemCompileResult(False)

    if str(compiler.cfg.pipeline.mode).lower() != "inference":
        raise RuntimeError("External HLS ecosystem compilation currently supports inference mode only")

    started = time.time()
    project_root = Path(str(ecosystem.get("project_root", "."))).expanduser().resolve()
    configured_dirs = tuple(
        (project_root / item).resolve() if not Path(item).expanduser().is_absolute() else Path(item).expanduser().resolve()
        for item in _tuple_strings(ecosystem.get("package_directories", ()))
    )
    discovery = discover_packages(DiscoveryRequest(
        project_root=project_root,
        configured_directories=configured_dirs,
        include_builtin=bool(ecosystem.get("include_builtin", True)),
        strict=bool(ecosystem.get("strict_discovery", True)),
    ))
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    discovery_json, discovery_md = write_discovery_report(discovery, reports_dir / "package_discovery")
    discovery_paths = {"json": discovery_json, "markdown": discovery_md}
    if not discovery.ok:
        raise RuntimeError("External package discovery failed; inspect package_discovery report")

    operator_cfg = ecosystem.get("operator_packages", {}) or {}
    if not isinstance(operator_cfg, Mapping):
        raise RuntimeError("ecosystem.operator_packages must be a mapping")
    operator_ids = _tuple_strings(operator_cfg.get("enable", ()))
    trust_cfg = ecosystem.get("trust", {}) or {}
    if not isinstance(trust_cfg, Mapping):
        raise RuntimeError("ecosystem.trust must be a mapping")
    for package_id in operator_ids:
        if trust_cfg.get(package_id) != "approved_for_reference":
            raise RuntimeError(f"Operator package {package_id!r} requires approved_for_reference trust")

    loaded = load_operator_packages(OperatorLoadRequest(
        catalogue=discovery.catalogue,
        package_ids=operator_ids,
        trust_level="approved_for_reference",
    ))
    loading_path = _write_json(reports_dir / "external_operator_loading.json", loaded.to_dict())
    if not loaded.ok:
        raise RuntimeError("External operator loading failed; inspect external_operator_loading.json")

    from fpgai.frontend import import_model_source
    from fpgai.layers.composites import expand_composite_layers

    graph = import_model_source(
        compiler.cfg.model.path,
        format_hint=getattr(compiler.cfg.model, "format", None),
        source_framework=getattr(compiler.cfg.model, "framework", None),
        pipeline_mode=getattr(compiler.cfg.pipeline, "mode", "inference"),
        target_board=((raw.get("targets", {}).get("platform", {}) or {}).get("board") or (raw.get("targets", {}) or {}).get("board")),
        external_operator_context=loaded.context,
    )
    graph = expand_composite_layers(graph)
    external_ops = [op for op in graph.ops if isinstance(op.attrs.get("_fpgai_external_operator"), Mapping)]
    if not external_ops:
        raise RuntimeError("Ecosystem compilation was enabled but the model contains no activated external operator")

    from fpgai.ir.liveness import analyze_tensor_liveness, write_tensor_liveness_report
    tensor_liveness = analyze_tensor_liveness(graph)
    liveness_json, liveness_md = write_tensor_liveness_report(tensor_liveness, reports_dir)

    from fpgai.backends.hls.buffer_allocation import (
        build_hls_buffer_allocation,
        build_legacy_buffer_provenance,
        write_hls_buffer_allocation_report,
    )
    if bool(tensor_liveness.get("has_branching", False)):
        hls_buffer_allocation = build_hls_buffer_allocation(
            graph, raw_cfg=raw, tensor_liveness=tensor_liveness
        )
        generated_buffer_provenance = dict(hls_buffer_allocation.get("resource_provenance", {}))
    else:
        generated_buffer_provenance = build_legacy_buffer_provenance(graph)
        hls_buffer_allocation = {
            "schema": "fpgai.hls-buffer-allocation/v1",
            "mode": "legacy_sequential",
            "graph_name": str(getattr(graph, "name", "main")),
            "slot_count": len(generated_buffer_provenance),
            "slots": [
                {"slot": index, "name": name, "cpp_type": None, "words": None, "tensors": list(entry.get("tensors", []))}
                for index, (name, entry) in enumerate(generated_buffer_provenance.items())
            ],
            "tensor_to_buffer": {
                tensor: name
                for name, entry in generated_buffer_provenance.items()
                for tensor in entry.get("tensors", [])
            },
            "resource_provenance": generated_buffer_provenance,
            "policy": "Historical sequential buffer naming is preserved for non-branching graphs.",
        }
    buffer_json, buffer_md = write_hls_buffer_allocation_report(hls_buffer_allocation, reports_dir)
    impl_root = get_path(raw, "implementations", {}) or {}
    if not isinstance(impl_root, Mapping):
        raise RuntimeError("implementations must be a mapping")
    enabled_impl_ids = _tuple_strings(impl_root.get("enable", ()))
    contracts = []
    for entry in _selected_entries(discovery.catalogue, enabled_impl_ids):
        if entry.asset_type != "implementation" or entry.source_path is None:
            raise RuntimeError(f"Package {entry.package_id!r} is not a loadable implementation package")
        contracts.append(implementation_contract_from_manifest(entry.source_path, manifest_hash=entry.manifest_hash))

    target_board = str(get_path(raw, "targets.platform.board", get_path(raw, "targets.board", "")) or "")
    precision_value = get_path(raw, "numerics.defaults.activation", "fp32")
    precision = "fp32" if isinstance(precision_value, Mapping) else str(precision_value or "fp32")
    if precision == "float":
        precision = "fp32"
    operator_preferences = impl_root.get("operators", {}) or {}
    node_preferences = impl_root.get("nodes", {}) or {}
    selected_contracts = {}
    node_selection_paths = {}

    for op in external_ops:
        provenance = dict(op.attrs["_fpgai_external_operator"])
        operator_id = str(provenance.get("operator_id", ""))
        selection_cfg = {}
        if isinstance(operator_preferences, Mapping):
            selection_cfg = operator_preferences.get(operator_id, operator_preferences.get(op.op_type, {})) or {}
        if isinstance(node_preferences, Mapping) and isinstance(node_preferences.get(op.name), Mapping):
            selection_cfg = {**dict(selection_cfg), **dict(node_preferences[op.name])}
        if not isinstance(selection_cfg, Mapping):
            raise RuntimeError("Implementation selection must be a mapping")
        compatibility = CompatibilityRequest(
            mode="inference",
            backend=str(selection_cfg.get("backend", "vitis_hls")),
            language="hls_cpp",
            board=target_board or None,
            precision=precision,
        )
        request = ImplementationSelectionRequest(
            operator_id=operator_id,
            compatibility=compatibility,
            preferred_packages=_tuple_strings(selection_cfg.get("preferred", ())),
            allow_fallback=bool(selection_cfg.get("allow_fallback", True)),
            policy=str(selection_cfg.get("policy", impl_root.get("selection_policy", "balanced"))),
        )
        selection = select_implementation(tuple(contracts), request)
        node_report_dir = reports_dir / "implementation_selection" / op.name
        selection_json, selection_md = write_implementation_selection_report(selection, node_report_dir)
        node_selection_paths[op.name] = {"json": str(selection_json), "markdown": str(selection_md)}
        if selection.selected is None:
            raise RuntimeError(f"No compatible external HLS implementation for node {op.name!r}")
        selected_contracts[op.name] = selection.selected

    composition_plan = build_hls_composition_plan(
        graph,
        selected_contracts=selected_contracts,
        selection_reports=node_selection_paths,
    )
    composition_json, composition_md = write_composition_report(composition_plan, reports_dir)

    part = str(get_path(raw, "targets.platform.part", "xck26-sfvc784-2LV-c") or "xck26-sfvc784-2LV-c")
    clock_mhz = float(get_path(raw, "targets.platform.clocks.0.target_mhz", 200.0) or 200.0)
    if not math.isfinite(clock_mhz) or clock_mhz <= 0:
        raise RuntimeError("Target clock must be a positive finite value")
    project = emit_hls_stub(
        graph=graph,
        out_dir=out_dir,
        top_name=str(get_path(raw, "pipeline.outputs.top_kernel_name", "deeplearn")),
        hls_options={
            "pipeline_mode": "inference",
            "weights_mode": str(get_path(raw, "memory.weights.mode", "embedded") or "embedded"),
            "part": part,
            "clk_mhz": int(clock_mhz),
            "run_csim": bool(build_stages.get("hls_project", True)),
            "run_csynth": bool(build_stages.get("hls_synthesis", False)),
            "export_ip": False,
            "raw_cfg": raw,
        },
        external_composition_plan=composition_plan,
    )

    validation_cfg = ecosystem.get("validation", {}) or {}
    validation_artifacts = None
    if isinstance(validation_cfg, Mapping) and bool(validation_cfg.get("enabled", False)):
        from fpgai.backends.hls.testbench import emit_tb_cpp
        from fpgai.validation.mixed_external_hls import prepare_mixed_external_validation

        validation_artifacts = prepare_mixed_external_validation(
            graph=graph, external_context=loaded.context, out_dir=out_dir, config=validation_cfg
        )
        in_words = _shape_words(graph, graph.inputs[0])
        out_words = _shape_words(graph, graph.outputs[0])
        emit_tb_cpp(
            project.hls_dir / "src",
            top_name=str(get_path(raw, "pipeline.outputs.top_kernel_name", "deeplearn")),
            in_words=in_words,
            out_words=out_words,
            weights_mode=str(get_path(raw, "memory.weights.mode", "embedded") or "embedded"),
            raw_cfg=raw,
        )
        tcl_text = project.run_tcl.read_text(encoding="utf-8")
        csim_args = f'csim_design -argv "{validation_artifacts.input_bin.resolve()} {validation_artifacts.output_bin.resolve()}"'
        tcl_text = tcl_text.replace("csim_design\n", csim_args + "\n")
        project.run_tcl.write_text(tcl_text, encoding="utf-8")
        from fpgai.validation.mixed_external_hls import run_portable_host_cpp_validation
        host_validation = run_portable_host_cpp_validation(
            graph=graph, composition_plan=composition_plan, artifacts=validation_artifacts, hls_dir=project.hls_dir
        )
        prepared_payload = json.loads(validation_artifacts.report_path.read_text(encoding="utf-8"))
        prepared_payload["host_cpp"] = host_validation
        validation_artifacts.report_path.write_text(
            json.dumps(prepared_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    selected_ids = tuple(dict.fromkeys((*operator_ids, *composition_plan.used_package_ids)))
    lock_entries = []
    for entry in _selected_entries(discovery.catalogue, selected_ids):
        lock_entries.append(LockedPackage(
            package_id=entry.package_id,
            version=entry.version,
            source=entry.source.value,
            manifest_sha256=entry.manifest_hash,
        ))
    lock_path = write_package_lock(PackageLock(tuple(lock_entries)), out_dir / "package-lock.yml")

    hls_run = None
    run_validation_csim = bool(validation_artifacts is not None and validation_cfg.get("run_vitis_csim", False))
    if bool(build_stages.get("hls_synthesis", False)) or run_validation_csim:
        hls_run = compiler._maybe_run_vitis_hls(
            project.hls_dir,
            build_stages={**dict(build_stages), "hls_synthesis": True},
        )
    validation_report = None
    if validation_artifacts is not None:
        from fpgai.validation.mixed_external_hls import finalize_mixed_external_validation
        validation_report = finalize_mixed_external_validation(
            validation_artifacts,
            hls_run=hls_run,
            atol=float(validation_cfg.get("atol", 1e-5)),
            rtol=float(validation_cfg.get("rtol", 1e-5)),
        )

    synthesis_characterization = None
    synthesis_characterization_paths = None
    if hls_run is not None and bool(getattr(hls_run, "csynth_ran", False)):
        from fpgai.analysis.hls_synthesis_characterization import (
            characterize_hls_synthesis,
            write_hls_synthesis_characterization,
        )
        declared_metrics = {
            node_name: contract.metrics.to_dict()
            for node_name, contract in selected_contracts.items()
        }
        synthesis_characterization = characterize_hls_synthesis(
            csynth_report_path=getattr(hls_run, "csynth_report", None),
            target_clock_mhz=clock_mhz,
            top_name=str(get_path(raw, "pipeline.outputs.top_kernel_name", "deeplearn")),
            participating_external_packages=tuple(dict.fromkeys(contract.package_id for contract in selected_contracts.values())),
            declared_implementation_metrics=declared_metrics,
            scope="mixed_graph_top",
        )
        synth_json, synth_md = write_hls_synthesis_characterization(
            synthesis_characterization, reports_dir
        )
        synthesis_characterization_paths = {
            "json": str(synth_json),
            "markdown": str(synth_md),
        }

    from fpgai.analysis.hls_bottleneck_diagnostics import analyze_hls_bottlenecks, write_hls_bottleneck_diagnostics
    bottleneck_diagnostics = analyze_hls_bottlenecks(
        None if hls_run is None else getattr(hls_run, "stdout_log", None),
        tensor_liveness=tensor_liveness,
        resource_provenance=generated_buffer_provenance,
    )
    bottleneck_json, bottleneck_md = write_hls_bottleneck_diagnostics(bottleneck_diagnostics, reports_dir)

    external_operator_records = [
        {"name": op.name, "op_type": op.op_type, **dict(op.attrs["_fpgai_external_operator"])}
        for op in external_ops
    ]
    selected_implementation_records = {
        name: contract.to_dict() for name, contract in selected_contracts.items()
    }

    external_ecosystem_manifest = {
        "operators": external_operator_records,
        "selected_implementations": selected_implementation_records,
        "package_lock": str(lock_path),
        "discovery_reports": {key: str(value) for key, value in discovery_paths.items()},
        "operator_loading_report": str(loading_path),
        "selection_reports": node_selection_paths,
        "composition_reports": {"json": str(composition_json), "markdown": str(composition_md)},
        "tensor_liveness": {"json": str(liveness_json), "markdown": str(liveness_md), "summary": {
            "activation_buffer_slots": tensor_liveness["activation_buffer_slots"],
            "maximum_simultaneously_live_tensors": tensor_liveness["maximum_simultaneously_live_tensors"],
            "has_branching": tensor_liveness["has_branching"],
            "sequential_current_buffer_compatible": tensor_liveness["sequential_current_buffer_compatible"],
        }},
        "hls_buffer_allocation": {
            "json": str(buffer_json),
            "markdown": str(buffer_md),
            "mode": hls_buffer_allocation.get("mode"),
            "slot_count": hls_buffer_allocation.get("slot_count"),
            "tensor_to_buffer": hls_buffer_allocation.get("tensor_to_buffer", {}),
        },
        "hls_bottleneck_diagnostics": {"json": str(bottleneck_json), "markdown": str(bottleneck_md), "summary": {
            "warning_count": bottleneck_diagnostics.get("warning_count", 0),
            "ii_violation_count": bottleneck_diagnostics.get("ii_violation_count", 0),
            "categories": bottleneck_diagnostics.get("categories", []),
        }},
        "hls_composition": composition_plan.to_dict(),
        "hls_project": {"hls_dir": str(project.hls_dir), "top_cpp": str(project.top_cpp), "run_tcl": str(project.run_tcl)},
        "validation": None if validation_artifacts is None else {
            "report": str(validation_artifacts.report_path),
            "input_bin": str(validation_artifacts.input_bin),
            "reference_output_bin": str(validation_artifacts.expected_bin),
            "hls_output_bin": str(validation_artifacts.output_bin),
            "status": validation_report.get("status") if validation_report else "prepared",
        },
        "synthesis_characterization": None if synthesis_characterization is None else {
            "status": synthesis_characterization.status,
            "validation_level": "hls_synthesized" if synthesis_characterization.status == "passed" else "unavailable",
            "reports": synthesis_characterization_paths,
            "scope": synthesis_characterization.scope,
            "target_met": synthesis_characterization.target_met,
        },
    }
    # Preserve the E4B single-operator manifest API for existing users while
    # exposing the E4C plural node-level representation for mixed graphs.
    if len(external_operator_records) == 1:
        only_node = external_operator_records[0]["name"]
        external_ecosystem_manifest["operator"] = external_operator_records[0]
        external_ecosystem_manifest["selected_implementation"] = selected_implementation_records[only_node]

    synthesis_required = bool(build_stages.get("hls_synthesis", False))
    synthesis_characterization_ok = (
        not synthesis_required
        or (synthesis_characterization is not None and synthesis_characterization.status == "passed")
    )
    manifest = {
        "version": compiler.cfg.version,
        "schema": "fpgai.external-ecosystem-compile/v1",
        "status": (
            "passed"
            if (hls_run is None or hls_run.ok) and synthesis_characterization_ok
            else "failed"
        ),
        "model_path": compiler.cfg.model.path,
        "pipeline_mode": compiler.cfg.pipeline.mode,
        "top_kernel_name": str(get_path(raw, "pipeline.outputs.top_kernel_name", "deeplearn")),
        "external_ecosystem": external_ecosystem_manifest,
        "build_stages": {str(key): bool(value) for key, value in build_stages.items()},
        "hls_ran": hls_run is not None,
        "hls_ok": None if hls_run is None else bool(hls_run.ok),
        "pipeline_stages": [
            {"name": "discover_packages", "status": "done"},
            {"name": "load_external_operator", "status": "done"},
            {"name": "import_model", "status": "done"},
            {"name": "select_implementations_per_node", "status": "done"},
            {"name": "compose_mixed_hls_project", "status": "done"},
            {"name": "run_hls", "status": "skipped" if hls_run is None else ("done" if hls_run.ok else "failed")},
            {
                "name": "characterize_hls_synthesis",
                "status": (
                    "skipped"
                    if hls_run is None or not bool(getattr(hls_run, "csynth_ran", False))
                    else ("done" if synthesis_characterization is not None and synthesis_characterization.status == "passed" else "failed")
                ),
            },
            {"name": "analyze_hls_bottlenecks", "status": "skipped" if hls_run is None else "done"},
            {"name": "analyze_tensor_liveness", "status": "done"},
            {"name": "allocate_hls_buffers", "status": "done"},
        ],
        "seconds": round(time.time() - started, 6),
        "usage": {"platform_scope": "research", "production_path": "morfics"},
    }
    _write_json(out_dir / "manifest.json", manifest)

    vivado_requested = bool(build_stages.get("vivado_project") or build_stages.get("vivado_implementation") or build_stages.get("bitstream"))
    if vivado_requested:
        from fpgai.engine.vivado_pipeline import _run_yaml_requested_vivado_bridge
        _run_yaml_requested_vivado_bridge(out_dir, raw, dict(build_stages))
        from fpgai.analysis.vivado_implementation_characterization import (
            characterize_vivado_implementation,
            write_vivado_implementation_characterization,
        )
        vivado_characterization = characterize_vivado_implementation(
            out_dir,
            target_clock_mhz=clock_mhz,
            external_provenance={
                "operators": external_operator_records,
                "selected_implementations": selected_implementation_records,
                "package_lock": str(lock_path),
            },
        )
        vivado_json, vivado_md = write_vivado_implementation_characterization(vivado_characterization, reports_dir)
        refreshed = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        refreshed.setdefault("external_ecosystem", {})["vivado_implementation_characterization"] = {
            "status": vivado_characterization["status"],
            "validation_level": vivado_characterization["validation_level"],
            "reports": {"json": str(vivado_json), "markdown": str(vivado_md)},
            "scope": vivado_characterization["scope"],
        }
        refreshed.setdefault("pipeline_stages", []).append({
            "name": "characterize_vivado_implementation",
            "status": "done" if vivado_characterization["status"] == "passed" else ("skipped" if vivado_characterization["status"] == "not_run" else "failed"),
        })
        _write_json(out_dir / "manifest.json", refreshed)

    return ExternalEcosystemCompileResult(True, graph, project.hls_dir, hls_run, manifest["external_ecosystem"])
