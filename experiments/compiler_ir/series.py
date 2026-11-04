#!/usr/bin/env python3
"""Single-entry, fail-closed, source-pinned FPGAI compiler-paper experiment series.

Standard-library orchestration only; all experiment implementations live in
src/fpgai_research/experiments. No RL source is imported, trained or modified.
Vendor tools are never invoked unless *both* --with-native and
--confirm-isolated-native are present. Raw original outputs remain create-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("series.json")
SCHEMA = "fpgai-research/compiler-ir-experiment-series/v1"
RESULTS = ROOT / "build" / "compiler_ir" / "series"


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()
    output: str | None = None
    evidence: str = "E0/E1"
    optional: bool = False


def _git(*parts: str) -> str:
    p = subprocess.run(["git", "-C", str(ROOT), *parts],
                       capture_output=True, text=True, timeout=15)
    if p.returncode:
        raise RuntimeError("Git checkout required: " + p.stderr.strip())
    return p.stdout.strip()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked evidence input: " + str(path))
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(piece)
    return sha.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    # The series manifest is the only mutable record, updated atomically as
    # sequential experiments finish. Each campaign output stays create-only.
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def _load_config():
    info = json.loads(CONFIG.read_text())
    if info.get("schema") != "fpgai-research/compiler-ir-experiment-series-config/v1":
        raise ValueError("Unexpected series config schema")
    expected = {
        "array_cases", "array_host_pairs", "row_pairs", "row_host_pairs",
        "chain_cases", "chain_host_cases", "chain_traces",
        "mutation_graphs", "mutation_host", "mutation_traces",
        "oracle_domains", "scaling_graphs", "scaling_repetitions",
        "nonlinear_host", "nonlinear_traces", "tensor_native_cases",
        "block_native_cases",
        "decision_graphs", "decision_host_checks", "decision_seed",
        "value_trace_cases",
    }
    for name in ("quick", "paper"):
        values = info["profiles"][name]
        if set(values) != expected or any(
                type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("Unexpected series profile parameter")
        if values["tensor_native_cases"] % 3 or values["block_native_cases"] % 3:
            raise ValueError("Native source groups must be complete triplets")
    clock = info["native"]
    if not (0 <= clock["clock_uncertainty_ns"] < clock["clock_ns"]
            and clock["tolerance"] > 0
            and type(clock["timeout_seconds"]) is int):
        raise ValueError("Unsafe native toolclock config")
    return info


def _python(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


def _module(name: str, *args: str) -> tuple[str, ...]:
    return _python("-m", "fpgai_research.experiments." + name, *args)


def plan(root: Path, profile: str, *, trained_mlp: str | None = None,
         trained_cnn: str | None = None, onnx: bool = True,
         with_native: bool = False, native_executable: str = "vitis_hls",
         native_timeout: int | None = None) -> tuple[Step, ...]:
    """Pure command plan, no local side effects. May be inspected before run."""
    config = _load_config()
    if profile not in config["profiles"]:
        raise ValueError("profile must be quick or paper")
    p = config["profiles"][profile]
    native = config["native"]
    timeout = native_timeout if native_timeout is not None else native["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError("Native per-case timeout must be 1..86400")
    clk = ("--part", native["part"], "--clock-ns", str(native["clock_ns"]),
           "--clock-uncertainty-ns", str(native["clock_uncertainty_ns"]),
           "--tolerance", str(native["tolerance"]))
    paths = {name: str(root / name) for name in (
        "array", "row", "chain", "cpu_paper_table", "schedule_mutations",
        "independent_oracle", "scaling", "nonlinear", "tensor_source",
        "tensor_source_audit", "tensor_source_report", "row_block_source",
        "row_block_source_report", "tensor_native_report",
        "row_block_native_report", "decision_interface", "rl_compatible_ir_r3_r4",
        "two_phi_counterfactual_e1", "rl_compatible_ir_e1_action_qualification", "multi_reader_certified_actions_e1", "unified_policy_interface_e1", "accounting_baseline_sensitivity_e0", "emitter_shadow_mutation_e1", "phi_scaling_sensitivity_e0", "tensor_exact_comparator_e0", "hls_mechanism_factorial_source_e1",
        "value_lifetime", "mapped_event_causality", "mapped_source_e1", "fanout_e0", "fanout_exact_e0", "fanout_source_e1", "fanout_source_optimum_e1", "fanout_phi_source_e1", "two_phi_readers_e1", "multi_reader_exact_e1", "branch_ablation_e0", "declared_bank_ports_e0", "graph_phi_e1", "graph_access_e1", "graph_semantic_oracle_e0", "conv_phi_boundary_e0_e1_numeric", "nchw_graph_e0_partial_e1", "nchw_all_intermediates_e1", "nchw_real_source_scratch_e1", "nchw_real_source_rw_e1", "nchw_native_pair_packages_e1", "trained_mlp_preflight",
        "trained_mlp_host", "trained_mlp_native_source",
        "trained_cnn_preflight",
    )}
    out = lambda name: ("--output", paths[name])
    common = ("numpy",)
    steps: list[Step] = []
    def add(name, command, *, deps=(), modules=common, executables=(),
            output=None, evidence="E0/E1", optional=False):
        steps.append(Step(name, tuple(command), tuple(deps), tuple(modules),
                          tuple(executables), output, evidence, optional))

    test_files = sorted(str(t.relative_to(ROOT)) for t in
                        (ROOT / "tests").glob("test_cst_*.py")
                        if t.name != "test_cst_nonlinear_onnx_integration.py")
    test_files.extend([
        "tests/test_architecture_interfaces.py",
        "tests/test_compositional_model_ir.py",
        "tests/test_temporal_schedule.py",
    ])
    if not test_files or any(not (ROOT / f).is_file() for f in test_files):
        raise RuntimeError("Compiler research test coverage files missing")
    add("compiler_tests", _python("-m", "pytest", "-q", *test_files),
        modules=("pytest", "numpy"), evidence="E0/E1-tests")
    if onnx:
        add("onnx_crosscheck", _python("-m", "pytest", "-q", "-rs",
            "tests/test_cst_nonlinear_onnx_integration.py"),
            modules=("pytest", "numpy", "onnx", "onnxruntime"),
            executables=("g++",), evidence="E1-independent-ONNX-GCC",
            optional=True)

    add("array", _module("cst_ir_campaign", *out("array"),
        "--max-cases", str(p["array_cases"]),
        "--host-check-pairs", str(p["array_host_pairs"])),
        deps=("compiler_tests",), executables=("g++",),
        output="array")
    add("row", _module("cst_row_campaign", *out("row"),
        "--max-pairs", str(p["row_pairs"]),
        "--host-pairs", str(p["row_host_pairs"])),
        deps=("compiler_tests",), executables=("g++",), output="row")
    add("chain", _module("cst_chain_campaign", *out("chain"),
        "--max-cases", str(p["chain_cases"]),
        "--host-cases", str(p["chain_host_cases"]),
        "--full-trace-cases", str(p["chain_traces"])),
        deps=("compiler_tests",), executables=("g++",), output="chain")
    add("cpu_paper_table", _module("cst_cpu_report",
        "--array-study", paths["array"], "--row-study", paths["row"],
        "--chain-study", paths["chain"], *out("cpu_paper_table")),
        deps=("array", "row", "chain"), output="cpu_paper_table",
        evidence="E0/E1-read-only-table")
    add("schedule_mutations", _module("cst_schedule_ablation",
        *out("schedule_mutations"), "--max-graphs", str(p["mutation_graphs"]),
        "--host-cases", str(p["mutation_host"]),
        "--row-trace-cases", str(p["mutation_traces"])),
        deps=("compiler_tests",), executables=("g++",),
        output="schedule_mutations")
    add("independent_oracle", _module("cst_oracle_campaign",
        *out("independent_oracle"), "--domains", str(p["oracle_domains"])),
        deps=("compiler_tests",), output="independent_oracle", evidence="E0")
    add("scaling", _module("cst_scaling_campaign", *out("scaling"),
        "--max-graphs", str(p["scaling_graphs"]),
        "--repetitions", str(p["scaling_repetitions"])),
        deps=("compiler_tests",), output="scaling", evidence="E0-CPU-timing")
    add("nonlinear", _module("cst_nonlinear_campaign", *out("nonlinear"),
        "--host-cases", str(p["nonlinear_host"]),
        "--trace-cases", str(p["nonlinear_traces"])),
        deps=("compiler_tests",), executables=("g++",), output="nonlinear")
    add("decision_interface", _module("cst_decision_campaign",
        *out("decision_interface"), "--graphs", str(p["decision_graphs"]),
        "--host-checks", str(p["decision_host_checks"]),
        "--seed", str(p["decision_seed"])),
        deps=("compiler_tests",), executables=("g++",),
        output="decision_interface", evidence="E0/E1-search-interface-controls")
    add("rl_compatible_ir_r3_r4", _module("cst_rl_interface_study",
        *out("rl_compatible_ir_r3_r4")),
        deps=("compiler_tests","decision_interface"),
        output="rl_compatible_ir_r3_r4",
        evidence="E0-policy-independent-certificate-graph-observation-R3-R4")
    add("rl_compatible_ir_e1_action_qualification",
        _module("cst_r2n_e1_action_study",
                *out("rl_compatible_ir_e1_action_qualification")),
        deps=("compiler_tests","decision_interface","rl_compatible_ir_r3_r4"),
        executables=("g++",),
        output="rl_compatible_ir_e1_action_qualification",
        evidence="E0-mapped-compiler-masks-plus-E1-actual-GCC-source-qualified-actions")
    add("multi_reader_certified_actions_e1",
        _module("cst_r2o_multi_reader_action_study",
                *out("multi_reader_certified_actions_e1"),
                "--graphs", "3" if profile == "paper" else "1"),
        deps=("compiler_tests","rl_compatible_ir_e1_action_qualification"),
        executables=("g++",),
        output="multi_reader_certified_actions_e1",
        evidence="E0-independent-Phi-multi-reader-source-and-E1-real-GCC")
    add("unified_policy_interface_e1",
        _module("cst_r2p_unified_interface_study",
                *out("unified_policy_interface_e1")),
        deps=("compiler_tests","rl_compatible_ir_e1_action_qualification",
              "multi_reader_certified_actions_e1"),
        executables=("g++",),
        output="unified_policy_interface_e1",
        evidence="policy-neutral-E0-source-plus-E1-GCC-common-interface-cross-domain")
    add("accounting_baseline_sensitivity_e0",
        _module("cst_r2q_accounting_baselines",
                *out("accounting_baseline_sensitivity_e0")),
        deps=("compiler_tests","multi_reader_certified_actions_e1"),
        output="accounting_baseline_sensitivity_e0",
        evidence="E0-adversarial-whole-tensor-baselines-and-inplace-lifetime-convention")
    add("emitter_shadow_mutation_e1",
        _module("cst_r2r_emitter_shadow_mutation",
                *out("emitter_shadow_mutation_e1"),
                "--graphs", "6" if profile == "paper" else "1",
                "--mutants-per-graph", "36" if profile == "paper" else "4"),
        deps=("compiler_tests","accounting_baseline_sensitivity_e0"),
        executables=("g++",),
        output="emitter_shadow_mutation_e1",
        evidence="E1-source-emitter-actual-slot-shadow-and-targeted-GCC-mutation-adequacy")
    add("phi_scaling_sensitivity_e0",
        _module("cst_r2s_phi_scaling",
                *out("phi_scaling_sensitivity_e0"),
                "--shape-limit", "7" if profile == "paper" else "1"),
        deps=("compiler_tests","accounting_baseline_sensitivity_e0"),
        output="phi_scaling_sensitivity_e0",
        evidence="E0-147-case-adversarial-Phi-scaling-and-reader-inplace-counterfactual")
    add("tensor_exact_comparator_e0",
        _module("cst_r2t_tensor_optimum",
                *out("tensor_exact_comparator_e0")),
        deps=("compiler_tests","accounting_baseline_sensitivity_e0"),
        output="tensor_exact_comparator_e0",
        evidence="E0-exact-contiguous-whole-tensor-comparator-bounded-optimality")
    add("hls_mechanism_factorial_source_e1",
        _module("cst_r2u_hls_mechanism_prepare",
                *out("hls_mechanism_factorial_source_e1")),
        deps=("compiler_tests","tensor_exact_comparator_e0"),
        output="hls_mechanism_factorial_source_e1",
        evidence="E1-source-only-32-Vitis-packages-2x2x2x2-mechanism-factorial")
    add("value_lifetime", _module("cst_value_trace_campaign",
        *out("value_lifetime"), "--max-cases", str(p["value_trace_cases"])),
        deps=("compiler_tests",), executables=("g++",),
        output="value_lifetime", evidence="E1-executed-Phi-and-value-last-use")
    add("mapped_event_causality", _module("cst_phi_event_campaign",
        *out("mapped_event_causality")),
        deps=("compiler_tests",), output="mapped_event_causality",
        evidence="E0-Phi-owner-logical-event-causality")
    add("fanout_e0", _module("cst_fanout_campaign", *out("fanout_e0")),
        deps=("compiler_tests",), output="fanout_e0",
        evidence="E0-abstract-two-consumer-lifetime-only")
    add("fanout_exact_e0", _module("cst_fanout_allocator_campaign",
        *out("fanout_exact_e0")),
        deps=("compiler_tests",), output="fanout_exact_e0",
        evidence="E0-bounded-exact-scratch-alias-allocation")
    add("fanout_source_e1", _module("cst_fanout_source_campaign",
        *out("fanout_source_e1")),
        deps=("compiler_tests",), executables=("g++",),
        output="fanout_source_e1", evidence="E1-executed-GCC-two-reader-source")
    add("fanout_source_optimum_e1",
        _module("cst_fanout_source_optimum_campaign",
                *out("fanout_source_optimum_e1")),
        deps=("compiler_tests",), executables=("g++",),
        output="fanout_source_optimum_e1",
        evidence="E0-exact-source-event-minimum-plus-E1-executed-GCC")
    add("fanout_phi_source_e1",
        _module("cst_phi_fanout_source_campaign",*out("fanout_phi_source_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="fanout_phi_source_e1",
        evidence="E0-mapped-MatMul-Phi-plus-E1-executed-GCC-fanout")
    add("two_phi_readers_e1",
        _module("cst_two_phi_reader_campaign",*out("two_phi_readers_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="two_phi_readers_e1",
        evidence="E0-Phi-mapped-multi-contribution-last-use-plus-E1-GCC")
    add("two_phi_counterfactual_e1",
        _module("cst_phi_counterfactual_effects",
            "--input-csv", str(root / "two_phi_readers_e1" / "cases.csv"),
            *out("two_phi_counterfactual_e1")),
        deps=("two_phi_readers_e1",),
        output="two_phi_counterfactual_e1",
        evidence="E1-read-only-source-Phi-causal-effect-audit-not-original-GCC")
    add("multi_reader_exact_e1",
        _module("cst_branch_exact_campaign",*out("multi_reader_exact_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="multi_reader_exact_e1",
        evidence="E0-provably-minimal-contiguous-Phi-multi-read-source-layout-plus-E1-GCC")
    add("branch_ablation_e0",
        _module("cst_branch_ablation_campaign",*out("branch_ablation_e0")),
        deps=("compiler_tests",),output="branch_ablation_e0",
        evidence="E0-controlled-finite-weak-certificate-mutation-ablation")
    add("declared_bank_ports_e0",
        _module("cst_branch_bank_port_campaign",*out("declared_bank_ports_e0")),
        deps=("compiler_tests",),output="declared_bank_ports_e0",
        evidence="E0-assumed-striped-ports-UNKNOWN-without-HLS-schedule")
    add("graph_phi_e1",
        _module("cst_graph_phi_campaign",*out("graph_phi_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="graph_phi_e1",
        evidence="E0-topological-graph-Phi-read-owner-last-use-plus-E1-executed-GCC")
    add("graph_access_e1",
        _module("cst_graph_access_campaign",*out("graph_access_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="graph_access_e1",
        evidence="E0-nonidentity-broadcast-Concat-access-map-plus-E1-GCC")
    add("graph_semantic_oracle_e0",
        _module("cst_graph_semantic_oracle_campaign",*out("graph_semantic_oracle_e0")),
        deps=("compiler_tests",),output="graph_semantic_oracle_e0",
        evidence="E0-independent-NumPy-graph-access-vs-correlated-map-falsification")
    add("conv_phi_boundary_e0_e1_numeric",
        _module("cst_conv_boundary_campaign",*out("conv_phi_boundary_e0_e1_numeric")),
        deps=("compiler_tests",),executables=("g++",),
        output="conv_phi_boundary_e0_e1_numeric",
        evidence="E0-conv-kernel-phi-and-NumPy-source-read-geometry-plus-E1-GCC-numeric-only")
    add("nchw_graph_e0_partial_e1",
        _module("cst_nchw_graph_campaign",*out("nchw_graph_e0_partial_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="nchw_graph_e0_partial_e1",
        evidence="E0-general-mixed-NCHW-graph-lifetime-NumPy-plus-E1-GCC-Conv-LOAD")
    add("nchw_all_intermediates_e1",
        _module("cst_nchw_intermediate_campaign",*out("nchw_all_intermediates_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="nchw_all_intermediates_e1",
        evidence="E1-original-GCC-operator-tensor-snapshots-vs-independent-NumPy")
    add("nchw_real_source_scratch_e1",
        _module("cst_nchw_shared_scratch_campaign",*out("nchw_real_source_scratch_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="nchw_real_source_scratch_e1",
        evidence="E1-actual-GCC-shared-C++-scratch-source-vs-disjoint-C++-source")
    add("nchw_real_source_rw_e1",
        _module("cst_nchw_access_campaign",*out("nchw_real_source_rw_e1")),
        deps=("compiler_tests",),executables=("g++",),
        output="nchw_real_source_rw_e1",
        evidence="E1-original-GCC-observed-every-NCHW-scratch-READ-WRITE-and-NumPy-dynamic-count")
    add("nchw_native_pair_packages_e1",
        _module("cst_nchw_native_pairs",*out("nchw_native_pair_packages_e1")),
        deps=("compiler_tests",),output="nchw_native_pair_packages_e1",
        evidence="E1-source-only-matched-native-Vitis-package-preparation-NO-original-vendor-tool")
    add("mapped_source_e1", _module("cst_mapped_refinement_campaign",
        *out("mapped_source_e1")), deps=("compiler_tests",),
        executables=("g++",), output="mapped_source_e1",
        evidence="E0-Phi-contract-to-E1-actual-compiler-source")

    add("tensor_source", _module("cst_native_campaign", "prepare",
        *out("tensor_source"), "--max-cases", str(p["tensor_native_cases"]),
        "--shape-catalog", ("paper" if profile == "paper" else "pilot"), *clk),
        deps=("compiler_tests",), output="tensor_source",
        evidence="E1-original-HLS-inputs-only")
    add("tensor_source_audit", _module("cst_native_campaign", "audit",
        "--study", paths["tensor_source"]),
        deps=("tensor_source",), evidence="E1-source-audit")
    add("tensor_source_report", _module("cst_native_report",
        "--study", paths["tensor_source"], *out("tensor_source_report")),
        deps=("tensor_source",), output="tensor_source_report",
        evidence="E1-empty-native-scorecard")
    add("row_block_source", _module("cst_block_native_prepare",
        *out("row_block_source"), "--max-cases", str(p["block_native_cases"]),
        "--shape-catalog", ("paper" if profile == "paper" else "pilot"), *clk),
        deps=("compiler_tests",), output="row_block_source",
        evidence="E1-original-HLS-inputs-only")
    add("row_block_source_report", _module("cst_block_native_report",
        "--study", paths["row_block_source"], *out("row_block_source_report")),
        deps=("row_block_source",), output="row_block_source_report",
        evidence="E1-empty-native-scorecard")

    if trained_mlp is not None:
        for name, module, extra, deps, requirements in (
            ("trained_mlp_preflight", "cst_trained_model_preflight",
             ("--kind", "mlp"), (), ("numpy", "onnx")),
            ("trained_mlp_host", "cst_trained_mlp_host",
             ("--host-samples", "2"), ("trained_mlp_preflight",),
             ("numpy", "onnx", "onnxruntime")),
            ("trained_mlp_native_source", "cst_trained_mlp_native_prepare",
             ("--sample-count", "2", "--part", native["part"],
              "--clock-ns", str(native["clock_ns"]),
              "--clock-uncertainty-ns", str(native["clock_uncertainty_ns"]),
              "--numeric-tolerance", str(native["tolerance"])),
             ("trained_mlp_preflight",), ("numpy", "onnx")),
        ):
            add(name, _module(module,
                "--trained-directory", trained_mlp, *out(name), *extra),
                deps=deps, modules=requirements,
                executables=("g++",) if name == "trained_mlp_host" else (),
                output=name, optional=True, evidence="E0/E1-trained-assets")
    if trained_cnn is not None:
        add("trained_cnn_preflight", _module("cst_trained_model_preflight",
            "--trained-directory", trained_cnn, "--kind", "cnn",
            *out("trained_cnn_preflight")),
            modules=("numpy", "onnx"), output="trained_cnn_preflight",
            evidence="E0-coverage-only", optional=True)

    if with_native:
        for family, prepared, total, report_module, native_report in (
            ("tensor", "tensor_source", p["tensor_native_cases"],
             "cst_native_report", "tensor_native_report"),
            ("row_block", "row_block_source", p["block_native_cases"],
             "cst_block_native_report", "row_block_native_report"),
        ):
            key = family + "_native_execute"
            add(key, _module("cst_native_campaign", "execute",
                "--study", paths[prepared], "--max-calls", str(total),
                "--execute-native", "--vitis-executable", native_executable,
                "--timeout-seconds", str(timeout)),
                deps=(prepared,), executables=(native_executable,),
                evidence="E2/E3-original-vitis", optional=False)
            add(native_report, _module(report_module,
                "--study", paths[prepared], *out(native_report)),
                deps=(key,), output=native_report, evidence="E2/E3-read-only")
    # Last step reads the original source-pinned manifest and all retained
    # raw CSV/JSON. It never fabricates absent hardware or trained results.
    add("evidence_index", _python(
        str(ROOT / "experiments" / "compiler_ir" / "export_evidence.py"),
        "--series-root", str(root), "--output", str(root / "evidence_index")),
        modules=(), output="evidence_index", evidence="E0/E1-read-only-index")
    return tuple(steps)


def _dependency_missing(step: Step) -> list[str]:
    missing = [m for m in step.modules if importlib.util.find_spec(m) is None]
    missing += [exe for exe in step.executables if shutil.which(exe) is None]
    return missing


def _artifacts(folder: Path) -> list[dict]:
    if folder.is_symlink():
        raise ValueError("Symlinked scientific evidence directory")
    if not folder.is_dir():
        raise ValueError("Successful stage did not create its output folder")
    records = []
    for p in sorted(folder.rglob("*")):
        if p.is_symlink():
            raise ValueError("Symlink inside evidence outputs: " + str(p))
        if p.is_file():
            records.append({
                "path": str(p.relative_to(folder)),
                "bytes": p.stat().st_size, "sha256": _sha(p),
            })
    if not records:
        raise ValueError("Stage reported success without source evidence files")
    return records


def _new_manifest(args, git_sha: str, root: Path) -> dict:
    return {
        "schema": SCHEMA, "source_revision": git_sha,
        "configuration_sha256": _sha(CONFIG),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "profile": args.profile or "paper",
        "trained_mlp": str(Path(args.trained_mlp).resolve()) if args.trained_mlp else None,
        "trained_cnn": str(Path(args.trained_cnn).resolve()) if args.trained_cnn else None,
        "onnx_enabled": not args.skip_onnx,
        "native_enabled": args.with_native,
        "native_executable": args.vitis_executable,
        "native_timeout": args.native_timeout,
        "result_directory": str(root),
        "steps": {}, "evidence_gates": {
            "native_vitis_hls_performed": False,
            "trained_model_e1_qualified": False,
            "external_prior_art_comparison_qualified": False,
            "conv_and_branch_lifetime_qualified": False,
            "physical_fpga_result_qualified": False,
        },
        "external_research_gates": _load_config()["external_research_gates"],
    }


def _run_step(step: Step, result_root: Path, sha: str) -> dict:
    logdir = result_root / "logs"
    logdir.mkdir(exist_ok=True)
    logfile = logdir / (step.name + ".log")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "1")
    missing = _dependency_missing(step)
    if missing:
        return {"status": "BLOCKED", "reason": "Missing dependencies: " + ", ".join(missing),
                "command": list(step.command), "evidence": step.evidence}
    if _git("rev-parse", "HEAD") != sha:
        return {"status": "FAILED", "reason": "Git HEAD changed during study",
                "command": list(step.command), "evidence": step.evidence}
    if step.output and (result_root / step.output).exists():
        return {"status": "FAILED", "reason": "Create-only stage output already exists",
                "command": list(step.command), "evidence": step.evidence}
    started = datetime.now(timezone.utc).isoformat()
    with logfile.open("w") as out:
        out.write("# source_revision=" + sha + "\n# command=" +
                  json.dumps(list(step.command)) + "\n")
        out.flush()
        try:
            proc = subprocess.run(step.command, cwd=ROOT, env=env,
                                  stdout=out, stderr=subprocess.STDOUT)
            code = proc.returncode
        except OSError as error:
            code = 127
            out.write("\nLauncher exception: " + repr(error) + "\n")
    result = {
        "status": "PASSED" if code == 0 else "FAILED",
        "exit_code": code, "command": list(step.command), "evidence": step.evidence,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(logfile.relative_to(result_root)), "log_sha256": _sha(logfile),
    }
    if code == 0 and step.output:
        try:
            result["artifacts"] = _artifacts(result_root / step.output)
        except ValueError as exc:
            result.update(status="FAILED", reason=str(exc))
    if _git("rev-parse", "HEAD") != sha:
        result.update(status="FAILED", reason="Source revision changed during execution")
    return result


def execute(args) -> int:
    config = _load_config()
    sha = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked changes exist. Run from a clean, committed, isolated checkout")
    if args.with_native and not args.confirm_isolated_native:
        raise RuntimeError("--with-native needs --confirm-isolated-native; never use active RL checkout")
    if args.confirm_isolated_native and not args.with_native:
        raise ValueError("--confirm-isolated-native requires --with-native")
    if args.resume:
        result_root = Path(args.resume).expanduser().resolve()
        if result_root.is_symlink() or not (result_root / "series_manifest.json").is_file():
            raise ValueError("Resume requires existing non-symlinked series_manifest.json")
        manifest = json.loads((result_root / "series_manifest.json").read_text())
        if (manifest.get("schema") != SCHEMA or manifest["source_revision"] != sha
                or manifest["configuration_sha256"] != _sha(CONFIG)):
            raise ValueError("Refusing resume with mismatched source, plan, or config")
        profile = manifest["profile"]
        trained_mlp, trained_cnn = manifest["trained_mlp"], manifest["trained_cnn"]
        onnx, with_native = manifest["onnx_enabled"], manifest["native_enabled"]
        native_executable = manifest["native_executable"]
        native_timeout = manifest["native_timeout"]
        if args.with_native != with_native:
            raise ValueError("Resume must repeat the original --with-native choice")
        if args.with_native and not args.confirm_isolated_native:
            raise ValueError("Native resume requires explicit isolated-host confirmation")
        if args.profile and args.profile != profile:
            raise ValueError("Cannot change experiment profile during resume")
        if args.trained_mlp and str(Path(args.trained_mlp).resolve()) != trained_mlp:
            raise ValueError("Cannot change trained assets on resume")
    else:
        profile = args.profile or "paper"
        trained_mlp, trained_cnn = args.trained_mlp, args.trained_cnn
        onnx, with_native = not args.skip_onnx, args.with_native
        native_executable, native_timeout = args.vitis_executable, args.native_timeout
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + sha[:10]
        result_root = (Path(args.output).expanduser().resolve() if args.output
                       else RESULTS / name)
        if result_root.exists() or result_root.is_symlink():
            raise FileExistsError("Experiment series is create-only: " + str(result_root))
        manifest = _new_manifest(args, sha, result_root)
    stages = plan(result_root, profile, trained_mlp=trained_mlp,
                  trained_cnn=trained_cnn, onnx=onnx, with_native=with_native,
                  native_executable=native_executable, native_timeout=native_timeout)
    if args.plan:
        print(json.dumps({
            "source_revision": sha, "profile": profile,
            "result_directory": str(result_root),
            "commands": [asdict(stage) for stage in stages],
            "external_research_gates": config["external_research_gates"],
        }, indent=2))
        return 0
    if not args.resume:
        result_root.mkdir(parents=True)
        _write_json(result_root / "series_manifest.json", manifest)
        shutil.copyfile(CONFIG, result_root / "source_plan.json")
    results = manifest["steps"]
    for i, step in enumerate(stages, 1):
        existing = results.get(step.name)
        if existing is not None:
            # Critical: never overwrite any campaign evidence, including
            # a partially written failed/interrupted source package.
            print(f"[{i}/{len(stages)}] {step.name}: existing {existing['status']}; not rerun")
            continue
        unmet = [dep for dep in step.dependencies
                 if results.get(dep, {}).get("status") != "PASSED"]
        if unmet:
            result = {"status": "BLOCKED",
                      "reason": "Unqualified prerequisites: " + ", ".join(unmet),
                      "evidence": step.evidence}
        else:
            print(f"[{i}/{len(stages)}] {step.name}: executing", flush=True)
            result = _run_step(step, result_root, sha)
        results[step.name] = result
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(result_root / "series_manifest.json", manifest)
        print(f"  -> {result['status']}" +
              (": " + result["reason"] if result.get("reason") else ""), flush=True)
    manifest["evidence_gates"]["native_vitis_hls_performed"] = any(
        results.get(family + "_native_execute", {}).get("status") == "PASSED"
        for family in ("tensor", "row_block"))
    manifest["evidence_gates"]["trained_model_e1_qualified"] = (
        results.get("trained_mlp_host", {}).get("status") == "PASSED")
    # Native tool invocation may complete with CSim failure, an unbounded
    # synthesis report or a missing case. A green orchestration step is NOT
    # a qualified original-HLS scientific experiment. Read the independently
    # generated pair scorecards and keep denominator and missing cases.
    native_matched = {}
    for family, report_name, result_file, count_key, total_key in (
        ("tensor", "tensor_native_report", "report.json",
         "paired_rows_with_source_attested_numerical_and_cycles", "paired_variant_rows"),
        ("row_block", "row_block_native_report", "summary.json",
         "pairs_with_original_csim_csynth_measurements", "paired_comparisons"),
    ):
        native_matched[family] = {
            "required_pairs": None, "qualified_pairs": 0,
            "all_pairs_qualified": False,
            "status": "NOT_REQUESTED" if not with_native else "UNQUALIFIED",
        }
        if with_native and results.get(report_name, {}).get("status") == "PASSED":
            raw = json.loads((result_root / report_name / result_file).read_text())
            numerator = raw.get(count_key)
            denominator = raw.get(total_key)
            if (type(numerator) is not int or type(denominator) is not int
                    or not 0 <= numerator <= denominator):
                raise ValueError("Invalid original HLS native audit denominators")
            native_matched[family].update({
                "required_pairs": denominator,
                "qualified_pairs": numerator,
                "all_pairs_qualified": denominator > 0 and numerator == denominator,
                "status": ("QUALIFIED" if denominator > 0 and numerator == denominator
                           else "INCOMPLETE_OR_FAILED"),
            })
    manifest["evidence_gates"]["all_native_pairs_qualified"] = bool(
        with_native and all(r["all_pairs_qualified"] for r in native_matched.values()))
    tally = {x: sum(r.get("status") == x for r in results.values())
             for x in ("PASSED", "FAILED", "BLOCKED")}
    summary = {
        "schema": "fpgai-research/compiler-ir-experiment-series-summary/v1",
        "source_revision": sha, "profile": profile,
        "experiment_steps": len(stages),
        "counts": tally,
        "native_tool_use_requested": with_native,
        "original_hls_matched_pair_qualification": native_matched,
        "all_original_hls_pairs_qualified": (
            manifest["evidence_gates"]["all_native_pairs_qualified"]),
        "source_evidence_qualified": (
            results.get("cpu_paper_table", {}).get("status") == "PASSED"),
        "hardware_quality_claim_established": False,
        "prior_art_novelty_claim_established": False,
        "trained_mlp_executed": manifest["evidence_gates"]["trained_model_e1_qualified"],
        "results_directory": str(result_root),
        "external_research_gates": config["external_research_gates"],
    }
    _write_json(result_root / "series_summary.json", summary)
    _write_json(result_root / "series_manifest.json", manifest)
    print("\nSERIES SUMMARY\n" + json.dumps(summary, indent=2, sort_keys=True))
    if tally["FAILED"]:
        return 1
    if args.require_all and (
            tally["BLOCKED"] or not with_native or not trained_mlp
            or not manifest["evidence_gates"]["all_native_pairs_qualified"]
            or not manifest["evidence_gates"]["trained_model_e1_qualified"]):
        print("Strict local-data gate not satisfied: original HLS measurements and "
              "trained model must be independently qualified; external prior art "
              "and novelty are still separate unresolved publication gates")
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "paper"), default=None)
    parser.add_argument("--output", help="New create-only result directory")
    parser.add_argument("--resume", help="Previously interrupted result directory")
    parser.add_argument("--plan", action="store_true",
                        help="Print commands and gates; do not execute or create folders")
    parser.add_argument("--doctor", action="store_true",
                        help="Inspect local PC readiness; do not start any experiment")
    parser.add_argument("--skip-onnx", action="store_true",
                        help="Explicitly omit independent ONNX Runtime test")
    parser.add_argument("--trained-mlp", help="Real original trained MNIST MLP asset folder")
    parser.add_argument("--trained-cnn", help="Real original trained MNIST CNN asset folder")
    parser.add_argument("--with-native", action="store_true",
                        help="Run original vendor Vitis HLS on two prepared native studies")
    parser.add_argument("--confirm-isolated-native", action="store_true",
                        help="Confirm this is not the active source-pinned RL checkout")
    parser.add_argument("--vitis-executable", default="vitis_hls")
    parser.add_argument("--native-timeout", type=int, default=None)
    parser.add_argument("--require-all", action="store_true",
                        help="Return nonzero when optional/local research gates are blocked")
    args = parser.parse_args(argv)
    if args.doctor:
        import doctor
        return doctor.main([
            "--checkout", str(ROOT),
            *(["--with-native"] if args.with_native else []),
            "--vitis-executable", args.vitis_executable,
            *(["--trained-mlp", args.trained_mlp] if args.trained_mlp else []),
            *(["--trained-cnn", args.trained_cnn] if args.trained_cnn else []),
        ])
    try:
        return execute(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print("SERIES PRE-FLIGHT ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
