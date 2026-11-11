"""Create-only cross-network FPGAI experiment matrix and read-only evidence table.

Four deterministic synthetic models x scalar/mapped x fixed-point precision x
requested clock. Preparing emits genuine compiler/Vitis/Vivado source scripts
and source-attested CSim reference inputs, but NEVER executes vendor tools.
Vendor runs require explicit '--execute'. All result metrics originate from
the canonical source/RTL-attested summarize_case and strict E3/E4 qualifiers.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
import shutil
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA = "fpgai-research/cross-network-benchmark/v1"
RESULT_SCHEMA = "fpgai-research/cross-network-results/v1"
NETWORKS = ("tiny_mlp", "cnn_pool_dense", "cnn_strided_dense",
            "cnn_two_conv_softmax")
MAPPINGS = ("scalar", "independent")
PRECISIONS = {"fx16_6": (16, 6), "fx12_5": (12, 5)}
CLOCKS_NS = (12.0, 20.0)
FAMILY = {"tiny_mlp": "mlp", "cnn_pool_dense": "cnn_pool",
          "cnn_strided_dense": "cnn_strided",
          "cnn_two_conv_softmax": "cnn_two_conv"}
COLUMNS = (
    "case", "network", "family", "mapping", "precision", "precision_bits",
    "integer_bits", "requested_clock_ns", "requested_clock_mhz",
    "source_cpp_sha256", "evidence", "csim_passed", "hls_qualified",
    "hls_latency_best_cycles", "hls_latency_worst_cycles", "hls_ii_min",
    "hls_nominal_worst_latency_us", "hls_estimated_lut", "hls_estimated_ff",
    "hls_estimated_dsp", "hls_estimated_bram18", "hls_estimated_uram",
    "e4_attested", "e4_qualified", "postroute_wns_ns", "postroute_timing_met",
    "postroute_drc_violations", "postroute_drc_clean",
    "postroute_lut", "postroute_ff", "postroute_dsp",
    "postroute_bram", "postroute_uram", "measured_board_latency_us",
    "measured_fmax_mhz", "hls_blockers", "e4_blockers",
)
_CASE_KEYS = {"case", "network", "family", "mapping", "precision",
              "precision_bits", "integer_bits", "clock_ns", "part",
              "model_onnx_sha256", "cpp_sha256", "testbench_sha256",
              "hls_tcl_sha256", "vivado_tcl_sha256",
              "functional_sha256", "operator_kinds",
              "input_shape", "output_shape"}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _checkout_revision() -> str | None:
    """Record source checkout revision; never infer it from a case path."""
    import re
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None


def _regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected ordinary file, not link: {path}")
    return path


def plan(*, networks=NETWORKS, mappings=MAPPINGS,
         precisions=tuple(PRECISIONS), clocks=CLOCKS_NS, part="xck26-sfvc784-2LV-c"):
    import re
    if not isinstance(part, str) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", part):
        raise ValueError("Invalid FPGA part")
    for actual, allowed, field in (
        (networks, NETWORKS, "networks"),
        (mappings, MAPPINGS, "mappings"),
        (precisions, PRECISIONS, "precisions"),
    ):
        if (not actual or len(set(actual)) != len(actual)
                or any(item not in allowed for item in actual)):
            raise ValueError(f"Invalid or duplicate {field}")
    if (not clocks or any(type(c) not in (float, int) or not isfinite(c)
                          or c <= 0 for c in clocks)
            or len(set(clocks)) != len(clocks)):
        raise ValueError("Requested clocks must be positive finite unique numbers")
    rows = []
    for network in networks:
        for mapping in mappings:
            for precision in precisions:
                for clock in clocks:
                    idx = len(rows)
                    bits, integer = PRECISIONS[precision]
                    rows.append({
                        "case": f"case_{idx:04d}",
                        "network": network, "family": FAMILY[network],
                        "mapping": mapping, "precision": precision,
                        "precision_bits": bits, "integer_bits": integer,
                        "clock_ns": float(clock), "part": part,
                    })
    return rows


def _model(path: Path, network: str):
    from examples import make_mlp_onnx, make_neural_network_onnx
    if network == "tiny_mlp":
        make_mlp_onnx.save_model(path)
    else:
        make_neural_network_onnx.save_model(network, path)


def prepare(output: str | Path, *, networks=NETWORKS, mappings=MAPPINGS,
            precisions=tuple(PRECISIONS), clocks=CLOCKS_NS,
            part="xck26-sfvc784-2LV-c", seed: int = 197,
            samples: int = 2, tolerance: float = 0.1) -> dict:
    from examples.compile_neural_network import operator_mapping
    import numpy as np
    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.compiler import compile_architecture
    from fpgai_research.compiler.neural_chain import compile_neural_chain
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.model_ir.contracts import ArchitectureRequest
    from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
    from fpgai_research.toolchains.vivado.project import write_vivado_package

    rows = plan(networks=networks, mappings=mappings, precisions=precisions,
                clocks=clocks, part=part)
    if type(seed) is not int or type(samples) is not int or not 1 <= samples <= 8:
        raise ValueError("Seed must be int and samples must be 1..8")
    if type(tolerance) not in (int, float) or not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite numerical tolerance required")
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Benchmark directory must be new; original studies are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("Benchmark output parent cannot be a symlink")

    with TemporaryDirectory(prefix=".fpgai-network-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        model_paths = {}
        for network in networks:
            path = stage / "models" / f"{network}.onnx"
            path.parent.mkdir(parents=True, exist_ok=True)
            _model(path, network)
            model_paths[network] = path

        for row in rows:
            path = model_paths[row["network"]]
            model = load_onnx(str(path))
            base = replace(ArchitectureRequest(),
                           precision_bits=row["precision_bits"],
                           integer_bits=row["integer_bits"])
            mapping = operator_mapping(model, mapping=row["mapping"])
            # Important: operator-local requests must NOT accidentally
            # restore default FX16_6 in a FX12_5 experiment.
            mapping = {
                name: replace(item, precision_bits=row["precision_bits"],
                              integer_bits=row["integer_bits"])
                for name, item in mapping.items()
            }
            arch = resolve(model, base, mapping)
            compiled = (compile_architecture(arch) if row["network"] == "tiny_mlp"
                        else compile_neural_chain(arch).compilation)
            emission = compiled.emission
            case = stage / "cases" / row["case"]
            case.parent.mkdir(exist_ok=True)
            shape = model.tensor(model.inputs[0]).shape
            rng = np.random.default_rng(seed)
            samples_x = rng.uniform(
                -0.25, 0.25, size=(samples, *shape)).astype(np.float32)
            write_vitis_package(
                str(case), model, emission, samples_x,
                part=row["part"], clock_ns=row["clock_ns"],
                tolerance=tolerance,
            )
            write_vivado_package(
                case, part=row["part"], clock_ns=row["clock_ns"])
            case_info = {
                **row,
                "model_onnx_sha256": _digest(path),
                "cpp_sha256": _digest(case / "model.cpp"),
                "testbench_sha256": _digest(case / "tb.cpp"),
                "hls_tcl_sha256": _digest(case / "run_hls.tcl"),
                "vivado_tcl_sha256": _digest(case / "vivado/run_vivado.tcl"),
                "functional_sha256": emission.manifest["functional_sha256"],
                "operator_kinds": [op.kind for op in model.operators],
                "input_shape": list(shape),
                "output_shape": list(model.tensor(model.outputs[0]).shape),
            }
            if case_info["cpp_sha256"] != json.loads(
                    (case / "manifest.json").read_text())["cpp_sha256"]:
                raise AssertionError("Source emission/manifest identity mismatch")
            (case / "network_case.json").write_text(
                json.dumps(case_info, sort_keys=True, indent=2) + "\n")

        manifest = {
            "schema": SCHEMA, "part": part, "seed": seed,
            "compiler_checkout_sha": _checkout_revision(),
            "samples_per_case": samples, "numerical_tolerance": tolerance,
            "case_count": len(rows), "network_count": len(networks),
            "cases": rows, "native_tools_executed": False,
            "all_onnx_models_synthetic_untrained": True,
            "evidence": "E1-source-prepared; no Vitis/Vivado invocations",
            "latency_cycles_or_resources_measured": False,
            "ready_to_publish": False,
        }
        (stage / "benchmark_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        stage.rename(destination)
    return manifest


def _inventory(root: Path):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Benchmark root must be an existing ordinary directory")
    manifest_file = _regular(root / "benchmark_manifest.json")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown cross-network benchmark schema")
    entries = manifest.get("cases")
    if not isinstance(entries, list) or len(entries) != manifest.get("case_count"):
        raise ValueError("Benchmark declared case population changed")
    expected = plan(
        networks=tuple(dict.fromkeys(x["network"] for x in entries)),
        mappings=tuple(dict.fromkeys(x["mapping"] for x in entries)),
        precisions=tuple(dict.fromkeys(x["precision"] for x in entries)),
        clocks=tuple(dict.fromkeys(x["clock_ns"] for x in entries)),
        part=manifest["part"],
    )
    if entries != expected:
        raise ValueError("Benchmark ordered case/axis declaration changed")
    for row in entries:
        case = root / "cases" / row["case"]
        if case.is_symlink() or not case.is_dir():
            raise ValueError("Missing or symlinked benchmark case")
        info = json.loads(_regular(case / "network_case.json").read_text())
        if set(info) != _CASE_KEYS or any(info[key] != value for key, value in row.items()):
            raise ValueError("Tampered case identity")
        if _digest(_regular(root / "models" / (row["network"] + ".onnx"))) != info["model_onnx_sha256"]:
            raise ValueError("Changed original ONNX model")
        if _digest(_regular(case / "model.cpp")) != info["cpp_sha256"]:
            raise ValueError("Changed emitted C++")
        files = {
            "tb.cpp": "testbench_sha256",
            "run_hls.tcl": "hls_tcl_sha256",
            "vivado/run_vivado.tcl": "vivado_tcl_sha256",
        }
        for rel, field in files.items():
            if _digest(_regular(case / rel)) != info[field]:
                raise ValueError("Changed source-attested native input: " + rel)
        source_meta = json.loads(_regular(case / "manifest.json").read_text())
        if (source_meta.get("cpp_sha256") != info["cpp_sha256"]
                or source_meta.get("testbench_sha256") != info["testbench_sha256"]
                or source_meta.get("functional_sha256") != info["functional_sha256"]
                or source_meta.get("precision_bits") != row["precision_bits"]
                or source_meta.get("integer_bits") != row["integer_bits"]
                or source_meta.get("clock_ns") != row["clock_ns"]
                or source_meta.get("part") != row["part"]):
            raise ValueError("Prepared numeric/clock/source manifest mismatch")
    return manifest


def tool_preflight(stage: str, executable: str | None = None) -> dict:
    """Pure executable discovery; never launch a proprietary native tool.

    The currently selected Python virtualenv does not automatically source
    AMD settings64.sh or add the Vitis HLS / Vivado binary directories to PATH.
    Prefer the caller's configured command or exact explicit executable.
    """
    if stage not in ("vitis", "vivado"):
        raise ValueError("Native tool stage must be vitis or vivado")
    name = "vitis_hls" if stage == "vitis" else "vivado"
    if executable is not None and (
            not isinstance(executable, str) or not executable.strip()
            or "\x00" in executable):
        raise ValueError("Explicit native executable must be a nonempty path or command")
    requested = executable if executable is not None else name
    found = shutil.which(requested)
    available = bool(found and Path(found).is_file())
    return {
        "schema": "fpgai-research/network-native-tool-preflight/v1",
        "stage": stage,
        "tool": name,
        "requested_executable": requested,
        "resolved_executable": str(Path(found).resolve()) if available else None,
        "available": available,
        "native_tools_executed": False,
        "hint": ("Source the installed AMD Vitis_HLS/Vivado settings64.sh in this "
                 "terminal, or pass --executable /absolute/path/to/" + name
                 + "; no prepared case has been modified by this check."),
    }


def vendor(root: str | Path, *, stage: str, execute: bool = False,
           first: int = 0, limit: int = 1,
           executable: str | None = None) -> dict:
    from fpgai_research.experiments.results import summarize_case
    from fpgai_research.experiments.suite_native_batch import _qualify_hls
    from fpgai_research.toolchains.vitis_hls.runner import run_vitis
    from fpgai_research.toolchains.vivado.runner import run_vivado

    if stage not in ("vitis", "vivado") or type(execute) is not bool:
        raise ValueError("Select vitis/vivado and explicit boolean execute")
    if type(first) is not int or type(limit) is not int or first < 0 or not 1 <= limit <= 32:
        raise ValueError("Select a valid bounded case slice")
    root = Path(root).absolute()
    manifest = _inventory(root)
    rows = manifest["cases"]
    selected = rows[first:first+limit]
    if not selected:
        raise ValueError("No matching benchmark case")
    # Resolve the native executable before entering the case loop, so a
    # missing AMD installation or unsourced environment cannot create
    # incomplete native output or break a prepared, create-only case.
    discovery = tool_preflight(stage, executable)
    if execute and not discovery["available"]:
        raise FileNotFoundError(
            discovery["tool"] + " executable not found: "
            + discovery["requested_executable"] + ". "
            + discovery["hint"])
    output = []
    for row in selected:
        case = root / "cases" / row["case"]
        receipt = case / ("vitis_invocation.json" if stage == "vitis" else "vivado/vivado_invocation.json")
        if receipt.exists() or receipt.is_symlink():
            raise FileExistsError("Native invocation already recorded; never rerun/overwrite: " + str(receipt))
        if stage == "vitis" and (case / "fpgai_hls_project").exists():
            raise FileExistsError("Existing HLS output; do not overwrite original native evidence")
        if stage == "vivado":
            source = summarize_case(str(case))
            if _qualify_hls(source) or source.get("csim_passed") is not True:
                raise ValueError("Vivado requires source-attested E3 HLS + numerically passing CSim")
            if (case / "vivado/research_implementation").exists() or (case / "vivado/reports").exists():
                raise FileExistsError("Existing routed output; do not rerun native Vivado")
        if execute:
            # Existing source/RTL-identity- and tool-version-attested runners.
            receipt_data = (
                run_vitis(case, executable=discovery["resolved_executable"])
                if stage == "vitis"
                else run_vivado(case, executable=discovery["resolved_executable"])
            )
            status = ("INVOCATION_SUCCEEDED_AWAITING_EVIDENCE_AUDIT"
                      if receipt_data.get("completed_successfully") else
                      "INVOCATION_FAILED_AWAITING_DIAGNOSTICS")
        else:
            status = "DRY_RUN_NOT_EXECUTED"
        output.append({"case": row["case"], "network": row["network"],
                       "stage": stage, "status": status})
    return {"schema": "fpgai-research/cross-network-vendor-plan/v1",
            "stage": stage, "executed": execute, "cases": output,
            "tool_available": discovery["available"],
            "resolved_executable": discovery["resolved_executable"],
            "note": "A successful native process is not an E3/E4 result. Run collect for qualified evidence."}


def _number(value):
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def collect(root: str | Path, output: str | Path) -> dict:
    from fpgai_research.experiments.results import summarize_case
    from fpgai_research.experiments.suite_native_batch import _qualify, _qualify_hls
    source = Path(root).absolute()
    manifest = _inventory(source)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Result output is create-only; original observations must be immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in manifest["cases"]:
        case = source / "cases" / item["case"]
        evidence = summarize_case(str(case))
        if evidence["source_sha256"] != json.loads(
                (case / "network_case.json").read_text())["cpp_sha256"]:
            raise ValueError("Native evidence does not match the prepared network source")
        hls_blockers = list(_qualify_hls(evidence))
        e4_blockers = list(_qualify(evidence))
        hls_ok = len(hls_blockers) == 0
        raw = evidence.get("metrics") or {}
        if hls_ok and _number(raw.get("latency_worst")) is None:
            hls_blockers.append("Missing bounded HLS worst-case latency cycles")
            hls_ok = False
        if not hls_ok:
            e4_blockers.append("E4 comparison requires independently qualified E3 HLS cycles/II")
        e4_ok = len(e4_blockers) == 0
        physical = evidence.get("vivado_metrics") or {}
        physical_util = physical.get("utilization") or {}
        physical_timing = physical.get("timing") or {}
        physical_drc = physical.get("drc") or {}
        worst = _number(raw.get("latency_worst")) if hls_ok else None
        row = {
            "case": item["case"], "network": item["network"],
            "family": item["family"], "mapping": item["mapping"],
            "precision": item["precision"], "precision_bits": item["precision_bits"],
            "integer_bits": item["integer_bits"],
            "requested_clock_ns": item["clock_ns"],
            "requested_clock_mhz": round(1000/item["clock_ns"], 6),
            "source_cpp_sha256": evidence["source_sha256"],
            "evidence": evidence["evidence"],
            "csim_passed": evidence.get("csim_passed"),
            "hls_qualified": hls_ok,
            "hls_latency_best_cycles": _number(raw.get("latency_best")) if hls_ok else None,
            "hls_latency_worst_cycles": worst,
            "hls_ii_min": _number(raw.get("ii_min")) if hls_ok else None,
            # Requested-clock arithmetic is only an E3 *nominal proxy*.
            # It is NOT observed RTL/core or board inference latency.
            "hls_nominal_worst_latency_us": (
                round(worst * item["clock_ns"] / 1000, 6) if worst is not None else None),
            "hls_estimated_lut": _number(raw.get("lut")) if hls_ok else None,
            "hls_estimated_ff": _number(raw.get("ff")) if hls_ok else None,
            "hls_estimated_dsp": _number(raw.get("dsp")) if hls_ok else None,
            "hls_estimated_bram18": _number(raw.get("bram_18k")) if hls_ok else None,
            "hls_estimated_uram": _number(raw.get("uram")) if hls_ok else None,
            "e4_attested": evidence["evidence"] == "E4-Vivado-implementation",
            "e4_qualified": e4_ok,
            "postroute_wns_ns": physical_timing.get("wns_ns") if physical else None,
            "postroute_timing_met": physical_timing.get("timing_met") if physical else None,
            "postroute_drc_violations": physical_drc.get("violations") if physical else None,
            "postroute_drc_clean": physical_drc.get("clean") if physical else None,
            "postroute_lut": physical_util.get("lut") if physical else None,
            "postroute_ff": physical_util.get("ff") if physical else None,
            "postroute_dsp": physical_util.get("dsp") if physical else None,
            "postroute_bram": physical_util.get("bram") if physical else None,
            "postroute_uram": physical_util.get("uram") if physical else None,
            "measured_board_latency_us": None,
            "measured_fmax_mhz": None,
            "hls_blockers": hls_blockers,
            "e4_blockers": e4_blockers,
        }
        rows.append(row)

    paired = {}
    for row in rows:
        key = (row["network"], row["precision"], row["requested_clock_ns"])
        paired.setdefault(key, {})[row["mapping"]] = row
    comparisons = []
    for key, pair in paired.items():
        baseline = pair.get("scalar")
        mapped = pair.get("independent")
        qualified = (baseline is not None and mapped is not None
                     and baseline["e4_qualified"] and mapped["e4_qualified"]
                     and baseline["hls_latency_worst_cycles"] is not None
                     and mapped["hls_latency_worst_cycles"] is not None)
        comparisons.append({
            "network": key[0], "precision": key[1], "requested_clock_ns": key[2],
            "both_strict_e4_qualified": qualified,
            "delta_worst_hls_cycles_mapped_minus_scalar": (
                mapped["hls_latency_worst_cycles"] - baseline["hls_latency_worst_cycles"]
                if qualified else None),
            "delta_postroute_lut_mapped_minus_scalar": (
                mapped["postroute_lut"] - baseline["postroute_lut"]
                if qualified else None),
            "note": "Only matched, numerically passing, strict timing/DRC-clean pairs have qualified deltas.",
        })
    best_feasible = []
    for key in sorted({(row["network"], row["mapping"], row["precision"])
                       for row in rows}):
        eligible = [row for row in rows if
                    (row["network"], row["mapping"], row["precision"]) == key
                    and row["e4_qualified"]]
        best_feasible.append({
            "network": key[0], "mapping": key[1], "precision": key[2],
            "fastest_qualified_requested_clock_ns": (
                min(r["requested_clock_ns"] for r in eligible) if eligible else None),
            "highest_qualified_tested_target_mhz": (
                max(r["requested_clock_mhz"] for r in eligible) if eligible else None),
            "fmax_measured_mhz": None,
            "qualification": (
                "CLOCK_TARGET_PASSED_ON_SELECTED_OOC_POINTS"
                if eligible else "NO_STRICT_E4_TARGET_QUALIFIED"),
        })
    result = {
        "schema": RESULT_SCHEMA,
        "source_benchmark_manifest_sha256": _digest(source / "benchmark_manifest.json"),
        "prepared_cases": len(rows), "networks": sorted({r["network"] for r in rows}),
        "source_only_cases": sum(r["evidence"] == "E1-HLS-source" for r in rows),
        "hls_qualified_cases": sum(r["hls_qualified"] for r in rows),
        "strict_e4_qualified_cases": sum(r["e4_qualified"] for r in rows),
        "matched_strict_e4_pairs": sum(c["both_strict_e4_qualified"] for c in comparisons),
        "rows": rows, "within_network_comparisons": comparisons,
        "best_qualified_tested_clock_targets": best_feasible,
        "limitations": [
            "Synthetic untrained models; numerical agreement is not classification accuracy",
            "Cross-network absolute latency is workload-specific; paired deltas only compare same network/clock/precision",
            "HLS cycles and nominal requested-clock latency are synthesis estimates, not hardware measurements",
            "Post-route timing/resources are OOC observations, not bitstream execution or board latency",
            "No exact achieved Fmax is inferred from sparse requested-clock sweeps",
            "Missing measurements remain null/TBD and failed designs remain in the denominator",
        ],
        "ready_to_publish": False,
    }
    with TemporaryDirectory(prefix=".fpgai-network-results-", dir=output.parent) as tmp:
        stage = Path(tmp)
        (stage / "results.json").write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        with (stage / "results.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows({
                name: ("TBD" if row[name] is None else
                       "; ".join(row[name]) if isinstance(row[name], list)
                       else row[name])
                for name in COLUMNS
            } for row in rows)
        md = [
            "# Cross-network FPGAI evidence (untrained synthetic models)", "",
            "| Network | Mapping | Precision | Clock (ns) | Evidence | CSim | HLS cycles (worst) | HLS BRAM18 | Routed LUT | WNS (ns) | Strict E4 |",
            "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in rows:
            keys = ("network","mapping","precision","requested_clock_ns",
                    "evidence","csim_passed","hls_latency_worst_cycles",
                    "hls_estimated_bram18","postroute_lut","postroute_wns_ns",
                    "e4_qualified")
            md.append("| " + " | ".join(
                "TBD" if r[key] is None else str(r[key]) for key in keys) + " |")
        md.extend(["", "## Scope", ""] +
                  ["- " + statement for statement in result["limitations"]] +
                  ["", "No results were invented or filled from a different source revision.", ""])
        (stage / "results.md").write_text("\n".join(md))
        stage.rename(output)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    planp = sub.add_parser("plan")
    preparep = sub.add_parser("prepare")
    for parser in (planp, preparep):
        parser.add_argument("--networks", nargs="+", choices=NETWORKS,
                            default=list(NETWORKS))
        parser.add_argument("--mappings", nargs="+", choices=MAPPINGS,
                            default=list(MAPPINGS))
        parser.add_argument("--precisions", nargs="+", choices=tuple(PRECISIONS),
                            default=list(PRECISIONS))
        parser.add_argument("--clocks", nargs="+", type=float,
                            default=list(CLOCKS_NS))
        parser.add_argument("--part", default="xck26-sfvc784-2LV-c")
    preparep.add_argument("--output", required=True)
    preparep.add_argument("--seed", type=int, default=197)
    preparep.add_argument("--samples", type=int, default=2)
    doctorp = sub.add_parser("doctor", help="Check AMD binary discovery without executing tools")
    doctorp.add_argument("--stage", required=True, choices=("vitis", "vivado"))
    doctorp.add_argument("--executable", help="Optional explicit native executable path")
    runp = sub.add_parser("run")
    runp.add_argument("--root", required=True)
    runp.add_argument("--stage", required=True, choices=("vitis", "vivado"))
    runp.add_argument("--first", type=int, default=0)
    runp.add_argument("--limit", type=int, default=1)
    runp.add_argument("--execute", action="store_true")
    runp.add_argument("--executable", help="Explicit Vitis HLS / Vivado executable")
    collectp = sub.add_parser("collect")
    collectp.add_argument("--root", required=True)
    collectp.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if args.action == "plan":
        rows = plan(networks=args.networks, mappings=args.mappings,
                    precisions=args.precisions, clocks=args.clocks,
                    part=args.part)
        output = {"case_count": len(rows), "cases": rows,
                  "native_tools_executed": False}
    elif args.action == "doctor":
        output = tool_preflight(args.stage, args.executable)
    elif args.action == "prepare":
        output = prepare(
            args.output, networks=args.networks, mappings=args.mappings,
            precisions=args.precisions, clocks=args.clocks, part=args.part,
            seed=args.seed, samples=args.samples)
    elif args.action == "run":
        output = vendor(args.root, stage=args.stage,
                        execute=args.execute, first=args.first, limit=args.limit,
                        executable=args.executable)
    else:
        output = collect(args.root, args.output)
        # CLI prints only the short totals, full source-qualified rows are on disk.
        output = {key: output[key] for key in (
            "schema", "prepared_cases", "networks", "hls_qualified_cases",
            "strict_e4_qualified_cases", "matched_strict_e4_pairs",
        )}
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
