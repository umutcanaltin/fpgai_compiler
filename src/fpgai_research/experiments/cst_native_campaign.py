"""Source-pinned matched HLS compiler IR case preparation, execution and audit.

Prepare is strictly source only. Execute is opt-in and synchronous on a
workstation with original licensed Vitis HLS; never run vendor tools on import.
Study is separate from frozen RL, historical case roots and the default
compiler. Native qualification comes exclusively from existing independently
attested summarize_case, not tool return code or fabricated QoR.
"""
from __future__ import annotations

import argparse
from dataclasses import replace as dataclass_replace
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.composition import compile_composed_architecture
from fpgai_research.compiler.row_interleaved import compile_row_interleaved
from fpgai_research.experiments.cst_row_campaign import model_for
from fpgai_research.experiments.results import summarize_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

SCHEMA = "fpgai-research/cst-ir-original-vitis-experiment/v1"
CASE_RE = re.compile(r"case_[0-9]{3}\Z")
_VARIANTS = ("whole_tensor", "column_major", "row_lifetime")
_DEFAULT_SHAPES = ((2, 3, 5, 2), (3, 3, 5, 2),
                   (3, 4, 7, 3), (5, 3, 7, 2))
# Prespecified mixed-size *exploratory hardware* groups. Source-only cases
# are not hardware measurements; no post-hoc selection by observed HLS QoR.
_PAPER_SHAPES = ((2, 3, 5, 2), (3, 4, 7, 3),
                 (8, 16, 32, 16), (16, 32, 32, 16))


def revision() -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[3]),
                               "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked immutable input: " + str(path))
    return sha256(path.read_bytes()).hexdigest()


def _cases(max_cases: int, *, shape_catalog: str = "pilot"):
    if shape_catalog not in ("pilot", "paper"):
        raise ValueError("shape_catalog must be pilot or paper")
    shape_set = _DEFAULT_SHAPES if shape_catalog == "pilot" else _PAPER_SHAPES
    if type(max_cases) is not int or not 3 <= max_cases <= 24:
        raise ValueError("max_cases must be 3..24")
    # Fully matched triplets, never a truncated unpaired baseline.
    if max_cases % 3:
        raise ValueError("max_cases must be a multiple of three")
    for idx in range(max_cases // 3):
        rows, inner, middle, outer = shape_set[idx % len(shape_set)]
        model = model_for(rows, inner, middle, outer)
        default = ArchitectureRequest()
        mapping_mode = "heterogeneous" if idx % 2 else "uniform"
        overrides = ({
            "a": dataclass_replace(default, pes=2, parallelism=2,
                                   simd=2, serialization=2),
            "b": dataclass_replace(default, pes=3, parallelism=2,
                                   simd=3, serialization=2),
        } if mapping_mode == "heterogeneous" else {})
        architecture = resolve(model, default, overrides)
        x = np.linspace(-0.125, 0.125, rows*inner,
                        dtype=np.float32).reshape(rows, inner)
        for variant in _VARIANTS:
            if variant == "row_lifetime":
                compiled = compile_row_interleaved(architecture)
                emission = compiled.emission
                edge_hash = compiled.evidence["contract_sha256"]
            else:
                layouts = {"h": "column_major"} if variant == "column_major" else {}
                compiled = compile_composed_architecture(
                    architecture, intermediate_layouts=layouts)
                emission = compiled.emission
                edge_hash = compiled.evidence["boundary_sha256"]
            yield idx, mapping_mode, variant, model, x, emission, edge_hash


def prepare(*, output: str | Path, max_cases: int = 12,
            part: str = "xck26-sfvc784-2LV-c",
            clock_ns: float = 32.0, clock_uncertainty_ns: float = 8.64,
            tolerance: float = 0.05,
            shape_catalog: str = "pilot") -> dict:
    """Create-only source-and-testbench package. Does NOT launch Vitis."""
    from math import isfinite
    if not (isfinite(clock_ns) and isfinite(clock_uncertainty_ns)
            and clock_ns > clock_uncertainty_ns >= 0):
        raise ValueError("Clock and uncertainty must be finite and feasible")
    if not (isfinite(tolerance) and tolerance > 0):
        raise ValueError("Tolerance must be positive and finite")
    # Validate candidate domain without writing or vendor calls.
    selected = list(_cases(max_cases, shape_catalog=shape_catalog))
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Create-only CST HLS study already exists")
    dest.mkdir(parents=True)
    records = []
    for case_no, (group, mapping_mode, variant, model, x, emission, edge_sha) in enumerate(selected):
        case_id = f"case_{case_no:03d}"
        directory = dest / case_id
        package = write_vitis_package(
            str(directory), model, emission, x,
            part=part, clock_ns=clock_ns, tolerance=tolerance,
            clock_uncertainty_ns=clock_uncertainty_ns)
        records.append({
            "case": case_id, "group": group, "mapping_mode": mapping_mode,
            "variant": variant,
            "model_shape": list(model.tensor(model.inputs[0]).shape),
            "functional_sha256": package["functional_sha256"],
            "cpp_sha256": package["cpp_sha256"],
            "testbench_sha256": package["testbench_sha256"],
            "tcl_sha256": file_hash(directory / "run_hls.tcl"),
            "edge_contract_sha256": edge_sha,
            "requested_clock_ns": clock_ns,
            "clock_uncertainty_ns": clock_uncertainty_ns,
            "numeric_tolerance": tolerance,
            "original_vitis_run": False,
            "expected_evidence": "E1-generated-source-only",
        })
    groups = {i: [r for r in records if r["group"] == i]
              for i in range(max_cases // 3)}
    for group in groups.values():
        if [r["variant"] for r in group] != list(_VARIANTS):
            raise AssertionError("Native HLS study lacks complete paired variants")
        if len({r["functional_sha256"] for r in group}) != 1:
            raise AssertionError("Paired IR cases do not share the Functional graph")
        if len({r["cpp_sha256"] for r in group}) != 3:
            raise AssertionError("All three HLS materializations must differ")
    manifest = {
        "schema": SCHEMA, "revision": revision(),
        "original_native_tool_executed_by_preparation": False,
        "shape_catalog": shape_catalog,
        "part": part, "clock_ns": clock_ns,
        "clock_uncertainty_ns": clock_uncertainty_ns,
        "tolerance": tolerance, "cases": records,
        "warnings": [
            "HLS synthesis result requires original Vitis and independent audit",
            "HLS clock target is not achieved post-route Fmax",
            "Source row-buffer length is not measured hardware BRAM",
            "This campaign is unrelated to frozen RL cases",
        ],
    }
    (dest / "compiler_ir_native_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_prepared(study: str | Path, *, require_current_revision: bool = True):
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Native study must be an actual directory")
    path = root / "compiler_ir_native_manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Original study manifest missing")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unexpected study manifest schema")
    records = manifest.get("cases")
    if not isinstance(records, list) or not records or len(records) > 24 or len(records) % 3:
        raise ValueError("Invalid paired study case count")
    if require_current_revision and manifest.get("revision") != revision():
        raise ValueError("Original source revision mismatch; do not execute stale study")
    seen = set()
    summaries = []
    for row in records:
        case_id = row.get("case")
        if not isinstance(case_id, str) or not CASE_RE.fullmatch(case_id) or case_id in seen:
            raise ValueError("Invalid or duplicate native case")
        seen.add(case_id)
        case = root / case_id
        if case.is_symlink() or not case.is_dir():
            raise ValueError("Missing or symlinked native case")
        for name, field in (("model.cpp", "cpp_sha256"),
                            ("tb.cpp", "testbench_sha256"),
                            ("run_hls.tcl", "tcl_sha256")):
            if file_hash(case / name) != row[field]:
                raise ValueError("Prepared HLS input changed: " + case_id + "/" + name)
        summary = summarize_case(str(case))
        if summary["source_sha256"] != row["cpp_sha256"]:
            raise ValueError("Unattributed C++ source in native study")
        summaries.append((row, case, summary))
    return manifest, summaries


def execute(study: str | Path, *, execute_native: bool = False,
            max_calls: int = 3, vitis_executable: str = "vitis_hls",
            timeout_seconds: int = 7200) -> dict:
    """Opt-in bounded real Vitis work; never silently resumes/re-runs cases."""
    if type(execute_native) is not bool or type(max_calls) is not int:
        raise ValueError("Explicit boolean and integral native budget required")
    if not 1 <= max_calls <= 24:
        raise ValueError("max_calls must be 1..24")
    manifest, cases = verify_prepared(study)
    if not execute_native:
        return {
            "schema": "fpgai-research/cst-ir-native-dry-run/v1",
            "original_vitis_calls": 0,
            "planned_cases": [row["case"] for row, _path, _s in cases[:max_calls]],
            "requires_explicit_execute_native": True,
        }
    if not manifest.get("revision"):
        raise ValueError("Native execution requires pinned Git revision")
    completed = []
    for row, case, prior in cases[:max_calls]:
        if (case / "vitis_invocation.json").exists() or (case / "logs").exists():
            completed.append({"case": row["case"], "status": "ALREADY_HAS_NATIVE_FILES"})
            continue
        receipt = run_vitis(case, executable=vitis_executable,
                            timeout_seconds=timeout_seconds)
        # Tool exit 0 is not E2/E3 qualification. Re-read independently.
        current = summarize_case(str(case))
        completed.append({
            "case": row["case"],
            "invocation_success": receipt.get("completed_successfully"),
            "native_evidence": current.get("evidence"),
            "csim_passed": current.get("csim_passed"),
            "hls_worst_cycles": (current.get("metrics") or {}).get("latency_worst"),
            "warnings": current.get("warnings"),
        })
    return {
        "schema": "fpgai-research/cst-ir-native-execution/v1",
        "original_vitis_calls_started": sum(
            "invocation_success" in c for c in completed),
        "cases": completed,
        "note": "Only independent read-only audit may establish numeric/HLS qualification",
    }


def audit(study: str | Path) -> dict:
    """Read-only synthesis evidence: never fabricate unavailable measurements."""
    # Historical scientific audits must remain readable after later code
    # releases, while native execution always requires the original source.
    manifest, cases = verify_prepared(study, require_current_revision=False)
    outcomes = []
    for row, case, result in cases:
        metrics = result.get("metrics") or {}
        raw_cycles = metrics.get("latency_worst")
        # Native report parser preserves XML integers as strings. Reject
        # unbounded and malformed intervals rather than assuming latency.
        cycles = (int(raw_cycles) if isinstance(raw_cycles, str)
                  and raw_cycles.isdecimal() else raw_cycles
                  if type(raw_cycles) is int and raw_cycles >= 0 else None)
        e3 = result.get("evidence") in (
            "E3-HLS-synthesis-report", "E4-Vivado-implementation")
        # Source-attested CSim+synthesis is not achieved HLS target timing.
        qualified = bool(e3 and result.get("csim_passed") is True
                         and cycles is not None)
        outcomes.append({
            "case": row["case"], "group": row["group"],
            "mapping_mode": row["mapping_mode"],
            "variant": row["variant"],
            "row_block": row.get("row_block"),
            "declared_scratch_elements": row.get("declared_scratch_elements"),
            "operator_count": row.get("operator_count"), "source_sha256": row["cpp_sha256"],
            "native_evidence": result.get("evidence"),
            "csim_passed": result.get("csim_passed"),
            "hls_csim_attested_with_bounded_latency": qualified,
            "latency_worst_cycles": cycles if e3 else None,
            "requested_period_scaled_hls_latency_ns": (
                cycles * row["requested_clock_ns"] if qualified else None),
            "timing_target_qualification_unverified": True,
            "lut": metrics.get("lut") if e3 else None,
            "ff": metrics.get("ff") if e3 else None,
            "dsp": metrics.get("dsp") if e3 else None,
            "bram_18k": metrics.get("bram_18k") if e3 else None,
            "warnings": result.get("warnings", []),
        })
    return {
        "schema": "fpgai-research/cst-ir-original-hls-audit/v1",
        "source_revision": manifest.get("revision"),
        "cases": outcomes,
        "cases_with_native_csim_and_bounded_csynth_cycles": sum(
            r["hls_csim_attested_with_bounded_latency"] for r in outcomes),
        "hls_clock_feasibility_established": False,
        "no_physical_fmax_claim": True,
        "read_only": True,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "dry-run", "execute", "audit"))
    p.add_argument("--output", help="New source-only root for prepare")
    p.add_argument("--study", help="Existing prepared source root")
    p.add_argument("--max-cases", type=int, default=12)
    p.add_argument("--shape-catalog", choices=("pilot", "paper"), default="pilot")
    p.add_argument("--max-calls", type=int, default=3)
    p.add_argument("--part", default="xck26-sfvc784-2LV-c")
    p.add_argument("--clock-ns", type=float, default=32.)
    p.add_argument("--clock-uncertainty-ns", type=float, default=8.64)
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--execute-native", action="store_true")
    p.add_argument("--vitis-executable", default="vitis_hls")
    p.add_argument("--timeout-seconds", type=int, default=7200)
    a = p.parse_args(argv)
    if a.action == "prepare":
        if not a.output:
            p.error("prepare requires --output")
        result = prepare(output=a.output, max_cases=a.max_cases,
                         part=a.part, clock_ns=a.clock_ns,
                         clock_uncertainty_ns=a.clock_uncertainty_ns,
                         tolerance=a.tolerance, shape_catalog=a.shape_catalog)
    else:
        if not a.study:
            p.error("dry-run, execute and audit require --study")
        if a.action == "audit":
            result = audit(a.study)
        else:
            result = execute(a.study, execute_native=(
                a.action == "execute" and a.execute_native),
                max_calls=a.max_calls, vitis_executable=a.vitis_executable,
                timeout_seconds=a.timeout_seconds)
            if a.action == "execute" and not a.execute_native:
                p.error("execute requires --execute-native")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
