"""C1 finite source-events to ACTUAL generated C++ E1 refinement bridge.

Use the author's immutable 54-profile / 486-event E0 study as an
independently SHA-pinned *population anchor*, then generate and EXECUTE
each of the same mapped fanout models in real GCC C++ using disjoint
and last-reader-reuse source scratch layouts. Real generated address
witnesses and ALL intermediate/final numerical values are checked
against the independent host reference by the existing R2h validator.

This is a NEW source-revision-specific E1 experiment, never an extension
of the frozen E0 study; do not infer cycle-accurate Φ, RTL RAM ports,
Vivado BRAM, achieved Fmax or board end-to-end latency.
NO Vitis/Vivado or GitHub Actions.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    emit_allocated_fanout_cpp,
)
from fpgai_research.experiments.cst_paper_c1_seeded_metamorphic import (
    _architecture, configurations,
)
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import (
    _git_revision,
)
from fpgai_research.model_ir.fanout_contract import (
    verify_fanout_events, verify_fanout_issue,
)
from fpgai_research.validation.r2h_allocated_fanout_cpp import (
    verify_allocated_gcc,
)

SCHEMA = "fpgai-research/C1-heldout-typed-event-to-generated-C++-E1/v1"
ORIGINAL_C1_SOURCE_REVISION = "637f2745721b32d6f5a7e02c4659b0623bc9eee7"
ORIGINAL_C1_REPORT_SHA256 = (
    "8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978"
)
ORIGINAL_C1_SCHEMA = "fpgai-research/C1-seeded-heldout-metamorphic-source-events/v1"
LAYOUTS = ("disjoint", "per_element_optimal")
FULL_PROFILES = 54


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"C1 original source artifact missing or symlinked: {path.name}")
    return sha256(path.read_bytes()).hexdigest()


def _key(seed, readers, shape, mapping):
    return (seed, readers, tuple(shape), mapping)


def _source_trace_sha(events) -> str:
    return sha256(json.dumps(
        [e.__dict__ for e in events], sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_original_c1(study: str | Path) -> dict:
    root = Path(study)
    report = root / "report.json"
    manifest = root / "artifact_sha256.json"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Original C1 E0 study missing or symlinked")
    digest = _sha(report)
    if digest != ORIGINAL_C1_REPORT_SHA256:
        raise ValueError("Original C1 E0 report SHA differs from author verification")
    if json.loads(manifest.read_text()) != {"report.json": digest}:
        raise ValueError("Original C1 E0 SHA inventory does not match report")
    data = json.loads(report.read_text())
    if (data.get("schema") != ORIGINAL_C1_SCHEMA
            or data.get("experiment_source_revision") != ORIGINAL_C1_SOURCE_REVISION
            or data.get("profiles") != FULL_PROFILES
            or data.get("traces") != FULL_PROFILES * 9
            or data.get("value_oracle_disagreements") != 0
            or data.get("issue_oracle_disagreements") != 0):
        raise ValueError("Original C1 E0 population or source revision changed")
    profiles = {}
    for row in data.get("source_reports", []):
        key = _key(row["seed"], row["readers"], row["shape"], row["mapping"])
        mutations = row.get("mutations", [])
        if key in profiles or len(mutations) != 9 or mutations[0].get(
                "mutation") != "canonical":
            raise ValueError("Original C1 E0 population ambiguous")
        profiles[key] = mutations[0]["event_trace_sha256"]
    if len(profiles) != FULL_PROFILES:
        raise ValueError("Original C1 E0 profile cohort incomplete")
    return profiles


def _assert_source_parent(case, original_trace_sha):
    arch, contract, original_events, offsets = _architecture(case)
    if (verify_fanout_events(arch, offsets, original_events).get("status")
            != "VERIFIED_E0" or
            verify_fanout_issue(arch, original_events).get("status")
            != "VERIFIED_E0"):
        raise ValueError("New E1 compiler source case failed E0 contract")
    trace_sha = _source_trace_sha(original_events)
    if trace_sha != original_trace_sha:
        raise ValueError(
            "C1 original E0 canonical event changed between original and E1 revisions")
    return arch, trace_sha


def _evaluate_one(case, original_sha: str, folder: Path, *, cxx: str):
    arch, event_sha = _assert_source_parent(case, original_sha)
    _, readers, m, k, n, q, mapping = case
    x = _input(m, k)
    arms = {}
    for layout in LAYOUTS:
        emission = emit_allocated_fanout_cpp(arch, layout=layout)
        source_path = folder / f"{layout}.cpp"
        source_path.write_text(emission.cpp)
        actual_sha = _sha(source_path)
        if actual_sha != emission.manifest.get("source_cpp_sha256"):
            raise ValueError("Generated actual C++ source SHA disagrees with manifest")
        stdout_path = folder / f"{layout}.stdout.txt"
        receipt = verify_allocated_gcc(
            arch, emission, runtime_input=x,
            cxx=cxx, original_trace_output=stdout_path)
        if (receipt.get("status")
                != "R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED"
                or receipt.get("source_cpp_sha256") != actual_sha
                or receipt.get("source_layout") != layout
                or receipt.get("original_GCC_compilations") != 1
                or receipt.get("original_GCC_executions") != 1
                or receipt.get("original_Vitis_calls") != 0
                or receipt.get("original_Vivado_calls") != 0
                or not stdout_path.is_file()):
            raise AssertionError("Actual generated C++ E1 receipt not qualified")
        arms[layout] = {
            "status": receipt["status"],
            "generated_cpp_sha256": actual_sha,
            "actual_stdout_sha256": _sha(stdout_path),
            "source_scratch_unit_slots": receipt["source_scratch_unit_slots"],
            "original_actual_address_witness_count":
                receipt["original_actual_address_witness_count"],
            "intermediate_numeric_max_absolute_error":
                receipt["original_all_intermediate_numeric_max_absolute_error"],
            "final_numeric_max_absolute_error":
                receipt["original_final_numeric_max_absolute_error"],
            "runtime_source_event_sha256":
                receipt["original_runtime_source_event_sha256"],
            "functional_sha256": receipt["functional_sha256"],
            "architecture_sha256": receipt["architecture_sha256"],
            "input_sha256": receipt["original_input_sha256"],
            "original_GCC_compilations": 1,
            "original_GCC_executions": 1,
        }
    full = arms["disjoint"]
    reused = arms["per_element_optimal"]
    for field in ("functional_sha256", "architecture_sha256", "input_sha256"):
        if full[field] != reused[field]:
            raise AssertionError(f"Matched E1 layouts differed on source {field}")
    if reused["source_scratch_unit_slots"] > full["source_scratch_unit_slots"]:
        raise AssertionError("Last-reader source allocator exceeded disjoint slot count")
    return {
        "seed": case[0], "readers": readers, "shape": [m, k, n, q],
        "mapping": mapping,
        "original_c1_e0_event_sha256": event_sha,
        "new_e1_reuse_source_slot_savings":
            full["source_scratch_unit_slots"] - reused["source_scratch_unit_slots"],
        "layouts": arms,
        "claims": "bounded actual generated GCC scratch/Numpy source only",
    }


def _index_files(root: Path) -> dict[str, str]:
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name not in ("artifact_sha256.json", "FAILED.json")
    )
    return {p.relative_to(root).as_posix(): _sha(p) for p in paths}


def run(*, original_c1_study: str | Path, output: str | Path,
        cxx: str = "g++", quick: bool = False) -> dict:
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("C1 E1 matched source study is strictly create-only")
    if shutil.which(cxx) is None:
        raise FileNotFoundError("Real host GNU C++ compiler required for C1 E1")
    cohort = load_original_c1(original_c1_study)
    cases = configurations()
    if len(cases) != FULL_PROFILES:
        raise ValueError("C1 E0 source generator no longer produces 54 profiles")
    expected = {_key(c[0], c[1], c[2:6], c[6]) for c in cases}
    if expected != set(cohort):
        raise ValueError("E1 candidate profiles differ from original C1 E0 cohort")
    revision = _git_revision()
    if re.fullmatch("[0-9a-f]{40}", revision) is None:
        raise ValueError("New C1 E1 experiment Git source revision unknown")
    if quick:
        cases = cases[:2]
    dest.mkdir(parents=True, exist_ok=False)
    case_records = []
    try:
        for index, case in enumerate(cases):
            case_dir = dest / f"case_{index:03d}"
            case_dir.mkdir()
            original_sha = cohort[_key(case[0], case[1], case[2:6], case[6])]
            receipt = _evaluate_one(case, original_sha, case_dir, cxx=cxx)
            (case_dir / "case.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n")
            case_records.append(receipt)
        metrics = {
            "schema": SCHEMA,
            "experiment_source_revision": revision,
            "original_E0_source_revision": ORIGINAL_C1_SOURCE_REVISION,
            "original_E0_report_sha256": ORIGINAL_C1_REPORT_SHA256,
            "status": "NEW_E1_EXECUTED_GCC_SOURCE_REFINEMENT_PAIRED_WITH_OLD_E0",
            "smoke_only": quick,
            "original_c1_e0_profiles_attested": FULL_PROFILES,
            "profiles": len(cases),
            "layouts_per_profile": len(LAYOUTS),
            "qualified_GCC_compilations": len(cases)*len(LAYOUTS),
            "qualified_GCC_executions": len(cases)*len(LAYOUTS),
            "all_original_C1_event_hashes_matched": True,
            "actual_source_slot_savings_count": sum(
                c["new_e1_reuse_source_slot_savings"] > 0 for c in case_records),
            "actual_source_slot_tie_count": sum(
                c["new_e1_reuse_source_slot_savings"] == 0 for c in case_records),
            "cases": case_records,
            "native_Vitis_calls": 0,
            "native_Vivado_calls": 0,
            "physical_compute_latency": None,
            "board_qualified": False,
            "limitations": [
                "New E1 source revision differs from original E0 source revision",
                "Both verifiers still share finite admitted model/topology",
                "C++ trace and NumPy conformance do not prove arbitrary inputs",
                "GCC host-float does NOT establish fixed-point Vitis equivalence",
                "Neither C++ issue traces nor source scalar slots prove RTL memory ports or cycles",
                "No FPGA latency, throughput or DRC qualification",
            ],
        }
        (dest / "summary.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
        index = _index_files(dest)
        (dest / "artifact_sha256.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        failed = {
            "schema": SCHEMA + "/failed-original-attempt",
            "experiment_source_revision": revision,
            "original_E0_report_sha256": ORIGINAL_C1_REPORT_SHA256,
            "completed_profiles_before_failure": len(case_records),
            "smoke_only": quick,
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "native_Vitis_calls": 0,
            "native_Vivado_calls": 0,
            "status": "PRESERVE_ORIGINAL_SOURCE_COUNTEREXAMPLE",
        }
        (dest / "FAILED.json").write_text(
            json.dumps(failed, indent=2, sort_keys=True) + "\n")
        raise
    return {
        "schema": SCHEMA, "experiment_source_revision": revision,
        "profiles": len(cases),
        "original_c1_event_hashes_matched": True,
        "gcc_compilations": len(cases) * len(LAYOUTS),
        "gcc_executions": len(cases) * len(LAYOUTS),
        "source_slot_savings_count": metrics["actual_source_slot_savings_count"],
        "source_slot_tie_count": metrics["actual_source_slot_tie_count"],
        "no_vendor_calls": True,
        "physical_compute_latency": None,
        "output": str(dest),
    }


def audit(study: str | Path) -> dict:
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("C1 E1 saved source study missing or symlinked")
    summary = root / "summary.json"
    inventory = root / "artifact_sha256.json"
    if summary.is_symlink() or inventory.is_symlink():
        raise ValueError("C1 E1 original evidence file symlink not allowed")
    expected = json.loads(inventory.read_text())
    actual = _index_files(root)
    if expected != actual:
        raise ValueError("C1 E1 original generated source or receipts changed")
    data = json.loads(summary.read_text())
    n = data.get("profiles")
    if (data.get("schema") != SCHEMA
            or data.get("status")
            != "NEW_E1_EXECUTED_GCC_SOURCE_REFINEMENT_PAIRED_WITH_OLD_E0"
            or not isinstance(n, int) or n < 1
            or (not data.get("smoke_only") and n != FULL_PROFILES)
            or data.get("original_E0_report_sha256") != ORIGINAL_C1_REPORT_SHA256
            or data.get("original_E0_source_revision") != ORIGINAL_C1_SOURCE_REVISION
            or re.fullmatch("[0-9a-f]{40}",
                            data.get("experiment_source_revision", "")) is None
            or data.get("qualified_GCC_compilations") != n*len(LAYOUTS)
            or data.get("qualified_GCC_executions") != n*len(LAYOUTS)
            or len(data.get("cases", [])) != n
            or data.get("native_Vitis_calls") != 0
            or data.get("native_Vivado_calls") != 0
            or data.get("physical_compute_latency") is not None
            or data.get("board_qualified") is not False
            or data.get("all_original_C1_event_hashes_matched") is not True):
        raise ValueError("C1 E1 original evidence source/population not qualified")
    for i, case in enumerate(data["cases"]):
        folder = root / f"case_{i:03d}"
        if (not folder.is_dir() or folder.is_symlink()
                or set(case.get("layouts", {})) != set(LAYOUTS)):
            raise ValueError("C1 E1 case source/layout missing")
        if json.loads((folder / "case.json").read_text()) != case:
            raise ValueError("C1 E1 case receipt differs from study summary")
        for layout in LAYOUTS:
            arm = case["layouts"][layout]
            if (arm["status"] !=
                    "R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED"
                    or arm["generated_cpp_sha256"] !=
                    _sha(folder / f"{layout}.cpp")
                    or arm["actual_stdout_sha256"] !=
                    _sha(folder / f"{layout}.stdout.txt")):
                raise ValueError("C1 E1 generated source/witness audit failed")
    return {
        "schema": SCHEMA,
        "source_revision": data["experiment_source_revision"],
        "original_e0_revision": ORIGINAL_C1_SOURCE_REVISION,
        "original_e0_sha256": ORIGINAL_C1_REPORT_SHA256,
        "profiles": n,
        "source_gcc_compilations": data["qualified_GCC_compilations"],
        "source_gcc_executions": data["qualified_GCC_executions"],
        "source_savings_profiles": data["actual_source_slot_savings_count"],
        "source_tie_profiles": data["actual_source_slot_tie_count"],
        "original_manifest_sha256": _sha(inventory),
        "source_artifact_count": len(actual),
        "new_compiler_or_vendor_calls": 0,
        "physical_compute_latency": None,
        "board_qualified": False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output")
    action.add_argument("--study")
    parser.add_argument("--original-c1-study")
    parser.add_argument("--quick", action="store_true",
                        help="Two-profile real GCC smoke only, not a paper cohort")
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args(argv)
    if args.study:
        out = audit(args.study)
    else:
        if not args.original_c1_study:
            parser.error("--original-c1-study required for new E1 source study")
        out = run(original_c1_study=args.original_c1_study,
                  output=args.output, quick=args.quick, cxx=args.cxx)
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
