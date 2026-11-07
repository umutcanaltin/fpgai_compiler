"""R4k: NEW create-only schedule-matched H-storage source experiment.

Original frozen R4d source + Vitis outputs are checked read-only before
creating any R4k result. Both new arms have the SAME source Phi event
schedule and numerical model; only full versus row storage capacities
and source scratch addressing differ. GCC real runtime E1 is executed
on both. Vitis packages are emitted, NEVER executed here.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    emit_row_retained_fanout_cpp,
)
from fpgai_research.codegen.hls.r4k_full_storage_row_schedule_cpp import (
    emit_full_storage_row_schedule_cpp,
)
from fpgai_research.experiments import cst_r4a_row_retention_study as r4a
from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments import cst_r4e_numeric_format_native as r4e
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
from fpgai_research.validation.r4k_matched_storage_gcc import verify_matched_gcc

SCHEMA = "fpgai-research/R4k-controlled-source-memory-row-schedule-E1/v1"
FORMAT = (32, 6)
ARMS = ("full_storage", "row_storage")


def _sha(file: Path) -> str:
    if file.is_symlink() or not file.is_file():
        raise ValueError("R4k original source file absent or symlinked")
    return sha256(file.read_bytes()).hexdigest()


def plan() -> dict:
    graph = r4a.CASES[r4d.CASE_INDEX]
    arch = r4a._case(graph, precision_bits=FORMAT[0],
                     integer_bits=FORMAT[1])
    original = emit_row_retained_fanout_cpp(arch)
    control = emit_full_storage_row_schedule_cpp(arch)
    if (original.manifest["source_event_order_sha256"]
            != control.manifest["source_event_order_sha256"]
            or original.manifest["architecture_sha256"]
            != control.manifest["architecture_sha256"]
            or original.manifest["functional_sha256"]
            != control.manifest["functional_sha256"]
            or (control.manifest["source_scratch_array_elements"] -
                original.manifest["source_scratch_array_elements"])
                != (graph[1][0] - 1) * graph[1][2]):
        raise ValueError("R4k source schedule/model/only-storage match failed")
    return {
        "schema": SCHEMA,
        "source_graph": [graph[0], list(graph[1]), graph[2]],
        "numeric_format": list(FORMAT),
        "source_schedule_sha256": original.manifest[
            "source_event_order_sha256"],
        "source_row_original_sha256": original.manifest["source_cpp_sha256"],
        "source_full_storage_sha256": control.manifest["source_cpp_sha256"],
        "source_scratch_elements_saved":
            (control.manifest["source_scratch_array_elements"] -
             original.manifest["source_scratch_array_elements"]),
        "original_R4d_E2_E3_32_6_required": True,
        "matched_source_arms": len(ARMS),
        "part": r4a.PART,
        "clock_ns": r4a.CLOCK_NS,
        "uncertainty_ns": r4a.UNCERTAINTY_NS,
        "csim_abs_tolerance": r4a.CSIM_ABS_TOLERANCE,
        "E1_GCC_required": True,
        "native_vendor_calls": 0,
        "physical_E4": None,
    }


def _read_only_original_preflight(study: Path, original_study: Path,
                                  plan_: dict) -> dict:
    # The R4e original audit checks original study source revision,
    # native receipts and source/CSim/HLS gate. No vendor execution.
    native = r4e.audit_native(study=study, original_study=original_study)
    records = native.get("observations", [])
    match = {item["arm"]: item for item in records
             if item.get("format") == list(FORMAT)}
    if (native.get("source_attested_arms") != 6
            or set(match) != {"full_H", "row_H"}
            or any(not match[a].get("E2_qualified")
                   or not match[a].get("E3_qualified") for a in match)):
        raise ValueError("R4k original source-attested 32/6 matched E2/E3 gate missing")
    old_row = study / "format_32_06" / "row_H" / "source.cpp"
    if _sha(old_row) != plan_["source_row_original_sha256"]:
        raise ValueError("R4k original 32/6 row schedule/source changed")
    original_tcl = study / "format_32_06" / "row_H" / "vitis_source" / "run_hls.tcl"
    return {
        "source_revision": native["source_revision"],
        "original_R4a_source_revision": native["original_R4a_source_revision"],
        "original_row_cpp_sha256": _sha(old_row),
        "original_tcl_sha256": _sha(original_tcl),
        "original_R4d_native_passed_32_6": True,
    }


def run(*, original_r4d: str | Path, original_r4a: str | Path,
        output: str | Path) -> dict:
    config = plan()
    study, frozen = Path(original_r4d), Path(original_r4a)
    base = _read_only_original_preflight(study, frozen, config)
    dest = Path(output)
    resolved = dest.resolve()
    if any(resolved == root or root in resolved.parents for root in (
            study.resolve(), frozen.resolve())):
        raise ValueError("R4k output cannot be inside original evidence")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R4k result directory must be create-only")
    rev = r4a._git_revision()
    graph = r4a.CASES[r4d.CASE_INDEX]
    arch = r4a._case(graph, precision_bits=FORMAT[0],
                     integer_bits=FORMAT[1])
    emissions = {
        "full_storage": emit_full_storage_row_schedule_cpp(arch),
        "row_storage": emit_row_retained_fanout_cpp(arch),
    }
    x = r4a._sample(graph[1][0], graph[1][1])
    dest.mkdir(parents=True, exist_ok=False)
    try:
        result_e1 = verify_matched_gcc(arch, runtime_input=x,
                                      trace_dir=dest)
        for arm in ARMS:
            folder = dest / arm
            folder.mkdir()
            emission = emissions[arm]
            (folder / "source.cpp").write_text(emission.cpp)
            (folder / "manifest.json").write_text(
                json.dumps(emission.manifest, indent=2,
                           sort_keys=True, allow_nan=False) + "\n")
            pkg = folder / "vitis_source"
            out = write_vitis_package(
                str(pkg), arch.functional,
                r4a._vitis_emission(arch, emission), x,
                part=r4a.PART, clock_ns=r4a.CLOCK_NS,
                clock_uncertainty_ns=r4a.UNCERTAINTY_NS,
                tolerance=r4a.CSIM_ABS_TOLERANCE,
            )
            if (_sha(pkg / "model.cpp")
                    != emission.manifest["source_cpp_sha256"]
                    or _sha(pkg / "tb.cpp") != out["testbench_sha256"]
                    or _sha(pkg / "run_hls.tcl") != base["original_tcl_sha256"]):
                raise ValueError("R4k Vitis package/source/clock attestation mismatch")
        p1, p2 = (dest / arm / "vitis_source" for arm in ARMS)
        if _sha(p1 / "tb.cpp") != _sha(p2 / "tb.cpp"):
            raise ValueError("R4k matched source arms differ in numerical testbench")
        summary = {
            "schema": SCHEMA,
            "source_revision": rev,
            "configuration": config,
            "original_preflight": base,
            "E1": result_e1,
            "source_only_E1_verified": True,
            "HLS_E2_run": False,
            "HLS_E3_run": False,
            "physical_E4_run": False,
            "new_vendor_Vitis_calls": 0,
            "new_vendor_Vivado_calls": 0,
            "scientific_limits": [
                "Source event order matched exactly; hardware cycles/II unmeasured.",
                "Only scratch storage mapping/capacity changed relative to the row source.",
                "No new fixed-point CSim or HLS synthesis executed.",
                "Original HLS 13 BRAM tie is not QoR of this new controlled pair.",
                "No physical FPGA timing, routing, utilization or power evidence.",
                "Original frozen studies remain unchanged.",
            ],
        }
        (dest / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2,
                       allow_nan=False) + "\n")
        digests = {
            str(p.relative_to(dest)): _sha(p)
            for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != "artifact_sha256.json"
        }
        (dest / "artifact_sha256.json").write_text(
            json.dumps(digests, indent=2, sort_keys=True) + "\n")
        return summary
    except Exception as err:
        (dest / "FAILED.json").write_text(json.dumps({
            "error_type": type(err).__name__,
            "error_message": str(err)[:4000],
            "source_revision": rev,
            "original_sources_modified": False,
        }, sort_keys=True, indent=2) + "\n")
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-r4d")
    parser.add_argument("--original-r4a")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.output is None:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return
    if not args.original_r4d or not args.original_r4a:
        parser.error("--original-r4d and --original-r4a required for E1 execution")
    result = run(original_r4d=args.original_r4d,
                 original_r4a=args.original_r4a, output=args.output)
    print(json.dumps({
        "source_revision": result["source_revision"],
        "E1_status": result["E1"]["status"],
        "source_only": True,
        "new_native_Vitis_calls": 0,
        "output": args.output,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
