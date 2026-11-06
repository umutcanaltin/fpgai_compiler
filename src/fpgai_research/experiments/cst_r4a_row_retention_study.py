"""R4a prospective first-paper row-retained intermediate source experiment.

A new create-only matched GCC E1 + Vitis-source-preparation study.
Every arm uses the existing R2g arithmetic/issue decoder; the
intervention only changes the row materialization/source memory
schedule. Historical frozen native cohorts are neither read nor written.
NO VITIS / VIVADO execution occurs here.

The profiles include small functional controls and larger source-memory
pressure controls, with both 2 and 4 fanout readers. Even if successful,
E1+packaging does NOT establish HLS BRAM/clock benefits.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2g_fanout_cpp import emit_fanout_cpp
from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    emit_row_retained_fanout_cpp,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
from fpgai_research.validation.r2g_fanout_cpp import verify_fanout_gcc
from fpgai_research.validation.row_retained_fanout_cpp import (
    verify_row_retained_gcc,
)

SCHEMA = "fpgai-research/R4a-matched-row-retention-E1-and-HLS-source/v1"
PART = "xck26-sfvc784-2LV-c"
CLOCK_NS = 12.0
UNCERTAINTY_NS = 1.5
CSIM_ABS_TOLERANCE = 0.02
SMALL = (
    (2, (2, 3, 3, 4), "scalar"),
    (3, (2, 2, 4, 4), "mixed"),
    (4, (3, 2, 3, 4), "mixed"),
)
MEDIUM = (
    (2, (16, 4, 16, 8), "mixed"),
    (4, (16, 4, 16, 4), "mixed"),
)
CASES = SMALL + MEDIUM

# IMPORTANT: the canonical R2f fixture and fanout contract admit each
# static dimension only in [1, 16]. Expanding that domain requires
# its own model/contract/verification study, not a silent benchmark edit.
# The four-reader large case was corrected before any original R4a run.


def _git_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout
    if status.strip():
        raise RuntimeError("R4a requires clean tracked source for attribution")
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True,
        text=True, check=True, timeout=15,
    ).stdout.strip()
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise ValueError("R4a exact source commit unavailable")
    return rev


def _case(config):
    readers, (m, k, n, q), policy = config
    model = sample_fanout(readers, m=m, k=k, n=n, q=q)
    default = ArchitectureRequest()
    kwargs = {}
    if policy == "mixed":
        kwargs["reader0"] = replace(default, pes=2)
        kwargs["reader1"] = replace(default, simd=2)
        for i in range(2, readers):
            kwargs[f"reader{i}"] = replace(
                default, pes=2 if i % 2 == 0 else 1, simd=2)
    elif policy != "scalar":
        raise ValueError("Unsupported prespecified Phi policy")
    return resolve(model, default, kwargs)


def _sample(m: int, k: int) -> np.ndarray:
    """Deterministic nonzero signed input, identical in both arms."""
    return np.asarray(
        [[((row * 3 + col * 5) % 17 - 8) / 16.0
          for col in range(k)] for row in range(m)],
        dtype=np.float32,
    )


def _vitis_emission(arch, emission) -> Emission:
    """Use the canonical Vitis package API with explicit source lineage."""
    point = arch.points[0].effective
    digest = sha256(emission.cpp.encode()).hexdigest()
    if emission.manifest.get("source_cpp_sha256") != digest:
        raise ValueError("Prospective E1 emitter SHA mismatch")
    return Emission(emission.cpp, {
        **emission.manifest,
        "cpp_sha256": digest,
        "precision_bits": point.precision_bits,
        "integer_bits": point.integer_bits,
    })


def plan(*, profile: str) -> dict:
    if profile not in ("quick", "paper"):
        raise ValueError("R4a profile must be quick or paper")
    selected = SMALL if profile == "quick" else CASES
    return {
        "schema": SCHEMA,
        "profile": profile,
        "prespecified_graphs": len(selected),
        "matched_arms": 2 * len(selected),
        "source_conformance": "E1_HOST_GCC_FLOAT32",
        "HLS_package_only": True,
        "native_vendor_calls": 0,
        "part": PART, "clock_ns": CLOCK_NS,
        "clock_uncertainty_ns": UNCERTAINTY_NS,
        "csim_absolute_tolerance": CSIM_ABS_TOLERANCE,
        "configs": [[r, list(shape), policy]
                    for r, shape, policy in selected],
    }


def run(*, output: str | Path, profile: str = "quick") -> dict:
    conf = plan(profile=profile)
    revision = _git_revision()
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R4a original output folder must be create-only")
    root.mkdir(parents=True, exist_ok=False)
    cases = SMALL if profile == "quick" else CASES
    rows = []
    try:
        for i, item in enumerate(cases):
            arch = _case(item)
            readers, (m, k, n, q), policy = item
            x = _sample(m, k)
            case_folder = root / f"case_{i:03d}"
            case_folder.mkdir()
            source_emissions = {
                "full_H": emit_fanout_cpp(arch),
                "row_H": emit_row_retained_fanout_cpp(arch),
            }
            receipts = {}
            for arm, emission in source_emissions.items():
                arm_folder = case_folder / arm
                arm_folder.mkdir()
                source_path = arm_folder / "source.cpp"
                source_path.write_text(emission.cpp)
                trace_path = arm_folder / "host_stdout.txt"
                if arm == "full_H":
                    receipt = verify_fanout_gcc(
                        arch, emission, runtime_input=x,
                        original_trace_output=trace_path)
                else:
                    receipt = verify_row_retained_gcc(
                        arch, emission, runtime_input=x,
                        original_trace_output=trace_path)
                (arm_folder / "host_receipt.json").write_text(
                    json.dumps(receipt, sort_keys=True, indent=2) + "\n")
                # Source/CSim/Tcl packaging does not execute a vendor tool.
                packaged = write_vitis_package(
                    str(arm_folder / "vitis_source"),
                    arch.functional, _vitis_emission(arch, emission), x,
                    part=PART, clock_ns=CLOCK_NS,
                    clock_uncertainty_ns=UNCERTAINTY_NS,
                    tolerance=CSIM_ABS_TOLERANCE,
                )
                checksums = {
                    name: sha256((arm_folder / "vitis_source" / name)
                                 .read_bytes()).hexdigest()
                    for name in ("model.cpp", "tb.cpp", "run_hls.tcl")
                }
                if checksums["model.cpp"] != receipt["source_cpp_sha256"]:
                    raise AssertionError("Vitis source differs from GCC-verified source")
                if packaged["testbench_sha256"] != checksums["tb.cpp"]:
                    raise AssertionError("Vitis testbench digest mismatch")
                receipts[arm] = {
                    "source_sha256": receipt["source_cpp_sha256"],
                    "GCC_host_receipt_sha256":
                        sha256((arm_folder / "host_receipt.json")
                               .read_bytes()).hexdigest(),
                    "E1_host_GCC_status": receipt["status"],
                    "host_stdout_sha256": sha256(
                        trace_path.read_bytes()).hexdigest(),
                    "source_scratch_elements":
                        emission.manifest["source_scratch_array_elements"],
                    "source_mutable_array_elements":
                        emission.manifest["source_all_declared_mutable_array_elements"],
                    "original_GCC_compilations": 1,
                    "original_GCC_executions": 1,
                    "source_vitis_package_sha256": checksums,
                }
            a, b = receipts["full_H"], receipts["row_H"]
            scratch_saving = (
                a["source_scratch_elements"]-b["source_scratch_elements"])
            total_saving = (a["source_mutable_array_elements"]
                            - b["source_mutable_array_elements"])
            if (scratch_saving != (m-1)*n
                    or total_saving != scratch_saving):
                raise AssertionError("R4a expected full-to-row storage effect changed")
            rows.append({
                "case": f"case_{i:03d}", "graph": list(item),
                "source_revision": revision,
                "full_H": a, "row_H": b,
                "source_scratch_elements_saved": scratch_saving,
                "source_mutable_array_elements_saved": total_saving,
                "source_all_mutable_fraction_saved":
                    total_saving / a["source_mutable_array_elements"],
                "source_only": True,
                "HLS_BRAM18K_delta": None,
                "native_E3_qualified": None,
                "physical_E4_qualified": None,
            })
    except Exception as exc:
        (root / "FAILED.json").write_text(json.dumps({
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:5000],
            "completed_case_count": len(rows),
            "source_revision": revision,
        }, indent=2, sort_keys=True) + "\n")
        raise
    if len(rows) != len(cases):
        raise AssertionError("R4a case denominator changed")
    summary = {
        "schema": SCHEMA, "source_revision": revision,
        "configuration": conf,
        "completed_original_GCC_matched_pairs": len(rows),
        "original_GCC_compilations": 2*len(rows),
        "original_GCC_executions": 2*len(rows),
        "total_source_scratch_elements_saved":
            sum(row["source_scratch_elements_saved"] for row in rows),
        "observed_HLS_E2_qualified": None,
        "observed_HLS_E3_qualified": None,
        "observed_physical_E4_qualified": None,
        "native_observation_status": "NOT_RUN",
        "native_Vitis_calls": 0,
        "native_Vivado_calls": 0,
        "rows": rows,
        "limitations": (
            "Row-retained source E1 and original fixed-point CSim packages only",
            "HLS CSynth and CSim not run by this command",
            "No FPGA hardware cycle port banking or clean DRC evidence",
            "No memory-saving novelty or performance superiority established",
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True,
                   allow_nan=False) + "\n")
    artifacts = sorted(
        p for p in root.rglob("*") if p.is_file()
        and p.name != "artifact_sha256.json")
    (root / "artifact_sha256.json").write_text(json.dumps(
        {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
         for p in artifacts}, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("quick", "paper"), default="quick")
    p.add_argument("--output")
    args = p.parse_args(argv)
    if args.output is None:
        print(json.dumps(plan(profile=args.profile),
                         indent=2, sort_keys=True))
        return
    r = run(profile=args.profile, output=args.output)
    print(json.dumps({
        "schema": SCHEMA, "source_revision": r["source_revision"],
        "matched_GCC_pairs": r["completed_original_GCC_matched_pairs"],
        "source_scratch_total_saved": r["total_source_scratch_elements_saved"],
        "new_Vitis_calls": 0, "new_Vivado_calls": 0,
        "original_output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
