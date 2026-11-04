"""Prepare EXACT matched 36-pair R2h fanout sources for real native Vitis HLS.

E1 source-only, 72 Vitis packages (36 identical graph/Phi/input/Tcl/TB
pairs with disjoint vs per-element optimized source scratch arrays).
Uses the existing canonical HLS package writer and numerical testbench.
It does NOT run any vendor tools and does not assert FPGA BRAM savings.

The historical 72 GCC sources/receipts remain frozen. A native study is
a distinct new source revision and new original toolchain campaign.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    LAYOUTS,emit_allocated_fanout_cpp,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    FANOUTS,SHAPES,POLICY_NAMES,policy_widths,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.semantics import architecture_fingerprint
from fpgai_research.toolchains.vitis_hls.package import (
    write_vitis_package,make_vitis_tcl,
)

SCHEMA="fpgai-research/cst-R2i-native-matched-fanout-source-preparation/v1"
PART="xck26-sfvc784-2LV-c"
STUDY="native-fanout-36-R2h-disjoint-vs-unit-slot"
SOURCE_ARMS=("disjoint","per_element_optimal")
TOTAL_PAIRS=len(FANOUTS)*len(SHAPES)*len(POLICY_NAMES)


def _file_sha(path:Path)->str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native source inputs must be ordinary immutable files")
    return sha256(path.read_bytes()).hexdigest()


def source_revision()->str:
    root=Path(__file__).resolve().parents[3]
    out=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
        capture_output=True,text=True,check=True,timeout=5).stdout.strip()
    if len(out)!=40:
        raise RuntimeError("Source-pinned HLS study requires full Git SHA")
    return out


def _source_manifest(arch,*,layout:str)->Emission:
    original=emit_allocated_fanout_cpp(arch,layout=layout)
    widths={(p.effective.precision_bits,p.effective.integer_bits)
            for p in arch.points}
    if len(widths)!=1:
        raise ValueError("Native fixed-point model requires consistent precision")
    bits,integer=next(iter(widths))
    return Emission(original.cpp,{
        **original.manifest,
        "cpp_sha256":sha256(original.cpp.encode()).hexdigest(),
        "precision_bits":bits,
        "integer_bits":integer,
        "source_storage":layout,
        "source_study_family":STUDY,
        "evidence":"E1-SOURCE_ONLY-VITIS-NOT_RUN",
        "native_csim_or_csynth_observed":False,
        "native_vendor_calls_during_prepare":0,
    })


def prepare(*,output:str|Path,part:str=PART,
            clock_ns:float=32.0,uncertainty_ns:float=8.64,
            tolerance:float=0.1)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original R2i Vitis source package folder must be unused")
    if (not all(type(x) in (float,int) and isfinite(x)
                for x in (clock_ns,uncertainty_ns,tolerance))
            or not 0<=uncertainty_ns<clock_ns or tolerance<=0):
        raise ValueError("Finite matching native clock/uncertainty/tolerance required")
    # Validate exact vendor Tcl semantics before emitting first source.
    make_vitis_tcl(part=part,clock_ns=clock_ns,
                   clock_uncertainty_ns=uncertainty_ns)
    worktree=Path(__file__).resolve().parents[3]
    dirty=subprocess.run(
        ["git","-C",str(worktree),"status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=5).stdout
    if dirty.strip():
        raise RuntimeError(
            "Original R2i native packages require clean tracked source checkout")
    revision=source_revision()
    root.mkdir(parents=True)
    cases=[]
    for fanout in FANOUTS:
        for dims in SHAPES:
            m,k,n,q=dims
            model=sample_fanout(fanout,m=m,k=k,n=n,q=q)
            input_array=_input(m,k)
            for policy in POLICY_NAMES:
                group=len(cases)//2
                base=ArchitectureRequest()
                arch=resolve(model,base,{
                    f"reader{i}":replace(base,pes=width)
                    for i,width in enumerate(policy_widths(fanout,policy))
                    if width!=1
                })
                for layout in SOURCE_ARMS:
                    case=f"case_{len(cases):03d}"
                    folder=root/case
                    emission=_source_manifest(arch,layout=layout)
                    package=write_vitis_package(
                        str(folder),model,emission,input_array,
                        part=part,clock_ns=clock_ns,
                        clock_uncertainty_ns=uncertainty_ns,tolerance=tolerance)
                    cases.append({
                        "case":case,"group":group,
                        "graph":f"fanout{fanout}_{m}x{k}x{n}x{q}",
                        "fanout":fanout,"shape":list(dims),
                        "mapping":policy,"source_storage":layout,
                        "source_revision":revision,
                        "functional_sha256":package["functional_sha256"],
                        "architecture_sha256":architecture_fingerprint(arch),
                        "source_cpp_sha256":_file_sha(folder/"model.cpp"),
                        "testbench_sha256":_file_sha(folder/"tb.cpp"),
                        "tcl_sha256":_file_sha(folder/"run_hls.tcl"),
                        "case_manifest_sha256":_file_sha(folder/"manifest.json"),
                        "original_source_scratch_unit_slots":
                            emission.manifest["source_scratch_array_elements"],
                        "original_declared_mutable_source_elements":
                            emission.manifest["source_all_declared_mutable_array_elements"],
                        "numeric_input_sha256":sha256(input_array.tobytes()).hexdigest(),
                        "original_native_vitis_calls":0,
                        "csim_csynth_observed":False,
                        "physical_qualified":False,
                    })
    if len(cases)!=2*TOTAL_PAIRS:
        raise AssertionError("R2i native source prespecified denominator incomplete")
    for i in range(0,len(cases),2):
        a,b=cases[i:i+2]
        if (a["source_storage"]!="disjoint"
                or b["source_storage"]!="per_element_optimal"
                or a["group"]!=b["group"]
                or a["graph"]!=b["graph"]
                or a["mapping"]!=b["mapping"]
                or a["functional_sha256"]!=b["functional_sha256"]
                or a["architecture_sha256"]!=b["architecture_sha256"]
                or a["numeric_input_sha256"]!=b["numeric_input_sha256"]
                or a["testbench_sha256"]!=b["testbench_sha256"]
                or a["tcl_sha256"]!=b["tcl_sha256"]
                or a["source_cpp_sha256"]==b["source_cpp_sha256"]
                or not a["original_source_scratch_unit_slots"]>
                       b["original_source_scratch_unit_slots"]):
            raise AssertionError("Unmatched original native source pair")
    summary={
        "schema":SCHEMA,"study":STUDY,
        "source_revision":revision,
        "part":part,
        "requested_clock_ns":clock_ns,
        "explicit_clock_uncertainty_ns":uncertainty_ns,
        "numerical_tolerance":tolerance,
        "source_cases":len(cases),
        "matched_pairs":len(cases)//2,
        "prespecified_full_population_pairs":TOTAL_PAIRS,
        "cases":cases,
        "original_vitis_calls":0,
        "source_only_not_native_evidence":True,
        "original_HLS_CSim_qualified_cases":0,
        "original_HLS_CSynth_qualified_cases":0,
        "physical_BRAM_or_Fmax_qualified_cases":0,
        "external_peer_comparison_executed":False,
        "warnings":[
            "All original Vitis CSim/CSynth metrics are NOT_RUN",
            "Small fanout graphs may map arrays to registers/LUTs rather than BRAM",
            "R2h source scalar unit-slot savings do not imply Vitis/physical BRAM savings",
            "Source and numerical testbench equality do not qualify native reports",
        ],
    }
    (root/"native_source_manifest.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def audit_prepared(study:str|Path,*,require_current_source:bool=True)->dict:
    """Pure read-only source/Tcl/testbench/manifest integrity audit."""
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R2i native source study folder missing")
    doc=root/"native_source_manifest.json"
    if doc.is_symlink() or not doc.is_file():
        raise ValueError("R2i study manifest missing")
    original=json.loads(doc.read_text())
    cases=original.get("cases")
    if (original.get("schema")!=SCHEMA
            or type(cases) is not list or len(cases)!=2*TOTAL_PAIRS
            or original.get("matched_pairs")!=TOTAL_PAIRS
            or original.get("source_cases")!=2*TOTAL_PAIRS):
        raise ValueError("R2i prepared source population mismatch")
    if (require_current_source and
            original.get("source_revision")!=source_revision()):
        raise ValueError("R2i native source revision differs from checkout")
    for index,entry in enumerate(cases):
        ordinal=index//2
        expected_fanout=FANOUTS[ordinal//(len(SHAPES)*len(POLICY_NAMES))]
        within=ordinal%(len(SHAPES)*len(POLICY_NAMES))
        expected_dims=SHAPES[within//len(POLICY_NAMES)]
        expected_mapping=POLICY_NAMES[within%len(POLICY_NAMES)]
        expected_graph="fanout{}_{}x{}x{}x{}".format(
            expected_fanout,*expected_dims)
        if (entry.get("case")!=f"case_{index:03d}"
                or entry.get("group")!=ordinal
                or entry.get("source_storage")!=SOURCE_ARMS[index%2]
                or entry.get("graph")!=expected_graph
                or entry.get("fanout")!=expected_fanout
                or entry.get("shape")!=list(expected_dims)
                or entry.get("mapping")!=expected_mapping
                or entry.get("source_revision")!=original.get("source_revision")):
            raise ValueError("R2i missing/out-of-order native source arm or fixed model/Phi")
        folder=root/entry["case"]
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Original native source arm folder missing")
        for filename,field in (
            ("model.cpp","source_cpp_sha256"),
            ("tb.cpp","testbench_sha256"),
            ("run_hls.tcl","tcl_sha256"),
            ("manifest.json","case_manifest_sha256"),
        ):
            if _file_sha(folder/filename)!=entry[field]:
                raise ValueError("R2i original native source input SHA differs")
        pkg=json.loads((folder/"manifest.json").read_text())
        if (pkg.get("cpp_sha256")!=entry["source_cpp_sha256"]
                or pkg.get("testbench_sha256")!=entry["testbench_sha256"]
                or pkg.get("part")!=original["part"]
                or pkg.get("clock_ns")!=original["requested_clock_ns"]
                or pkg.get("explicit_clock_uncertainty_ns")!=
                    original["explicit_clock_uncertainty_ns"]
                or pkg.get("source_storage")!=entry["source_storage"]
                or pkg.get("source_study_family")!=STUDY
                or pkg.get("functional_sha256")!=entry["functional_sha256"]
                or pkg.get("architecture_sha256")!=entry["architecture_sha256"]
                or pkg.get("source_scratch_array_elements")!=
                    entry["original_source_scratch_unit_slots"]
                or pkg.get("source_all_declared_mutable_array_elements")!=
                    entry["original_declared_mutable_source_elements"]
                or pkg.get("numerical_tolerance")!=original["numerical_tolerance"]):
            raise ValueError("R2i native case/source manifest contract mismatch")
    for group in range(TOTAL_PAIRS):
        a,b=cases[group*2:group*2+2]
        if (a["functional_sha256"]!=b["functional_sha256"]
                or a["architecture_sha256"]!=b["architecture_sha256"]
                or a["numeric_input_sha256"]!=b["numeric_input_sha256"]
                or a["testbench_sha256"]!=b["testbench_sha256"]
                or a["tcl_sha256"]!=b["tcl_sha256"]
                or a["source_cpp_sha256"]==b["source_cpp_sha256"]
                or a["original_source_scratch_unit_slots"]<=
                   b["original_source_scratch_unit_slots"]):
            raise ValueError("R2i prepared native matched pair control failure")
    return {
        "schema":SCHEMA,
        "status":"ORIGINAL_NATIVE_SOURCE_ONLY_PREPARED",
        "source_revision":original["source_revision"],
        "cases":len(cases),
        "matched_pairs":TOTAL_PAIRS,
        "original_Vitis_HLS_calls_during_audit":0,
        "native_CSim_or_CSynth_qualified_by_source_preparation":False,
        "physical_BRAM_savings_established":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output")
    group.add_argument("--audit")
    p.add_argument("--part",default=PART)
    p.add_argument("--clock-ns",type=float,default=32.0)
    p.add_argument("--uncertainty-ns",type=float,default=8.64)
    p.add_argument("--tolerance",type=float,default=0.1)
    args=p.parse_args(argv)
    report=(prepare(
        output=args.output,part=args.part,clock_ns=args.clock_ns,
        uncertainty_ns=args.uncertainty_ns,tolerance=args.tolerance)
        if args.output else audit_prepared(args.audit))
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
