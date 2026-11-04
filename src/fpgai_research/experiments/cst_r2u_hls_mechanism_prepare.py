"""R2u prospective Vitis HLS mechanism factorial: 2 graphs x 16 arms.

Factorial controls (all on the SAME R2h static generated-source compiler):
  - disjoint vs element-exact source scratch reuse,
  - constant table versus switch-case lookup for physical addresses,
  - default Vitis RAM inference versus explicit RAM_2P/BRAM binding,
  - exact scalar array declaration versus declared capacity padded to 64.

Every case has the exact same graph, Phi, numeric input, clock, testbench
and Tcl within its graph block. Scratch-address maps/logical numerical
ops are held fixed within each (graph, storage-layout) arm. Source
preparation is E1 only: NO vendor calls, no GCC, no claims that unused
padding survives memory synthesis or that a BRAM directive is honored.

Native CSynth/CSim and E3 qualification are separate opt-in work.
Historical original R2i/R2j/R2l/R2m evidence is never mutated.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import itertools
import json
from math import isfinite
from pathlib import Path
import re
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    emit_allocated_fanout_cpp,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.toolchains.vitis_hls.package import (
    make_vitis_tcl, write_vitis_package,
)

SCHEMA = "fpgai-research/cst-R2u-Vitis-HLS-memory-mechanism-factorial-prep/v1"
STUDY = "r2u-vitis-controlled-memory-lookup-binding-padding"
PART = "xck26-sfvc784-2LV-c"
CLOCK_NS = 8.0
UNCERTAINTY_NS = 1.5
TOLERANCE = 0.01
GRAPHS = (
    (2, (2, 3, 5, 4), "all_pes2"),
    (4, (3, 3, 4, 4), "all_pes2"),
)
FACTORS = {
    "storage": ("disjoint", "per_element_optimal"),
    "address": ("table", "switch"),
    "binding": ("auto", "ram_2p_bram"),
    "padding": ("exact", "pad64"),
}
FACTOR_NAMES = tuple(FACTORS)
ARMS = tuple(itertools.product(*(FACTORS[key] for key in FACTOR_NAMES)))
CASES = len(GRAPHS) * len(ARMS)
TABLE = re.compile(r"^static const int r2h_slot_([A-Za-z][A-Za-z0-9_]*)\[([0-9]+)\] = \{([0-9,]+)\};$", re.M)


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _rev() -> str:
    repo=Path(__file__).resolve().parents[3]
    dirty=subprocess.run(
        ["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=10,
    ).stdout
    if dirty.strip():
        raise RuntimeError("R2u original vendor study requires clean tracked Git")
    revision=subprocess.run(
        ["git","-C",str(repo),"rev-parse","HEAD"],
        capture_output=True,text=True,check=True,timeout=10,
    ).stdout.strip()
    if re.fullmatch("[0-9a-f]{40}",revision) is None:
        raise RuntimeError("R2u original full source revision missing")
    return revision


def _arch(index: int):
    if type(index) is not int or not 0 <= index < len(GRAPHS):
        raise ValueError("R2u requires one of two preselected source graphs")
    readers,dims,policy=GRAPHS[index]
    model=sample_fanout(readers,m=dims[0],k=dims[1],n=dims[2],q=dims[3])
    req=ArchitectureRequest()
    arch=resolve(model,req,{
        f"reader{i}":replace(req,pes=2) for i in range(readers)
    })
    if tuple(arch.points[i+1].effective.pes
             for i in range(readers))!=(2,)*readers:
        raise AssertionError("R2u prespecified effective Phi unexpectedly changed")
    return arch


def _switch_address_cpp(source: str) -> tuple[str,int]:
    matches=list(TABLE.finditer(source))
    if not matches:
        raise ValueError("R2u source has no original static slot tables")
    names=set()
    for match in matches:
        name,length_raw,values_raw=match.groups()
        if name in names:
            raise ValueError("R2u duplicate original address table")
        names.add(name)
        length=int(length_raw)
        values=[int(x) for x in values_raw.split(",")]
        if len(values)!=length:
            raise ValueError("R2u malformed source slot table length")
    lookup={}
    for match in matches:
        name,length_raw,values_raw=match.groups()
        values=[int(x) for x in values_raw.split(",")]
        functions=[f"static int r2u_lookup_{name}(int logical) {{",
                   "  switch(logical) {"]
        functions.extend(f"    case {i}: return {slot};"
                         for i,slot in enumerate(values))
        functions.extend(["    default: return -1;","  }","}"])
        lookup[name]="\n".join(functions)
    rewritten=TABLE.sub(lambda match:lookup[match.group(1)],source)
    total=0
    for name in names:
        pattern=rf"\br2h_slot_{name}\[([A-Za-z_][A-Za-z0-9_]*)\]"
        rewritten,count=re.subn(
            pattern,lambda m:f"r2u_lookup_{name}({m.group(1)})",rewritten)
        if count < 1:
            raise ValueError("R2u switch generated but original table never read")
        total+=count
    if TABLE.search(rewritten) or re.search(r"\br2h_slot_[A-Za-z0-9_]+\[",rewritten):
        raise ValueError("R2u hybrid source retains original address table")
    return rewritten,total


def emit_factorial_arm(arch, *,storage:str,address:str,binding:str,padding:str) -> Emission:
    arms={"storage":storage,"address":address,"binding":binding,"padding":padding}
    if any(value not in FACTORS[key] for key,value in arms.items()):
        raise ValueError("R2u source arm must use prespecified factorial levels")
    original=emit_allocated_fanout_cpp(arch,layout=storage)
    source=original.cpp
    original_slots=original.manifest["source_scratch_array_elements"]
    if type(original_slots) is not int or original_slots<1:
        raise ValueError("R2u original source scratch capacity invalid")
    lookup_count=0
    if address=="switch":
        source,lookup_count=_switch_address_cpp(source)
    padded=(
        original_slots if padding=="exact"
        else ((original_slots+63)//64)*64
    )
    original_decl=f"  data_t scratch[{original_slots}]={{}};"
    if source.count(original_decl)!=1:
        raise ValueError("R2u canonical original scratch declaration changed")
    replacement=f"  data_t scratch[{padded}]={{}};"
    if binding=="ram_2p_bram":
        replacement+="\n#pragma HLS bind_storage variable=scratch type=ram_2p impl=bram"
    source=source.replace(original_decl,replacement,1)
    if source==original.cpp:
        # The exact/table/auto/exact control intentionally equals R2h,
        # not a new source mutation; it is still a valid control arm.
        if arms!={
            "storage":storage,"address":"table",
            "binding":"auto","padding":"exact",
        }:
            raise AssertionError("R2u requested nontrivial source intervention absent")
    fmt={(point.effective.precision_bits,point.effective.integer_bits)
         for point in arch.points}
    if len(fmt)!=1:
        raise ValueError("R2u consistent fixed-point precision required")
    bits,integer=next(iter(fmt))
    return Emission(source,{
        **original.manifest,
        "cpp_sha256":_sha(source.encode()),
        "source_cpp_sha256":_sha(source.encode()),
        "source_baseline_original_cpp_sha256":_sha(original.cpp.encode()),
        "source_scratch_array_elements":padded,
        "source_logical_scratch_slots":original_slots,
        "source_pad_unused_declared_elements":padded-original_slots,
        "source_all_declared_mutable_array_elements":
            padded+original.manifest["source_accumulator_array_elements"]+
            original.manifest["source_counter_array_elements"],
        "source_storage":storage,
        "source_study_family":STUDY,
        "experimental_address_lookup":address,
        "experimental_memory_binding":binding,
        "experimental_array_padding":padding,
        "experimental_switch_access_count":lookup_count,
        "experimental_pragma_requested_not_verified":binding=="ram_2p_bram",
        "precision_bits":bits,"integer_bits":integer,
        "native_vitis_calls_during_source_emission":0,
        "physical_memory_measurements":None,
    })


def _fixed_arm(index:int,arm:tuple[str,...]) -> tuple[dict,Emission,object]:
    arch=_arch(index)
    names=dict(zip(FACTOR_NAMES,arm))
    emission=emit_factorial_arm(arch,**names)
    return names,emission,arch


def prepare(*,output:str|Path,clock_ns:float=CLOCK_NS,
            uncertainty_ns:float=UNCERTAINTY_NS,
            tolerance:float=TOLERANCE,part:str=PART)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2u original HLS factorial folder strictly create-only")
    if (any(type(v) not in (int,float) or not isfinite(v)
            for v in (clock_ns,uncertainty_ns,tolerance))
            or clock_ns<=0 or not 0<=uncertainty_ns<clock_ns
            or not 0<tolerance<=0.1):
        raise ValueError("R2u finite clock/uncertainty/tolerance controls required")
    make_vitis_tcl(part=part,clock_ns=clock_ns,
                   clock_uncertainty_ns=uncertainty_ns)
    revision=_rev()
    # Structural preflight ALL source emission levels before creating a
    # source-pinned original folder; an unsupported lookup, directive or
    # duplicate arm must fail without leaving an abandoned partial study.
    prepared=[]
    for graph_index in range(len(GRAPHS)):
        arch=_arch(graph_index)
        within=[]
        for arm in ARMS:
            levels,emitted,_=_fixed_arm(graph_index,arm)
            within.append((levels,emitted))
        if len({em.manifest["cpp_sha256"] for _,em in within})!=len(ARMS):
            raise ValueError("R2u preflight duplicate C++ within source factorial")
        prepared.append(within)
    root.mkdir(parents=True)
    rows=[]
    for graph_index,(readers,dims,policy) in enumerate(GRAPHS):
        arch=_arch(graph_index)
        model=arch.functional
        x=_input(dims[0],dims[1])
        input_sha=_sha(x.tobytes())
        for arm_index,arm in enumerate(ARMS):
            case=f"case_{len(rows):03d}"
            levels,emitted=prepared[graph_index][arm_index]
            folder=root/case
            manifest=write_vitis_package(
                str(folder),model,emitted,x,part=part,clock_ns=clock_ns,
                clock_uncertainty_ns=uncertainty_ns,tolerance=tolerance)
            files={name:_sha((folder/name).read_bytes())
                   for name in ("model.cpp","tb.cpp","run_hls.tcl","manifest.json")}
            if files["model.cpp"]!=manifest["cpp_sha256"]:
                raise AssertionError("R2u Vitis package source hash diverged")
            rows.append({
                "case":case,"graph_index":graph_index,
                "graph":f"fanout{readers}_{'x'.join(map(str,dims))}",
                "source_storage":levels["storage"],"mapping":policy,
                "source_revision":revision,"fanout":readers,"shape":list(dims),
                "functional_sha256":manifest["functional_sha256"],
                "architecture_sha256":manifest["architecture_sha256"],
                "numeric_input_sha256":input_sha,
                "source_cpp_sha256":files["model.cpp"],
                "testbench_sha256":files["tb.cpp"],
                "tcl_sha256":files["run_hls.tcl"],
                "case_manifest_sha256":files["manifest.json"],
                "source_logical_scratch_slots":
                    manifest["source_logical_scratch_slots"],
                "source_declared_scratch_elements":
                    manifest["source_scratch_array_elements"],
                "source_event_order_sha256":manifest["source_event_order_sha256"],
                **levels,
            })
    if len(rows)!=CASES:
        raise AssertionError("R2u 32-case factorial cohort incomplete")
    _check_pairs(rows)
    summary={
        "schema":SCHEMA,"source_revision":revision,
        "source_prepared_cases":CASES,
        "graph_blocks":len(GRAPHS),
        "arms_per_graph":len(ARMS),
        "factor_levels":{k:list(v) for k,v in FACTORS.items()},
        "rows":rows,"part":part,"clock_ns":clock_ns,
        "clock_uncertainty_ns":uncertainty_ns,
        "numerical_tolerance":tolerance,
        "source_only_no_new_vitis_calls":True,
        "native_csynth_qualified_cases":0,
        "physical_routed_BRAM":None,
        "limitations":[
            "HLS directives may not bind inferred scratch; verify resulting RTL/report before attribution",
            "Unaccessed padded elements may disappear under HLS optimization",
            "Switch-case lookup changes address-generation logic, not source value semantics",
            "All E3 outcomes UNKNOWN until user executes Vitis on original package",
            "Source-level scratch slots are not mapped FPGA BRAM primitives",
        ],
    }
    with (root/"factorial_source_manifest.json").open("x") as outfile:
        json.dump(summary,outfile,indent=2,sort_keys=True)
        outfile.write("\n")
    return {k:v for k,v in summary.items() if k!="rows"}


def _check_pairs(rows:list[dict])->None:
    if len(rows)!=CASES:
        raise ValueError("R2u entire 32-arm prospective factorial required")
    for gi in range(len(GRAPHS)):
        block=rows[gi*len(ARMS):(gi+1)*len(ARMS)]
        if [tuple(row[key] for key in FACTOR_NAMES) for row in block]!=list(ARMS):
            raise ValueError("R2u source factor level ordering changed")
        for field in ("architecture_sha256","functional_sha256",
                      "numeric_input_sha256","testbench_sha256","tcl_sha256"):
            if len({row[field] for row in block})!=1:
                raise ValueError("R2u source controls not matched within same graph")
        if len({row["source_cpp_sha256"] for row in block})!=len(ARMS):
            raise ValueError("R2u factorial interventions produced identical C++")
    if len({row["source_cpp_sha256"] for row in rows})!=CASES:
        raise ValueError("R2u source variants unexpectedly duplicate across graphs")


def audit_prepared(*,study:str|Path,require_current_source:bool=True)->dict:
    root=Path(study)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R2u original factorial study directory missing")
    path=root/"factorial_source_manifest.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("R2u original factorial source manifest missing")
    summary=json.loads(path.read_text())
    if (summary.get("schema")!=SCHEMA
            or summary.get("source_prepared_cases")!=CASES
            or summary.get("graph_blocks")!=len(GRAPHS)
            or summary.get("arms_per_graph")!=len(ARMS)
            or summary.get("factor_levels")!={k:list(v) for k,v in FACTORS.items()}):
        raise ValueError("R2u original full source factorial definition changed")
    if require_current_source and summary.get("source_revision")!=_rev():
        raise ValueError("R2u original source revision differs from checkout")
    rows=summary.get("rows")
    if not isinstance(rows,list):
        raise ValueError("R2u original source rows absent")
    _check_pairs(rows)
    for i,row in enumerate(rows):
        if (row.get("case")!=f"case_{i:03d}"
                or row.get("graph_index")!=i//len(ARMS)
                or row.get("source_revision")!=summary["source_revision"]):
            raise ValueError("R2u original source case identity altered")
        folder=root/row["case"]
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError("R2u source package folder missing")
        for filename,field in (
            ("model.cpp","source_cpp_sha256"),
            ("tb.cpp","testbench_sha256"),
            ("run_hls.tcl","tcl_sha256"),
            ("manifest.json","case_manifest_sha256"),
        ):
            file=folder/filename
            if file.is_symlink() or not file.is_file() or _sha(file.read_bytes())!=row[field]:
                raise ValueError("R2u original source/CSim/Tcl/package SHA mismatch")
        arch=_arch(row["graph_index"])
        levels={key:row[key] for key in FACTOR_NAMES}
        expected=emit_factorial_arm(arch,**levels)
        if expected.cpp.encode()!=(folder/"model.cpp").read_bytes():
            raise ValueError("R2u original generated memory intervention changed")
        pkg=json.loads((folder/"manifest.json").read_text())
        if (pkg.get("cpp_sha256")!=row["source_cpp_sha256"]
                or pkg.get("source_study_family")!=STUDY
                or pkg.get("source_scratch_array_elements")!=
                    row["source_declared_scratch_elements"]
                or pkg.get("source_logical_scratch_slots")!=
                    row["source_logical_scratch_slots"]
                or pkg.get("numerical_tolerance")!=summary["numerical_tolerance"]
                or pkg.get("explicit_clock_uncertainty_ns")!=
                    summary["clock_uncertainty_ns"]
                or pkg.get("part")!=summary["part"]):
            raise ValueError("R2u source or native controls disagree with original case")
    return {
        "schema":"fpgai-research/cst-R2u-source-only-32-arm-audit/v1",
        "source_revision":summary["source_revision"],
        "attested_source_cases":CASES,
        "complete_graph_blocks":len(GRAPHS),
        "source_rederived_from_current_compiler":require_current_source,
        "original_vitis_calls_in_audit":0,
        "physical_BRAM_known":False,
    }


def main(argv:list[str]|None=None)->None:
    p=argparse.ArgumentParser(description=__doc__)
    one=p.add_mutually_exclusive_group(required=True)
    one.add_argument("--output")
    one.add_argument("--audit")
    p.add_argument("--clock-ns",type=float,default=CLOCK_NS)
    p.add_argument("--uncertainty-ns",type=float,default=UNCERTAINTY_NS)
    p.add_argument("--tolerance",type=float,default=TOLERANCE)
    p.add_argument("--part",default=PART)
    args=p.parse_args(argv)
    result=prepare(output=args.output,clock_ns=args.clock_ns,
                   uncertainty_ns=args.uncertainty_ns,tolerance=args.tolerance,
                   part=args.part) if args.output else audit_prepared(study=args.audit)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
