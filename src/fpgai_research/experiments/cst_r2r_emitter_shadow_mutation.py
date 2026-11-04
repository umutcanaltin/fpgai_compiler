"""R2r prospective emitter-level mutation + dynamic source shadow memory (E1).

Compile ACTUAL generated R2h C++ with a source-injected independent
per-physical-slot ownership checker. Then deliberately mutate the
memory operations of that instrumented source, not Python event tuples.
The shadow checker tracks stores, loads and all-reader read
multiplicity derived from the FunctionalModel topology, NOT from
claimed FPGAI_R2G or FPGAI_SLOT console witness lines.

Original R2h GCC verifier additionally checks emitted numerical values,
event order and exact planned slot addresses. The two layers have
distinct detection reasons and preserve all survivors.

This is source instrumentation, not binary/RTL shadow memory,
arbitrary-DAG proof, native HLS/DRC/BRAM, or a trained RL result.
No GitHub Actions or vendor calls. New results are create-only and
original source-revision pinned; the frozen R2h/R2i/R2j/R2l/R2q
experiment results are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    emit_allocated_fanout_cpp,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.fanout_contract import admit_fanout
from fpgai_research.validation.r2h_allocated_fanout_cpp import verify_allocated_gcc

SCHEMA = "fpgai-research/cst-R2r-emitted-C++-dynamic-shadow-mutation-E1/v1"
MUTATION_FAMILIES = (
    "producer_store_slot",
    "producer_read_slot",
    "reader_store_slot",
    "add_left_read_slot",
    "add_right_read_slot",
    "add_store_slot",
    "final_read_slot",
    "skip_producer_store",
    "producer_numeric",
)
DELTAS = (1, 2, 3, 4)
CASES = (
    (2, (2, 2, 3, 4), "scalar"),
    (2, (2, 3, 5, 4), "all_pes2"),
    (3, (2, 2, 3, 4), "scalar"),
    (3, (2, 3, 5, 4), "all_pes2"),
    (4, (2, 2, 3, 4), "scalar"),
    (4, (3, 3, 4, 4), "all_pes2"),
)
EXPECTED_ALL_MUTANTS = len(CASES) * len(MUTATION_FAMILIES) * len(DELTAS)


def _hash(s: str) -> str:
    return sha256(s.encode()).hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    found = source.count(old)
    if found < 1:
        raise ValueError("R2r source instrumentation anchor unavailable: " + repr(old))
    return source.replace(old, new, 1)


def _arch(spec: tuple):
    readers, shape, policy = spec
    m, k, n, q = shape
    model = sample_fanout(readers, m=m, k=k, n=n, q=q)
    default = ArchitectureRequest()
    return resolve(model, default, {
        f"reader{i}": replace(default, pes=2)
        for i in range(readers) if policy == "all_pes2"
    })


def _with_cpp(source: str, original: Emission) -> Emission:
    return Emission(source, {
        **original.manifest,
        "source_cpp_sha256": _hash(source),
    })


def shadow_instrument(original_arch, emission: Emission) -> Emission:
    """Instrument every known source scratch LOAD/STORE with a second witness.

    The memory guards enforce per-slot ownership and topology-derived
    read obligations. No dependence on the original source event log
    or the original slot allocation to determine occupant identities.
    """
    spec = admit_fanout(original_arch)
    m, n, q = spec.m, spec.n, spec.q
    cap = emission.manifest["source_scratch_array_elements"]
    if cap < 2 or emission.manifest["source_layout"] != "per_element_optimal":
        raise ValueError("R2r requires actual bounded optimized R2h emission")
    names = spec.intermediates
    lengths = {
        name: m * (n if name == spec.h else q) for name in names
    }
    offsets = {}
    cursor = 0
    remaining = []
    for name in names:
        offsets[name] = cursor
        count = lengths[name]
        remaining.extend(
            [q * len(spec.readers) if name == spec.h else 1] * count
        )
        cursor += count
    values = cursor
    if values < 1:
        raise AssertionError("R2r no source scratch values")

    cpp = emission.cpp
    if not cpp.endswith("}\n"):
        raise ValueError("R2r requires one original complete generated C++ function")
    cpp = "#include <cstdlib>\n" + cpp

    # Actual access, occupant checks and required-read counters sit in
    # these wrappers; original FPGAI_SLOT statements are left unchanged.
    # If an actual load/store address changes without its printed trace,
    # the shadow check sees it BEFORE the printed trace is emitted.
    shadow = f"""
  int shadow_owner[{cap}];
  for(int slot=0;slot<{cap};++slot) shadow_owner[slot]=-1;
  int shadow_remaining[{values}]={{{",".join(map(str,remaining))}}};
  unsigned char shadow_committed[{values}]={{}};
  auto shadow_fault = [&](const char*why,int logical,int slot)->void {{
    std::fprintf(stderr,"FPGAI_SHADOW %s logical=%d slot=%d\\n",why,logical,slot);
    std::exit(73);
  }};
  auto shadow_store = [&](int slot,int logical,data_t value)->void {{
    if(slot<0 || slot>={cap} || logical<0 || logical>={values})
      shadow_fault("BOUNDS",logical,slot);
    if(shadow_committed[logical] || shadow_owner[slot]!=-1)
      shadow_fault("OVERWRITE_LIVE_OR_DOUBLE_COMMIT",logical,slot);
    scratch[slot]=value;
    shadow_owner[slot]=logical;
    shadow_committed[logical]=1;
  }};
  auto shadow_load = [&](int slot,int logical)->data_t {{
    if(slot<0 || slot>={cap} || logical<0 || logical>={values})
      shadow_fault("READ_BOUNDS",logical,slot);
    if(shadow_owner[slot]!=logical || shadow_remaining[logical]<=0)
      shadow_fault("WRONG_OWNER_OR_READ_COUNT",logical,slot);
    data_t value=scratch[slot];
    --shadow_remaining[logical];
    if(shadow_remaining[logical]==0) shadow_owner[slot]=-1;
    return value;
  }};
  auto shadow_skip_store = [&](int slot,int logical,data_t value)->void {{
    (void)slot;(void)logical;(void)value;
  }};
  auto shadow_finish = [&]()->void {{
    for(int logical=0;logical<{values};++logical)
      if(!shadow_committed[logical] || shadow_remaining[logical]!=0)
        shadow_fault("INCOMPLETE_READ_OR_COMMIT",logical,-1);
    for(int slot=0;slot<{cap};++slot)
      if(shadow_owner[slot]!=-1)
        shadow_fault("UNRELEASED_PHYSICAL_SLOT",shadow_owner[slot],slot);
  }};
"""
    original_decl = f"  data_t scratch[{cap}]={{}};"
    if cpp.count(original_decl) != 1:
        raise ValueError("R2r original scratch declaration changed")
    cpp = cpp.replace(original_decl,original_decl + "\n" + shadow,1)

    cpp = _replace_once(
        cpp, "scratch[h_store_addr]=sum;",
        f"shadow_store(h_store_addr,{offsets[spec.h]}+h_idx,sum);")
    for reader in spec.readers:
        cpp = _replace_once(
            cpp, "data_t h_val=scratch[h_addr];",
            f"data_t h_val=shadow_load(h_addr,{offsets[spec.h]}+h_idx);")
    for i, reader_value in enumerate(spec.reader_values):
        cpp = _replace_once(
            cpp, f"scratch[reader_store_addr]=acc[{i}][output_idx];",
            f"shadow_store(reader_store_addr,{offsets[reader_value]}+output_idx,acc[{i}][output_idx]);")
    current = spec.reader_values[0]
    for next_value, new_output in zip(spec.reader_values[1:],spec.add_values):
        cpp = _replace_once(
            cpp,"const data_t left=scratch[left_address];",
            f"const data_t left=shadow_load(left_address,{offsets[current]}+idx);")
        cpp = _replace_once(
            cpp,"const data_t right=scratch[right_address];",
            f"const data_t right=shadow_load(right_address,{offsets[next_value]}+idx);")
        cpp = _replace_once(
            cpp,"scratch[add_store_addr]=sum;",
            f"shadow_store(add_store_addr,{offsets[new_output]}+idx,sum);")
        current = new_output
    cpp = _replace_once(
        cpp,"const data_t v=scratch[source_addr];",
        f"const data_t v=shadow_load(source_addr,{offsets[current]}+idx);")
    if not cpp.endswith("}\n"):
        raise ValueError("R2r generated source no longer terminates in function brace")
    cpp = cpp[:-2] + "  shadow_finish();\n}\n"
    # Strong, explicit no-uninstrumented-original-scratch-access check:
    # shadow lambda bodies use scratch[], but all original read/write
    # assignment sites must now be guarded.
    for forbidden in (
        "scratch[h_store_addr]", "scratch[h_addr]", "scratch[reader_store_addr]",
        "scratch[left_address]", "scratch[right_address]",
        "scratch[add_store_addr]", "scratch[source_addr]",
    ):
        if forbidden in cpp:
            raise ValueError("R2r source retains uninstrumented scratch access "+forbidden)
    return _with_cpp(cpp, emission)


def mutate_instrumented(emission: Emission, *,
                        family: str, delta: int) -> Emission:
    """One changed emitted C++ operation per variant, not a Python E0 event."""
    if family not in MUTATION_FAMILIES or delta not in DELTAS:
        raise ValueError("R2r requires a prespecified source mutation")
    cpp = emission.cpp
    cap = emission.manifest["source_scratch_array_elements"]
    anchors = {
        "producer_store_slot": "shadow_store(h_store_addr,",
        "producer_read_slot": "shadow_load(h_addr,",
        "reader_store_slot": "shadow_store(reader_store_addr,",
        "add_left_read_slot": "shadow_load(left_address,",
        "add_right_read_slot": "shadow_load(right_address,",
        "add_store_slot": "shadow_store(add_store_addr,",
        "final_read_slot": "shadow_load(source_addr,",
    }
    if family in anchors:
        old = anchors[family]
        # Mutate ACTUAL shadow-guarded address, while original R2h
        # printed slot-witness address remains untouched. If a checker
        # trusts only the printed witness, the invalid access can escape.
        function_name, first_arg = old.split("(",1)
        slot_expression = first_arg.removesuffix(",")
        new = f"{function_name}(({slot_expression}+{delta})%{cap},"
        cpp = _replace_once(cpp, old, new)
    elif family == "skip_producer_store":
        old = "shadow_store(h_store_addr,"
        if old not in cpp:
            raise ValueError("R2r producer store missing")
        # Only suppress one actual source H write, not its printed
        # COMMIT/ADDRESS labels. The independent shadow must fail.
        oldstatement = re.search(r"shadow_store\(h_store_addr,[^;]+;",cpp)
        if oldstatement is None:
            raise ValueError("R2r producer statement no longer canonical")
        direct = oldstatement.group()
        skipped = direct.replace("shadow_store(", "shadow_skip_store(",1)
        cpp = _replace_once(
            cpp, direct,
            f"if(h_idx=={delta-1}) {{ {skipped} }} else {{ {direct} }}")
    else:
        original = "sum+=input[x_idx]*w0[w_idx];"
        cpp = _replace_once(
            cpp, original,
            f"sum+=input[x_idx]*w0[w_idx]+data_t({delta});")
    if cpp == emission.cpp:
        raise AssertionError("R2r mutant did not change actual C++")
    return _with_cpp(cpp, emission)


def _original_revision() -> str:
    repo=Path(__file__).resolve().parents[3]
    status=subprocess.run(
        ["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=10).stdout
    if status.strip():
        raise RuntimeError("R2r original GCC experiment requires clean tracked Git revision")
    revision=subprocess.run(
        ["git","-C",str(repo),"rev-parse","HEAD"],
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}",revision) is None:
        raise RuntimeError("R2r original revision must be a full Git SHA")
    return revision


def run(*,output: str | Path,graphs: int=6,mutants_per_graph: int=36) -> dict:
    """Prespecified 6 x 36 = 216 create-only source mutant GCC case matrix."""
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2r original source mutation study is create-only")
    if type(graphs) is not int or not 1<=graphs<=len(CASES):
        raise ValueError("R2r graphs must be in 1..6")
    if type(mutants_per_graph) is not int or not 1<=mutants_per_graph<=36:
        raise ValueError("R2r mutants_per_graph must be in 1..36")
    if shutil.which("g++") is None:
        raise FileNotFoundError("R2r requires real local GNU g++")
    revision=_original_revision()
    root.mkdir(parents=True)
    original_e1=0
    results=[]
    for graph, graph_spec in enumerate(CASES[:graphs]):
        folder=root/f"graph_{graph:03d}"
        folder.mkdir()
        arch=_arch(graph_spec)
        original=emit_allocated_fanout_cpp(
            arch,layout="per_element_optimal")
        instrumented=shadow_instrument(arch,original)
        with (folder/"original_instrumented.cpp").open("x") as stream:
            stream.write(instrumented.cpp)
        with (folder/"original_manifest.json").open("x") as stream:
            json.dump(instrumented.manifest,stream,indent=2,sort_keys=True)
            stream.write("\n")
        input_matrix=_input(graph_spec[1][0],graph_spec[1][1])
        baseline_trace=folder/"original_stdout.txt"
        try:
            receipt=verify_allocated_gcc(
                arch,instrumented,runtime_input=input_matrix,
                original_trace_output=baseline_trace)
        except Exception as exc:
            (folder/"FAILED_BASELINE.json").write_text(json.dumps({
                "type":type(exc).__name__,
                "reason":str(exc)[:4000],
                "source_revision":revision,
                "instrumented_source_sha256":_hash(instrumented.cpp),
            },indent=2,sort_keys=True)+"\n")
            raise
        if (receipt["status"]!="R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED"
                or receipt["source_cpp_sha256"] != _hash(instrumented.cpp)):
            raise AssertionError("R2r original shadowed source baseline not qualified")
        (folder/"baseline_verified.json").write_text(
            json.dumps(receipt,indent=2,sort_keys=True)+"\n")
        original_e1+=1
        for ix,(family,delta) in enumerate(
                ((f,d) for f in MUTATION_FAMILIES for d in DELTAS)):
            if ix >= mutants_per_graph:
                break
            mutated=mutate_instrumented(instrumented,family=family,delta=delta)
            stem=f"mutant_{ix:03d}"
            with (folder/(stem+".cpp")).open("x") as out:
                out.write(mutated.cpp)
            status="SURVIVED"
            reason=None
            error_type=None
            try:
                check=verify_allocated_gcc(
                    arch,mutated,runtime_input=input_matrix)
                if check.get("status")!="R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED":
                    status="UNKNOWN_VERIFIER_RESULT"
            except RuntimeError as exc:
                reason=str(exc)[:4000]
                error_type=type(exc).__name__
                if "GCC compile failed" in reason:
                    status="INVALID_COMPILE"
                elif "FPGAI_SHADOW " in reason:
                    status="DETECTED_DYNAMIC_SHADOW"
                else:
                    status="UNKNOWN_RUNTIME_FAILURE"
            except AssertionError as exc:
                reason=str(exc)[:4000]
                error_type=type(exc).__name__
                status="DETECTED_R2H_CHECKER"
            except ValueError as exc:
                reason=str(exc)[:4000]
                error_type=type(exc).__name__
                # Only the post-GCC decoded R2H event/slot errors count
                # as source-checker detection. Preflight contract/SHA
                # rejection never counts toward actual mutation adequacy.
                status=("DETECTED_R2H_CHECKER" if reason.startswith("R2H_")
                        else "INVALID_PREFLIGHT")
            row={
                "graph":graph,"family":family,"delta":delta,
                "case":stem,"status":status,
                "mutant_source_sha256":_hash(mutated.cpp),
                "baseline_source_sha256":_hash(instrumented.cpp),
                "original_source_revision":revision,
                "error_type":error_type,"reason":reason,
                "original_gcc_attempted":status!="INVALID_PREFLIGHT",
                "new_original_vitis_calls":0,"new_original_vivado_calls":0,
            }
            with (folder/(stem+".json")).open("x") as out:
                json.dump(row,out,indent=2,sort_keys=True)
                out.write("\n")
            results.append(row)
    outcome={}
    for row in results:
        outcome[row["status"]]=outcome.get(row["status"],0)+1
    summary={
        "schema":SCHEMA,
        "original_source_revision":revision,
        "original_graphs":graphs,
        "original_nonmutated_GCC_baselines_qualified":original_e1,
        "original_mutant_count":len(results),
        "planned_full_mutant_population":EXPECTED_ALL_MUTANTS,
        "mutant_status_counts":outcome,
        "true_unsafe_mutants_detected":outcome.get("DETECTED_DYNAMIC_SHADOW",0)+
            outcome.get("DETECTED_R2H_CHECKER",0),
        "surviving_mutants":outcome.get("SURVIVED",0),
        "compile_invalid_mutants":outcome.get("INVALID_COMPILE",0),
        "preflight_invalid_mutants":outcome.get("INVALID_PREFLIGHT",0),
        "unknown_runtime_failures":outcome.get("UNKNOWN_RUNTIME_FAILURE",0),
        "original_gcc_attempts":graphs+sum(
            int(row["original_gcc_attempted"]) for row in results),
        "original_vitis_calls":0,"original_vivado_calls":0,
        "rl_policies_trained":0,
        "evidence":"real generated C++ source with independent slot occupant/read multiplicity shadow and R2h GCC/NumPy verifier",
        "limits":[
            "Instrumented source, not binary-level memory or native RTL correctness",
            "Shadow reads counts derive from restricted fanout topology, independent of printed event traces",
            "No source-level lifetime optimality theorem or hardware RAM-port scheduling",
            "Any survivor, compiler-invalid or unknown runtime mutant remains visible, not a passing detection",
            "Mutation suite is selected, not statistically representative of arbitrary compiler bugs",
        ],
    }
    with (root/"summary.json").open("x") as out:
        json.dump(summary,out,indent=2,sort_keys=True)
        out.write("\n")
    return summary


def audit(study: str | Path) -> dict:
    """Immutable receipt/source SHA review; does not recompile or run GCC."""
    root=Path(study)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R2r original study missing or symlinked")
    summary_path=root/"summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise ValueError("R2r original summary missing")
    summary=json.loads(summary_path.read_text())
    revision=summary.get("original_source_revision")
    graphs=summary.get("original_graphs")
    mutants=summary.get("original_mutant_count")
    if (summary.get("schema")!=SCHEMA
            or not isinstance(revision,str)
            or re.fullmatch("[0-9a-f]{40}",revision) is None
            or type(graphs) is not int or not 1<=graphs<=len(CASES)
            or type(mutants) is not int or mutants<1 or mutants%graphs
            or mutants//graphs>36
            or summary.get("original_nonmutated_GCC_baselines_qualified")!=graphs
            or type(summary.get("original_gcc_attempts")) is not int
            or not graphs<=summary["original_gcc_attempts"]<=graphs+mutants
            or summary.get("original_vitis_calls")!=0
            or summary.get("original_vivado_calls")!=0):
        raise ValueError("R2r invalid original study population/source revision")
    current={}
    for gi in range(graphs):
        folder=root/f"graph_{gi:03d}"
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError("R2r original graph receipt folder missing")
        src=folder/"original_instrumented.cpp"
        manifest=folder/"original_manifest.json"
        baseline=folder/"baseline_verified.json"
        stdout=folder/"original_stdout.txt"
        if any(not p.is_file() or p.is_symlink()
               for p in (src,manifest,baseline,stdout)):
            raise ValueError("R2r baseline original C++/GCC source missing")
        original_text=src.read_text()
        original_sha=_hash(original_text)
        source_manifest=json.loads(manifest.read_text())
        original_emission=Emission(original_text,source_manifest)
        old=json.loads(baseline.read_text())
        if (old.get("source_cpp_sha256")!=original_sha
                or old.get("original_stdout_sha256")!=sha256(stdout.read_bytes()).hexdigest()
                or json.loads(manifest.read_text()).get("source_cpp_sha256")!=original_sha):
            raise ValueError("R2r original baseline source/trace SHA mismatch")
        for mi,(family,delta) in enumerate(
                ((f,d) for f in MUTATION_FAMILIES for d in DELTAS)):
            if mi>=mutants//graphs:
                break
            stem=f"mutant_{mi:03d}"
            path=folder/(stem+".cpp")
            receipt=folder/(stem+".json")
            if any(not p.is_file() or p.is_symlink() for p in (path,receipt)):
                raise ValueError("R2r original mutant source/receipt missing")
            row=json.loads(receipt.read_text())
            if (row.get("graph")!=gi or row.get("family")!=family
                    or row.get("delta")!=delta
                    or row.get("case")!=stem
                    or row.get("original_source_revision")!=revision
                    or row.get("baseline_source_sha256")!=original_sha
                    or row.get("mutant_source_sha256")!=sha256(path.read_bytes()).hexdigest()
                    or path.read_text() != mutate_instrumented(
                        original_emission,family=family,delta=delta).cpp
                    or row.get("mutant_source_sha256")==original_sha
                    or row.get("status") not in (
                        "DETECTED_DYNAMIC_SHADOW","DETECTED_R2H_CHECKER",
                        "SURVIVED","INVALID_COMPILE","INVALID_PREFLIGHT",
                        "UNKNOWN_RUNTIME_FAILURE","UNKNOWN_VERIFIER_RESULT")):
                raise ValueError("R2r forged or mismatched source mutant evidence")
            status=row["status"]
            if status=="DETECTED_DYNAMIC_SHADOW" and (
                    "FPGAI_SHADOW " not in (row.get("reason") or "")
                    or row.get("error_type")!="RuntimeError"):
                raise ValueError("R2r shadow classification unsupported by original receipt")
            if row.get("original_gcc_attempted") is not (
                    status!="INVALID_PREFLIGHT"):
                raise ValueError("R2r original GCC invocation scope forged")
            current[status]=current.get(status,0)+1
    observed_attempted=graphs+mutants-current.get("INVALID_PREFLIGHT",0)
    if (summary.get("original_gcc_attempts")!=observed_attempted
            or summary.get("mutant_status_counts")!=current
            or summary.get("true_unsafe_mutants_detected")!=
                current.get("DETECTED_DYNAMIC_SHADOW",0)+
                    current.get("DETECTED_R2H_CHECKER",0)
            or summary.get("surviving_mutants")!=current.get("SURVIVED",0)
            or summary.get("compile_invalid_mutants")!=current.get("INVALID_COMPILE",0)
            or summary.get("preflight_invalid_mutants")!=current.get("INVALID_PREFLIGHT",0)
            or summary.get("unknown_runtime_failures")!=current.get("UNKNOWN_RUNTIME_FAILURE",0)):
        raise ValueError("R2r original mutation adequacy denominator altered")
    return {
        "schema":"fpgai-research/cst-R2r-emitter-mutant-read-only-SHA-audit/v1",
        "original_source_revision":revision,
        "original_graphs":graphs,
        "original_mutants":mutants,
        "status_counts":current,
        "original_GCC_compilations_during_audit":0,
        "original_vendor_calls_during_audit":0,
        "independently_replayed_GCC":False,
        "limitations":"Receipt and mutated C++ SHA consistency, not independent GCC/numerical replay",
    }


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--study")
    parser.add_argument("--audit-output")
    parser.add_argument("--graphs",type=int,default=6)
    parser.add_argument("--mutants-per-graph",type=int,default=36)
    args=parser.parse_args(argv)
    if bool(args.output)==bool(args.study):
        parser.error("Choose one of --output original GCC or --study audit")
    if args.study:
        result=audit(args.study)
        if args.audit_output:
            path=Path(args.audit_output)
            if path.exists() or path.is_symlink():
                raise FileExistsError("R2r audit output strictly create-only")
            with path.open("x") as out:
                json.dump(result,out,indent=2,sort_keys=True)
                out.write("\n")
    else:
        if args.audit_output:
            parser.error("--audit-output is for --study only")
        result=run(output=args.output,graphs=args.graphs,
                   mutants_per_graph=args.mutants_per_graph)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
