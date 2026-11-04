"""R3g: independent small-domain event-semantics oracle and adjacent-swap campaign.

The reference interprets bounded source events from first principles rather
than invoking the production verifier. It SHARES the declared static
functional/architecture model and permitted fixed 2..4-reader topology:
agreement is evidence for this finite model, not a mechanized theorem,
arbitrary-DAG proof, physical scheduling proof, or external compiler baseline.

No GCC, Vitis, Vivado, training or RL environment calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import _git_revision
from fpgai_research.experiments.cst_r3e_c1_fanout_counterexamples import CASES
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.fanout_contract import (
    FanoutEvent, admit_fanout, disjoint_fanout_offsets, generate_fanout_events,
    verify_fanout_events, verify_fanout_issue,
)

SCHEMA = "fpgai-research/R3g-C1-independent-finite-event-oracle/v1"


def reference_event_oracle(arch, events: tuple[FanoutEvent, ...],
                           offsets: dict[str, int]) -> tuple[bool, bool]:
    """Two independent YES/NO decisions: value safety and Phi issue order.

    Uses *model topology admission* but neither production event verifier nor
    its private issue-rank helper, pending-event trace, or slot allocator.
    """
    spec = admit_fanout(arch)
    r, m, k, n, q = len(spec.readers), spec.m, spec.k, spec.n, spec.q
    if (type(events) is not tuple
            or any(not isinstance(ev, FanoutEvent) for ev in events)
            or type(offsets) is not dict):
        return False, False
    sizes = {t: m*(n if t == spec.h else q) for t in spec.intermediates}
    if (set(offsets) != set(sizes)
            or any(type(v) is not int or v < 0 for v in offsets.values())):
        return False, False
    # Construct complete expected per-value consumer tokens from the graph,
    # not from the observed event trace.
    needed = {}
    for row in range(m):
        for kk in range(n):
            needed[(spec.h, row*n+kk)] = {
                (reader, row*q+col) for reader in spec.readers
                for col in range(q)}
    previous = spec.reader_values[0]
    add_inputs = {}
    for i, merger in enumerate(spec.add_ops):
        right = spec.reader_values[i+1]
        add_inputs[merger] = (previous, right)
        previous = spec.add_values[i]
    for j, tensor in enumerate(spec.reader_values):
        consumer = spec.add_ops[0] if j <= 1 else spec.add_ops[j-1]
        for out in range(m*q):
            needed[(tensor, out)] = {(consumer, out)}
    for j, tensor in enumerate(spec.add_values):
        consumer = spec.add_ops[j+1] if j+1<len(spec.add_ops) else spec.final
        for out in range(m*q):
            needed[(tensor, out)] = {(consumer, out)}
    if not needed:
        return False, False

    committed, reductions = set(), {}
    observed_reads = {key: set() for key in needed}
    live_addresses = {}
    safety = True
    for ev in events:
        key = (ev.tensor, ev.element)
        if key not in needed:
            safety = False
            break
        addr = offsets[ev.tensor] + ev.element
        if ev.kind == "CONTRIBUTE":
            row, col = divmod(ev.element, n)
            expected_x = row*k+ev.reduction
            expected_w = ev.reduction*n+col
            if (ev.tensor != spec.h or ev.operation != spec.producer
                    or type(ev.reduction) is not int or not 0 <= ev.reduction < k
                    or ev.output_element != -1 or ev.phi_owner is not None
                    or ev.source_element != expected_x
                    or ev.weight_element != expected_w
                    or key in committed
                    or ev.reduction in reductions.get(key, set())):
                safety = False
                break
            reductions.setdefault(key, set()).add(ev.reduction)
        elif ev.kind == "COMMIT":
            if (ev.operation or ev.output_element != -1 or ev.reduction != -1
                    or ev.source_element != -1 or ev.weight_element != -1
                    or ev.phi_owner is not None or key in committed
                    or addr in live_addresses):
                safety = False
                break
            if ev.tensor == spec.h:
                complete = reductions.get(key, set()) == set(range(k))
            elif ev.tensor in spec.reader_values:
                idx = spec.reader_values.index(ev.tensor)
                row = ev.element//q
                complete = all((spec.readers[idx], ev.element) in
                               observed_reads[(spec.h, row*n+kk)]
                               for kk in range(n))
            else:
                idx = spec.add_values.index(ev.tensor)
                left, right = add_inputs[spec.add_ops[idx]]
                complete = all((spec.add_ops[idx], ev.element) in
                               observed_reads[(src, ev.element)]
                               for src in (left,right))
            if not complete:
                safety = False
                break
            committed.add(key)
            live_addresses[addr] = key
        elif ev.kind == "READ":
            token = (ev.operation, ev.output_element)
            if (ev.reduction != -1 or ev.source_element != -1
                    or ev.weight_element != -1
                    or token not in needed[key]
                    or token in observed_reads[key]
                    or key not in committed
                    or live_addresses.get(addr) != key):
                safety = False
                break
            if ev.tensor == spec.h:
                if ev.operation not in spec.readers:
                    safety = False
                    break
                reader_ix = spec.readers.index(ev.operation)
                e = arch.points[reader_ix+1].effective
                row, kk = divmod(ev.element, n)
                out_row, col = divmod(ev.output_element, q)
                if (row != out_row or ev.phi_owner !=
                        (0, col % e.pes, kk % e.simd, 0)):
                    safety = False
                    break
            elif ev.phi_owner is not None:
                safety = False
                break
            observed_reads[key].add(token)
            if observed_reads[key] == needed[key]:
                del live_addresses[addr]
        else:
            safety = False
            break
    if safety:
        safety = (committed == set(needed) and not live_addresses
                  and all(observed_reads[key] == tokens
                          for key, tokens in needed.items()))

    # Independent forward rank enumeration; this DOES NOT call the runtime
    # inverse source decoder or the production's forward-rank verifier.
    expected_order = []
    for ordinal in range(m*n*q):
        for idx, reader in enumerate(spec.readers):
            e = arch.points[idx+1].effective
            matching = []
            for row in range(m):
                for col in range(q):
                    start = (col//e.pes)*e.pes
                    active = min(e.pes, q-start)
                    for kk in range(n):
                        rank = row*n*q + start*n + kk*active+(col-start)
                        if rank == ordinal:
                            matching.append((
                                reader, row*n+kk, row*q+col,
                                (0,col%e.pes,kk%e.simd,0)))
            if len(matching) != 1:
                return safety, False
            expected_order.extend(matching)
    observed_order = [
        (ev.operation,ev.element,ev.output_element,ev.phi_owner)
        for ev in events if ev.kind=="READ" and ev.tensor==spec.h
    ]
    return safety, observed_order == expected_order


def _case(config):
    readers,m,k,n,q,pe=config
    model=sample_fanout(readers,m=m,k=k,n=n,q=q)
    base=ArchitectureRequest()
    overrides={"reader0":ArchitectureRequest(pes=pe)} if pe!=1 else {}
    arch=resolve(model,base,overrides)
    spec=admit_fanout(arch)
    return arch, generate_fanout_events(arch), disjoint_fanout_offsets(spec)


def evaluate(config: tuple[int,...]) -> dict:
    if config not in CASES:
        raise ValueError("R3g: unregistered source-oracle case")
    arch, original, offsets=_case(config)
    variants=[("canonical",original)]
    # Every adjacent transposition of the unmodified, complete event stream.
    for i in range(len(original)-1):
        changed=list(original)
        changed[i],changed[i+1]=changed[i+1],changed[i]
        variants.append((f"adjacent_{i:04d}",tuple(changed)))
    # Additional negative mutants not reliably covered by adjacent swaps.
    first_h_read=next(i for i,e in enumerate(original)
                      if e.kind=="READ" and e.tensor=="h")
    first_h_commit=next(i for i,e in enumerate(original)
                        if e.kind=="COMMIT" and e.tensor=="h")
    moved=list(original)
    early=moved.pop(first_h_read)
    moved.insert(0,early)
    variants.append(("early_read",tuple(moved)))
    moved=list(original)
    early=moved.pop(first_h_commit)
    moved.insert(0,early)
    variants.append(("premature_commit",tuple(moved)))
    variants.append(("missing_last_event",original[:-1]))
    variants.append(("duplicate_h_read",original[:first_h_read+1]+
                     (original[first_h_read],)+original[first_h_read+1:]))
    corrupted=list(original)
    from dataclasses import replace
    corrupted[first_h_read]=replace(corrupted[first_h_read],
                                    phi_owner=(99,99,99,99))
    variants.append(("wrong_phi_owner",tuple(corrupted)))
    policy=Counter()
    mismatch=[]
    for name,events in variants:
        oracle_safety, oracle_issue=reference_event_oracle(arch,events,offsets)
        certified_safety=verify_fanout_events(arch,offsets,events)["status"]=="VERIFIED_E0"
        certified_issue=verify_fanout_issue(arch,events)["status"]=="VERIFIED_E0"
        policy[(oracle_safety,oracle_issue)]+=1
        if (oracle_safety,oracle_issue)!=(certified_safety,certified_issue):
            mismatch.append({
                "mutant":name,"oracle":[oracle_safety,oracle_issue],
                "compiler":[certified_safety,certified_issue]})
    if mismatch:
        raise AssertionError(
            "R3g: independent source-event oracle disagrees: "+repr(mismatch[:3]))
    if policy[(True,True)]<1 or policy[(False,False)]<1:
        raise AssertionError("R3g: positive and negative control populations missing")
    return {
        "case":list(config),
        "source_event_count":len(original),
        "variant_count":len(variants),
        "adjacent_swaps":len(original)-1,
        "value_safety_and_issue_both_true":policy[(True,True)],
        "value_safe_but_issue_nonconformant":policy[(True,False)],
        "value_unsafe_but_issue_conformant":policy[(False,True)],
        "value_safety_and_issue_both_false":policy[(False,False)],
        "false_acceptances":0,
        "false_rejections":0,
        "model_scope":"static 2..4-reader MatMul+Add+Relu source event semantics",
    }


def run() -> dict:
    rows=[evaluate(c) for c in CASES]
    total=sum(r["variant_count"] for r in rows)
    return {
        "schema":SCHEMA,
        "original_source_cases":len(CASES),
        "tested_event_trace_variants":total,
        "adjacent_swap_mutants":sum(r["adjacent_swaps"] for r in rows),
        "oracle_disagreements":sum(r["false_acceptances"]+r["false_rejections"]
                                   for r in rows),
        "rows":rows,
        "original_gcc_calls":0,"native_vitis_calls":0,
        "original_vivado_calls":0,"rl_training_calls":0,
        "limits":[
            "Both checkers share a bounded static model, not independently proven arbitrary graphs",
            "Adjacent swaps and four extra mutants are finite selected perturbations",
            "Source event order and Phi issue are not hardware clock-cycle timing",
            "This is neither independent competitor artifact nor trained-agent evaluation",
        ],
    }


def create_only(report:dict,*,output:str|Path,source_revision:str)->None:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R3g: original results are create-only")
    if (report.get("schema")!=SCHEMA
            or re.fullmatch(r"[a-f0-9]{40}",source_revision) is None):
        raise ValueError("R3g: source revision or schema invalid")
    root.mkdir(parents=True,exist_ok=False)
    (root/"r3g_independent_event_oracle.json").write_text(
        json.dumps({**report,"source_revision":source_revision},
                   indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def audit(study:str|Path)->dict:
    root=Path(study)
    file=root/"r3g_independent_event_oracle.json"
    if root.is_symlink() or not root.is_dir() or file.is_symlink() or not file.is_file():
        raise ValueError("R3g: missing or symlinked source receipt")
    j=json.loads(file.read_text())
    rows=j.get("rows")
    if (j.get("schema")!=SCHEMA
            or re.fullmatch(r"[a-f0-9]{40}",j.get("source_revision") or "") is None
            or j.get("original_source_cases")!=len(CASES)
            or not isinstance(rows,list) or len(rows)!=len(CASES)
            or any(j.get(x)!=0 for x in
                   ("original_gcc_calls","native_vitis_calls",
                    "original_vivado_calls","rl_training_calls"))
            or j.get("oracle_disagreements")!=0):
        raise ValueError("R3g: source cohort invalid")
    total=0
    swaps=0
    for expected,r in zip(CASES,rows):
        if (r.get("case")!=list(expected)
                or type(r.get("source_event_count")) is not int
                or r["source_event_count"]<2
                or r.get("adjacent_swaps")!=r["source_event_count"]-1
                or r.get("variant_count")!=r["adjacent_swaps"]+6
                or r.get("false_acceptances")!=0
                or r.get("false_rejections")!=0
                or sum(r.get(x,-1) for x in (
                    "value_safety_and_issue_both_true",
                    "value_safe_but_issue_nonconformant",
                    "value_unsafe_but_issue_conformant",
                    "value_safety_and_issue_both_false"))!=r["variant_count"]):
            raise ValueError("R3g: saved independent semantic comparison altered")
        total+=r["variant_count"]
        swaps+=r["adjacent_swaps"]
    if (total!=j.get("tested_event_trace_variants")
            or swaps!=j.get("adjacent_swap_mutants")):
        raise ValueError("R3g: source trace denominator changed")
    return {
        "schema":SCHEMA,
        "source_revision":j["source_revision"],
        "tested_event_trace_variants":total,
        "independent_oracle_disagreements":0,
        "source_receipt_sha256":sha256(file.read_bytes()).hexdigest(),
        "new_compiler_gcc_vitis_vivado_calls":0,
    }


def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--study")
    args=parser.parse_args(argv)
    if args.study:
        print(json.dumps(audit(args.study),sort_keys=True,indent=2))
        return
    revision=_git_revision()
    result=run()
    create_only(result,output=args.output,source_revision=revision)
    print(json.dumps({
        "schema":SCHEMA,"source_revision":revision,
        "original_source_cases":result["original_source_cases"],
        "tested_event_trace_variants":result["tested_event_trace_variants"],
        "oracle_disagreements":result["oracle_disagreements"],
        "original_vendor_calls":0,
    },sort_keys=True,indent=2))


if __name__=="__main__":
    main()
