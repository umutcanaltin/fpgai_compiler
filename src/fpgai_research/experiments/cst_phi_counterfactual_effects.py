"""Causal Phi-change discrimination audit over original TWO-reader GCC cases.

A changed Phi-owner hash is NOT by itself a changed tensor lifetime.
This independent secondary analysis compares the original, fixed
Functional graph and weights with each reader-specific mapping choice.
It tests whether source-executed last-reader release and physical scratch
footprint truly change, or whether only the Phi identity/source does.

This reads recorded original experiment CSV. It neither regenerates an
original source nor elevates CSV rows to independently attested GCC logs.
It is a falsifiable hypothesis audit, NOT prior-art parity or Vitis QoR.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

SCHEMA="fpgai-research/cst-two-reader-phi-causal-effects/v1"
POLICIES=("scalar","reader0_phi_only","reader1_phi_only","both_independent")
LAYOUTS=("disjoint","last_reader_reuse")


def _required_hash(row,key):
    value=row.get(key)
    if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid original GCC case hash: "+key)
    return value


def analyze_rows(rows):
    if not isinstance(rows,list) or len(rows)!=24:
        raise ValueError("Require all 24 original two-reader GCC cases")
    grouped={}
    for row in rows:
        if not isinstance(row,dict) or row.get("policy") not in POLICIES or row.get("layout") not in LAYOUTS:
            raise ValueError("Unexpected experiment policy or layout")
        shape=row.get("shape")
        if not isinstance(shape,str) or not isinstance(json.loads(shape),list):
            raise ValueError("Invalid shape identity")
        key=(shape,row["layout"],row["policy"])
        if key in grouped:
            raise ValueError("Duplicate causal treatment")
        for name in ("functional_sha256","architecture_sha256","cpp_sha256",
                     "reader0_phi_sha256","reader1_phi_sha256",
                     "producer_phi_sha256","source_event_sha256",
                     "last_use_sha256"):
            _required_hash(row,name)
        if row.get("numeric_pass") not in ("True",True) or row.get("executed_trace_pass") not in ("True",True):
            raise ValueError("Unqualified or missing original GCC case")
        for k in ("source_scratch_slots","disjoint_slots","h_required_reads"):
            try:
                n=int(row[k])
            except (ValueError,TypeError,KeyError) as exc:
                raise ValueError("Missing original scalar observation: "+k) from exc
            if n<1:
                raise ValueError("Nonpositive original source observation")
        grouped[key]=row
    shapes=sorted(set(shape for shape,_,_ in grouped))
    if len(shapes)!=3:
        raise ValueError("Require three independent prespecified shapes")
    outcomes=[]
    for shape in shapes:
        for layout in LAYOUTS:
            baseline=grouped.get((shape,layout,"scalar"))
            if baseline is None:
                raise ValueError("Missing baseline in prespecified matrix")
            for policy in POLICIES[1:]:
                variant=grouped.get((shape,layout,policy))
                if variant is None:
                    raise ValueError("Incomplete Phi-change matrix")
                if (baseline["functional_sha256"]!=variant["functional_sha256"]
                        or baseline["producer_phi_sha256"]!=variant["producer_phi_sha256"]
                        or baseline["layout"]!=variant["layout"]):
                    raise ValueError("Causal Phi comparisons require unchanged graph and producer")
                r0=baseline["reader0_phi_sha256"]!=variant["reader0_phi_sha256"]
                r1=baseline["reader1_phi_sha256"]!=variant["reader1_phi_sha256"]
                # A requested hardware knob may be ineffective on a
                # tiny odd/tail shape. Preserve that negative result rather
                # than rejecting/cherry-picking the candidate.
                if (policy=="reader0_phi_only" and r1 or
                        policy=="reader1_phi_only" and r0):
                    raise ValueError("Phi intervention unexpectedly changed control reader")
                outcomes.append({
                    "shape":shape,"layout":layout,"intervention":policy,
                    "reader0_phi_changed":r0,"reader1_phi_changed":r1,
                    "requested_phi_change_effective":r0 or r1,
                    "functional_held_fixed":True,
                    "producer_phi_held_fixed":True,
                    "actual_cpp_changed":(
                        baseline["cpp_sha256"]!=variant["cpp_sha256"]),
                    "mapped_source_events_changed":(
                        baseline["source_event_sha256"]!=variant["source_event_sha256"]),
                    "source_executed_last_use_changed":(
                        baseline["last_use_sha256"]!=variant["last_use_sha256"]),
                    "physical_source_scratch_slots_changed":(
                        int(baseline["source_scratch_slots"])!=
                        int(variant["source_scratch_slots"])),
                    "source_scratch_slot_delta":(
                        int(baseline["source_scratch_slots"])-
                        int(variant["source_scratch_slots"])),
                    "source_read_count_changed":(
                        int(baseline["h_required_reads"])!=
                        int(variant["h_required_reads"])),
                    "native_vitis_calls":0,
                })
        for policy in POLICIES:
            a=grouped[(shape,"disjoint",policy)]
            b=grouped[(shape,"last_reader_reuse",policy)]
            if (a["functional_sha256"]!=b["functional_sha256"]
                    or a["architecture_sha256"]!=b["architecture_sha256"]
                    or a["source_event_sha256"]!=b["source_event_sha256"]):
                raise ValueError("Scratch-layout control changed semantics or Phi")
            if int(a["source_scratch_slots"])<=int(b["source_scratch_slots"]):
                raise ValueError("Actual source scratch reuse not smaller than disjoint")
    lifetime=sum(c["source_executed_last_use_changed"] for c in outcomes)
    capacity=sum(c["physical_source_scratch_slots_changed"] for c in outcomes)
    result={
        "schema":SCHEMA,"original_cases":24,"original_shapes":3,
        "controlled_phi_contrasts":len(outcomes),
        "effective_phi_interventions":sum(c["requested_phi_change_effective"]
                                           for c in outcomes),
        "ineffective_phi_interventions":sum(not c["requested_phi_change_effective"]
                                             for c in outcomes),
        "changed_source_cpp_contrasts":sum(c["actual_cpp_changed"] for c in outcomes),
        "changed_source_event_contrasts":sum(c["mapped_source_events_changed"] for c in outcomes),
        "changed_last_use_contrasts":lifetime,
        "changed_scratch_capacity_contrasts":capacity,
        "lifetime_discrimination_established_by_this_study":lifetime>0,
        "mapping_conditional_capacity_difference_established":capacity>0,
        "novelty_relative_to_prior_art_established":False,
        "original_vendor_calls":0,
        "limitations":[
            "Secondary original CSV analysis; no independent replay of GCC receipts",
            "Changed Phi-event hashes may reflect identity labels, not schedule timing",
            "No original physical bank/BRAM/clock or RTL verification",
            "Three static MatMul fanout shapes do not generalize to arbitrary graphs",
            "A controlled internal ablation is not external peer-artifact parity",
        ],
    }
    return result,outcomes


def analyze_file(*,input_csv:str|Path,output:str|Path):
    src,dest=Path(input_csv),Path(output)
    if src.is_symlink() or not src.is_file() or src.stat().st_size>4*1024*1024:
        raise ValueError("Original case CSV must be a bounded ordinary file")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Counterfactual secondary study is create-only")
    with src.open(newline="") as f:
        rows=list(csv.DictReader(f))
    result,outcomes=analyze_rows(rows)
    result["input_csv_sha256"]=sha256(src.read_bytes()).hexdigest()
    dest.mkdir(parents=True)
    (dest/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (dest/"controlled_phi_contrasts.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    print(json.dumps(analyze_file(input_csv=a.input_csv,output=a.output),
                     indent=2,sort_keys=True))


if __name__=="__main__":
    main()
