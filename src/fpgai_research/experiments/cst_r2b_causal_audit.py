"""Independent READ-ONLY R2b E1 causal audit: collapse duplicated scratch arms.

This is a secondary audit of the *original 24 saved GCC sources/traces*,
not 24 new executions, a native HLS study, or external-priority evidence.
The two independently generated scratch layouts for one graph/Phi policy
are paired *controls*: 18 original layout-scoped contrasts correspond
to only NINE unique graph/Phi interventions, NOT 18 independent trials.

Recompute the complete Functional IR, original input bytes, numerical
outputs, original source-reported READ sequence, per-H final-read rank,
last consumer identity and all counterfactual tallies from case assets.
Never replace original counter values or source files.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import (
    SHAPES, POLICIES, LAYOUTS, _input,
)
from fpgai_research.experiments.cst_r2b_source_audit import audit
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint,
)
from fpgai_research.validation.r2b_interleaved_cpp import _decode_trace
from fpgai_research.validation.reference import evaluate

SCHEMA="fpgai-research/cst-R2b-independent-causal-audit-v1"


def _file(path:Path)->bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Original R2b source evidence missing or symlinked: "+str(path))
    if path.stat().st_size>8*1024*1024:
        raise ValueError("Unbounded original R2b source evidence")
    return path.read_bytes()


def _last_read_actual(events,shared_tensor:str):
    last_global={}
    last_read_rank={}
    read_rank=0
    for i,event in enumerate(events):
        if event.kind=="READ" and event.tensor==shared_tensor:
            last_global[event.element]=i
            last_read_rank[event.element]=(
                read_rank,event.consumer,event.output_element)
            read_rank+=1
    return (
        tuple(sorted(last_global.items())),
        tuple(sorted((element,*v) for element,v in last_read_rank.items())),
    )


def evaluate_original(study:str|Path)->dict:
    root=Path(study)
    preliminary=audit(root) # verify all 24 complete original stdout/cpp hashes first
    summary=json.loads(_file(root/"summary.json"))
    csv_rows=list(csv.DictReader(_file(root/"cases.csv").decode().splitlines()))
    if len(csv_rows)!=24 or summary.get("prespecified_cases")!=24:
        raise ValueError("R2b original case population is not the predeclared 24")
    expected={(tuple(shape),p[0],layout) for shape in SHAPES
              for p in POLICIES for layout in LAYOUTS}
    evidence={}
    for row in csv_rows:
        try:
            shape=tuple(json.loads(row["shape"]))
            key=(shape,row["policy"],row["layout"])
        except (KeyError,ValueError,TypeError) as exc:
            raise ValueError("Invalid original R2b case key") from exc
        if key not in expected or key in evidence:
            raise ValueError("Unexpected or duplicate original graph/mapping/layout")
        m,k,n,q=shape
        original=json.loads(_file(root/row["case"]/"original_case.json"))
        verified=json.loads(_file(root/row["case"]/"verified_e1.json"))
        model=sample_branch(m,k,n,q)
        func_sha=functional_fingerprint(model)
        policy=next(p for p in POLICIES if p[0]==row["policy"])
        base=ArchitectureRequest()
        overrides={
            model.operators[i].name:replace(base,**opts)
            for i,opts in ((1,policy[1]),(2,policy[2])) if opts}
        arch=resolve(model,base,overrides)
        arch_sha=architecture_fingerprint(arch)
        manifest=original["source_manifest"]
        if (row["functional_sha256"]!=func_sha or
                row["architecture_sha256"]!=arch_sha or
                manifest["functional_sha256"]!=func_sha or
                manifest["architecture_sha256"]!=arch_sha or
                tuple(original["shape"])!=shape or
                original["policy"]!=row["policy"] or
                original["layout"]!=row["layout"]):
            raise ValueError("Original graph/Phi source identity differs from fixed protocol")
        x=np.asarray(original["input_float32_values"],dtype=np.float32)
        if (x.size!=m*k or
                sha256(x.tobytes()).hexdigest()!=original["input_float32_sha256"] or
                not np.array_equal(x.reshape((m,k)),_input(m,k))):
            raise ValueError("Original fixed source numerical input changed")
        expected_out=np.asarray(
            evaluate(model,{model.inputs[0]:x.reshape((m,k))})[
                model.outputs[0]],dtype=np.float32).reshape(-1).astype(np.float64)
        raw=_file(root/row["case"]/"original_gcc_trace.txt").decode()
        events,observed_out=_decode_trace(raw,n_outputs=m*q,max_events=100_000)
        err=float(np.max(np.abs(observed_out-expected_out)))
        # Strict same tolerance used by the original E1 verifier.
        if (not np.allclose(observed_out,expected_out,atol=1e-5,rtol=1e-4)
                or abs(err-float(row["original_numeric_max_abs_error"]))>1e-6
                or abs(err-float(verified["max_absolute_numeric_error"]))>1e-6):
            raise ValueError("Independent original source NumPy result does not match")
        global_last,read_last=_last_read_actual(
            events,model.operators[0].outputs[0])
        if (len(read_last)!=m*n or
                len([e for e in events if e.kind=="READ" and e.tensor=="h"])!=2*m*n*q or
                [list(e) for e in global_last]!=
                    verified["observed_h_last_use_indices"] or
                [list(e) for e in read_last]!=
                    verified["observed_h_final_read_rank_and_owner"]):
            raise ValueError("Original source H-only last-reader population changed")
        evidence[key]={
            "case":row["case"],"shape":shape,
            "source_sha256":row["source_cpp_sha256"],
            "global_last":global_last,
            "read_last":read_last,
            "output":tuple(observed_out),
            "scratch_slots":int(row["source_scratch_slots"]),
            "source_event_sha256":row["actual_source_event_sha256"],
            "summary_global_change":int(row["changed_h_last_read_elements_vs_scalar"]),
            "summary_read_change":int(row["changed_h_final_read_only_rank_or_owner_vs_scalar"]),
            "summary_consumer_change":int(row["changed_h_final_reader_identity_vs_scalar"]),
        }
    if set(evidence)!=expected:
        raise ValueError("Not all prespecified graph/Phi/layout source cases exist")
    interventions=[]
    layout_contrasts=[]
    for shape in SHAPES:
        for policy, *_ in POLICIES:
            dis=evidence[(shape,policy,"disjoint")]
            shared=evidence[(shape,policy,"last_reader_reuse")]
            if (dis["global_last"]!=shared["global_last"]
                    or dis["read_last"]!=shared["read_last"]
                    or dis["output"]!=shared["output"]
                    or dis["source_event_sha256"]!=shared["source_event_sha256"]
                    or dis["scratch_slots"]<=shared["scratch_slots"]):
                raise ValueError("Storage policy is not a controlled source-schedule pair")
            layout_contrasts.append({
                "shape":list(shape),"policy":policy,
                "source_scratch_disjoint":dis["scratch_slots"],
                "source_scratch_reuse":shared["scratch_slots"],
                "source_scratch_saving":dis["scratch_slots"]-shared["scratch_slots"],
                "mapping_conditioned_scratch_saving_established":False,
            })
            baseline=evidence[(shape,"scalar","disjoint")]
            if policy=="scalar":
                continue
            if tuple(e[0] for e in dis["global_last"])!=tuple(
                    e[0] for e in baseline["global_last"]):
                raise ValueError("Causal value-element population mismatch")
            counts={
                "changed_global_last_event_indices":sum(
                    a[1]!=b[1] for a,b in
                    zip(baseline["global_last"],dis["global_last"])),
                "changed_H_only_final_read_rank_or_consumer":sum(
                    a[1:]!=b[1:] for a,b in
                    zip(baseline["read_last"],dis["read_last"])),
                "changed_final_consumer_identity":sum(
                    a[2]!=b[2] for a,b in
                    zip(baseline["read_last"],dis["read_last"])),
            }
            if (counts["changed_global_last_event_indices"]!=
                    dis["summary_global_change"] or
                counts["changed_H_only_final_read_rank_or_consumer"]!=
                    dis["summary_read_change"] or
                counts["changed_final_consumer_identity"]!=
                    dis["summary_consumer_change"] or
                any((
                    dis["summary_global_change"]!=evidence[
                        (shape,policy,layout)]["summary_global_change"] or
                    dis["summary_read_change"]!=evidence[
                        (shape,policy,layout)]["summary_read_change"] or
                    dis["summary_consumer_change"]!=evidence[
                        (shape,policy,layout)]["summary_consumer_change"])
                    for layout in LAYOUTS)):
                raise ValueError("Source counterfactual case CSV disagrees with raw GCC events")
            interventions.append({
                "shape":list(shape),"policy":policy,
                "two_layout_controls_agree":True,
                **counts,
            })
    unique_total=len(interventions)
    if unique_total!=9 or len(layout_contrasts)!=12:
        raise ValueError("Original R2b intervention/layout denominator changed")
    def positive(key):
        return sum(row[key]>0 for row in interventions)
    global_effects=positive("changed_global_last_event_indices")
    read_effects=positive("changed_H_only_final_read_rank_or_consumer")
    reader_effects=positive("changed_final_consumer_identity")
    if (
        summary["original_controlled_phi_contrasts"]!=2*unique_total or
        summary["changed_original_executed_h_last_read_contrasts"]!=2*global_effects or
        summary["source_read_only_rank_or_owner_changed_contrasts"]!=2*read_effects or
        summary["source_read_only_rank_or_owner_unchanged_contrasts"]!=2*(unique_total-read_effects) or
        summary["source_final_reader_identity_changed_contrasts"]!=2*reader_effects or
        summary["source_scratch_saving_layout_pairs"]!=len(layout_contrasts) or
        summary["original_numeric_and_event_passes"]!=24
    ):
        raise ValueError("Original source summary contradicts independent raw causal tallies")
    return {
        "schema":SCHEMA,
        "original_source_revision":summary["source_revision"],
        "prespecified_original_GCC_source_cases":24,
        "original_source_cases_read_only_reverified":preliminary["original_cases_verified"],
        "prespecified_source_storage_layout_pairs":12,
        "layout_replicated_phi_comparisons":18,
        "UNIQUE_graph_Phi_interventions":unique_total,
        "global_event_index_changed_unique_interventions":global_effects,
        "H_only_final_read_rank_or_consumer_changed_unique_interventions":read_effects,
        "H_only_final_read_rank_or_consumer_unchanged_unique_interventions":
            unique_total-read_effects,
        "final_consumer_identity_changed_unique_interventions":reader_effects,
        "original_reported_layout_scoped_read_changes":2*read_effects,
        "both_layout_controls_numerically_and_event_equivalent":True,
        "actual_mapping_conditional_source_scratch_capacity_established":False,
        "source_level_Phi_aware_consumption_order_effect_established":
            read_effects>0,
        "independent_peer_algorithm_or_firstness_established":False,
        "hardware_cycle_or_FPGA_QoR_effect_established":False,
        "trained_RL_performance_established":False,
        "original_GCC_compilations_or_executions_this_audit":0,
        "original_vendor_calls_this_audit":0,
        "unique_interventions":interventions,
        "original_layout_controls":layout_contrasts,
        "warnings":[
            "18 reported layout-scoped contrasts include two equivalent storage layouts for each of nine distinct graph/Phi interventions (not independent statistical samples)",
            "R2b C++ unrolls a newly fixed issue scheduler; do not equate it with a hardware PE cycle schedule",
            "Source-event and H-only READ ordering differences do not prove reduced source memory capacity, HLS resources or original scientific firstness",
            "Results concern only three static MatMul diamonds and one constant input per graph",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--output",help="Optional NEW create-only destination directory")
    a=p.parse_args(argv)
    dest=Path(a.output) if a.output else None
    if dest is not None and (dest.exists() or dest.is_symlink()):
        raise FileExistsError("Independent R2b causal audit report must be create-only")
    result=evaluate_original(a.study)
    if dest is not None:
        dest.mkdir(parents=True)
        (dest/"r2b_causal_audit.json").write_text(
            json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
