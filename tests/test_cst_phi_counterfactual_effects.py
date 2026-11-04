"""Synthetic counterfactual controls, never counted as original GCC paper runs."""
import csv
from hashlib import sha256

import pytest

from fpgai_research.experiments.cst_phi_counterfactual_effects import (
    analyze_rows,analyze_file,
)


def h(x):
    return sha256(str(x).encode()).hexdigest()


def rows():
    result=[]
    for shape in ("[2,2,3,2]","[3,3,5,3]","[2,4,4,3]"):
        for policy in ("scalar","reader0_phi_only","reader1_phi_only",
                       "both_independent"):
            for layout in ("disjoint","last_reader_reuse"):
                r0=policy in ("reader0_phi_only","both_independent")
                r1=policy in ("reader1_phi_only","both_independent")
                result.append({
                    "shape":shape,"policy":policy,"layout":layout,
                    "functional_sha256":h(("functional",shape)),
                    "architecture_sha256":h((shape,policy)),
                    "producer_phi_sha256":h(("producer",shape)),
                    "reader0_phi_sha256":h((shape,"reader0",r0)),
                    "reader1_phi_sha256":h((shape,"reader1",r1)),
                    "cpp_sha256":h((shape,policy,layout,"source")),
                    "source_event_sha256":h((shape,policy,"events")),
                    "last_use_sha256":h((shape,"last-use")),
                    "h_required_reads":"16",
                    "source_scratch_slots":(
                        "100" if layout=="disjoint" else "80"),
                    "disjoint_slots":"100",
                    "numeric_pass":"True","executed_trace_pass":"True",
                })
    return result


def test_changed_phi_source_identity_is_not_a_lifetime_claim():
    result,cases=analyze_rows(rows())
    assert result["original_cases"]==24
    assert result["controlled_phi_contrasts"]==18
    assert result["changed_source_cpp_contrasts"]==18
    assert result["changed_source_event_contrasts"]==18
    assert result["changed_last_use_contrasts"]==0
    assert result["changed_scratch_capacity_contrasts"]==0
    assert result["lifetime_discrimination_established_by_this_study"] is False
    assert result["novelty_relative_to_prior_art_established"] is False


def test_observed_actual_source_last_use_change_is_reported_not_invented():
    data=rows()
    changed=next(r for r in data
                 if r["policy"]=="reader0_phi_only" and
                 r["layout"]=="disjoint")
    changed["last_use_sha256"]=h("genuine-different-timestamp")
    result,cases=analyze_rows(data)
    assert result["changed_last_use_contrasts"]==1
    assert result["lifetime_discrimination_established_by_this_study"] is True
    assert result["novelty_relative_to_prior_art_established"] is False


@pytest.mark.parametrize("fault",[
    "missing_case","duplicate_case","changed_functional",
    "changed_wrong_reader","failed_gcc","broken_scratch_control",
])
def test_noncausal_or_invalid_cases_are_rejected(fault):
    data=rows()
    if fault=="missing_case":
        data.pop()
    if fault=="duplicate_case":
        data[1]=dict(data[0])
    if fault=="changed_functional":
        data[1]["functional_sha256"]=h("mutated-graph")
    if fault=="changed_wrong_reader":
        target=next(r for r in data if r["policy"]=="reader0_phi_only")
        target["reader1_phi_sha256"]=h("should-remain-fixed")
    if fault=="failed_gcc":
        data[0]["numeric_pass"]="False"
    if fault=="broken_scratch_control":
        data[0]["source_scratch_slots"]="70"
    with pytest.raises(ValueError):
        analyze_rows(data)


def test_create_only_secondary_counterfactual_file_has_no_vendor_calls(tmp_path):
    csv_path=tmp_path/"cases.csv"
    data=rows()
    with csv_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    dest=tmp_path/"effects"
    result=analyze_file(input_csv=csv_path,output=dest)
    assert result["original_vendor_calls"]==0
    assert (dest/"controlled_phi_contrasts.csv").is_file()
    assert not result["novelty_relative_to_prior_art_established"]
    with pytest.raises(FileExistsError,match="create-only"):
        analyze_file(input_csv=csv_path,output=dest)


def test_requested_phi_knob_that_does_not_change_effective_owner_is_negative():
    data=rows()
    target=next(r for r in data if r["policy"]=="reader1_phi_only" and
                r["layout"]=="disjoint")
    baseline=next(r for r in data if r["shape"]==target["shape"] and
                  r["policy"]=="scalar" and r["layout"]=="disjoint")
    target["reader1_phi_sha256"]=baseline["reader1_phi_sha256"]
    result,contrasts=analyze_rows(data)
    assert result["ineffective_phi_interventions"]==1
    assert result["effective_phi_interventions"]==17
    assert any(c["requested_phi_change_effective"] is False for c in contrasts)
    assert result["novelty_relative_to_prior_art_established"] is False
