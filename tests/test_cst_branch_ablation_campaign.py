"""Explicit comparator outcomes, reproducibility and honest evidence ceilings."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_branch_ablation_campaign import run,probe_catalog
from fpgai_research.model_ir.branch_ablation import MODES,ablation_admission


def test_complete_contract_detects_every_prespecified_mutation():
    arch,probes=probe_catalog()
    assert len(probes)>=7
    for label,layout,events,truth in probes:
        assert ablation_admission(
            arch,layout,events,mode="full_phi_and_last_use")==truth


def test_owner_only_mutant_distinguishes_cardinality_from_phi_conditioned_certificate():
    arch,probes=probe_catalog()
    label,offsets,events,_=next(x for x in probes if x[0]=="forged_phi_owner")
    assert ablation_admission(arch,offsets,events,mode="read_cardinality_no_phi")
    assert not ablation_admission(arch,offsets,events,mode="full_phi_and_last_use")


def test_first_read_ack_proxy_is_not_sufficient_for_multiple_matmul_reads():
    arch,probes=probe_catalog()
    label,offsets,events,_=next(x for x in probes
                                if x[0]=="missing_second_output_read")
    assert ablation_admission(arch,offsets,events,mode="first_read_ack")
    assert not ablation_admission(arch,offsets,events,mode="full_phi_and_last_use")
    with pytest.raises(ValueError,match="Unsupported"):
        ablation_admission(arch,offsets,events,mode="imaginary_method")


def test_prespecified_ablation_evidence_keeps_all_denominators(tmp_path):
    folder=tmp_path/"ablation"
    report=run(output=folder)
    assert report["cases"]==7
    assert report["rows"]==7*len(MODES)
    assert report["full_spec_accepted"]==2
    assert report["full_spec_rejected"]==5
    assert report["mode_results"]["full_phi_and_last_use"]=={
        "accepted":2,"false_accepts_relative_to_full":0,
        "false_rejects_relative_to_full":0}
    assert report["mode_results"]["metadata_only"]["false_accepts_relative_to_full"]==5
    assert report["mode_results"]["read_cardinality_no_phi"]["false_accepts_relative_to_full"]>=1
    assert report["mode_results"]["first_read_ack"]["false_accepts_relative_to_full"]>=1
    assert report["original_vitis_calls"]==0
    with (folder/"cases.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==7*len(MODES)
    assert json.loads((folder/"summary.json").read_text())==report
    with pytest.raises(FileExistsError):
        run(output=folder)
