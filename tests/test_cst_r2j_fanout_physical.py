"""R2j: fully enrolled 72-case fanout Vivado OOC bridge, no vendor calls.

Exercises source-pinned source-only study, blocked/native-unknown
denominator preservation, create-only physical export, dry-run and
non-overwrite integrity rules. Does not invent qualified HLS RTL and
never executes Vitis/Vivado from pytest.
"""
from __future__ import annotations

import csv
import json

import pytest

from fpgai_research.experiments.cst_r2i_fanout_native_prepare import prepare
from fpgai_research.experiments.cst_r2j_fanout_physical import (
    TOTAL_CASES,TOTAL_PAIRS,SCHEMA,
    prepare_routed_study,audit_routed_study,
    run_routed_study,export_routed_evidence,
)


@pytest.fixture(scope="module")
def prospective(tmp_path_factory):
    base=tmp_path_factory.mktemp("r2j_source_only")
    native=base/"source_only_R2i_72"
    physical=base/"original_R2j_72"
    prepared=prepare(output=native)
    assert prepared["source_cases"]==TOTAL_CASES==72
    created=prepare_routed_study(native_study=native,output=physical)
    assert created["prespecified_cases"]==72
    assert created["prespecified_pairs"]==36
    return native,physical


def test_unexecuted_native_72_cases_preserve_36_blocked_physical_pairs(prospective):
    native,physical=prospective
    manifest=json.loads((physical/"physical_study_manifest.json").read_text())
    assert manifest["schema"]==SCHEMA
    assert manifest["original_vivado_calls"]==0
    assert manifest["rtl_packaged_cases"]==0
    assert manifest["not_yet_qualified_hls_cases"]==72
    assert len(manifest["cases"])==72
    for i,row in enumerate(manifest["cases"]):
        assert row["case"]==f"case_{i:03d}"
        assert row["group"]==i//2
        assert row["source_storage"]==(
            "disjoint" if i%2==0 else "per_element_optimal")
        assert row["original_hls_qualified"] is False
        assert row["original_hls_csim_qualified"] is False
        assert row["physical_package_status"]=="BLOCKED_NO_QUALIFIED_ORIGINAL_HLS"
        assert not (physical/row["case"]).exists()
    audited=audit_routed_study(physical)
    assert audited["prespecified_cases"]==72
    assert audited["routed_qualified_cases"]==0
    assert audited["vivado_invocation_receipts"]==0
    assert len(audited["cases"])==72
    assert all(not row["routed_observation_valid"] for row in audited["cases"])
    assert all(not row["board_bitstream_qualified"] for row in audited["cases"])


def test_full_36_pair_physical_export_does_not_make_unknown_zero(prospective,tmp_path):
    _,physical=prospective
    dest=tmp_path/"original_physical_export"
    summary=export_routed_evidence(study=physical,output=dest)
    assert summary["prespecified_cases"]==72
    assert summary["prespecified_pairs"]==36
    assert summary["routed_qualified_cases"]==0
    assert summary["routed_qualified_pairs"]==0
    assert summary["unqualified_or_unattempted_cases"]==72
    assert summary["new_native_calls_during_export"]==0
    pairs=json.loads((dest/"matched_pairs.json").read_text())
    assert len(pairs)==36
    assert all(pair["status"]=="UNKNOWN_OR_REJECTED_PHYSICAL_PAIR"
               for pair in pairs)
    assert all(pair["physical_bram_tile_difference"] is None
               for pair in pairs)
    with (dest/"pairs.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f)))==36
    with pytest.raises(FileExistsError,match="create-only"):
        export_routed_evidence(study=physical,output=dest)


def test_first_qualified_BRAM_estimate_pair_dry_run_is_zero_vendor(
        prospective):
    _,physical=prospective
    result=run_routed_study(physical,start_case=40,max_cases=2)
    assert result["status"]=="DRY_RUN_NO_VIVADO"
    assert result["new_vivado_calls"]==0
    assert [r["case"] for r in result["cases"]]==["case_040","case_041"]
    assert all(r["status"]=="BLOCKED_NO_QUALIFIED_ORIGINAL_HLS"
               for r in result["cases"])
    with pytest.raises(ValueError,match="confirm-isolated-native"):
        run_routed_study(physical,execute_native=True,
                         start_case=40,max_cases=2)
    with pytest.raises(ValueError,match="bounded original Vivado case range"):
        run_routed_study(physical,start_case=71,max_cases=2)


def test_physical_source_manifest_tamper_or_reuse_is_rejected(
        prospective,tmp_path):
    native,physical=prospective
    with pytest.raises(FileExistsError,match="create-only"):
        prepare_routed_study(
            native_study=native,output=physical)
    doc=physical/"physical_study_manifest.json"
    original=doc.read_text()
    tampered=json.loads(original)
    tampered["cases"][40]["case"]="case_999"
    doc.write_text(json.dumps(tampered,sort_keys=True)+"\n")
    try:
        with pytest.raises(ValueError,match="Unexpected physical study case order"):
            audit_routed_study(physical)
    finally:
        doc.write_text(original)
