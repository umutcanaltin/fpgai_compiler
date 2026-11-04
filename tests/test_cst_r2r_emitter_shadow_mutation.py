"""R2r exact source emission, dynamic per-slot memory guards and E1 mutations.

Tests are bounded local GNU GCC source executions. Full 216-mutant
original scientific run is a separate CLI command, never CI/Actions.
"""
from __future__ import annotations

import json
import shutil

import pytest

from fpgai_research.experiments import cst_r2r_emitter_shadow_mutation as sut
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    emit_allocated_fanout_cpp,
)


@pytest.fixture(scope="module")
def gcc_cohort(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("R2r actual original GCC required")
    root = tmp_path_factory.mktemp("r2r_test") / "original"
    result = sut.run(output=root, graphs=1, mutants_per_graph=36)
    return root, result


def test_actual_original_emitter_instrumentation_covers_every_source_access():
    arch=sut._arch(sut.CASES[0])
    original=emit_allocated_fanout_cpp(arch,layout="per_element_optimal")
    modified=sut.shadow_instrument(arch,original)
    assert modified.cpp!=original.cpp
    assert sut._hash(modified.cpp)==modified.manifest["source_cpp_sha256"]
    assert modified.manifest["source_slot_map"]==original.manifest["source_slot_map"]
    assert "shadow_remaining" in modified.cpp
    assert "shadow_owner" in modified.cpp
    assert "shadow_finish();" in modified.cpp
    assert "FPGAI_SLOT" in modified.cpp
    assert "shadow_store(h_store_addr," in modified.cpp
    assert "shadow_load(h_addr," in modified.cpp
    assert "shadow_store(reader_store_addr," in modified.cpp
    assert "shadow_load(left_address," in modified.cpp
    assert "shadow_load(right_address," in modified.cpp
    assert "shadow_store(add_store_addr," in modified.cpp
    assert "shadow_load(source_addr," in modified.cpp


@pytest.mark.parametrize("family",sut.MUTATION_FAMILIES)
def test_each_mutation_changes_real_generated_source_in_a_unique_way(family):
    arch=sut._arch(sut.CASES[0])
    original=sut.shadow_instrument(
        arch,emit_allocated_fanout_cpp(arch,layout="per_element_optimal"))
    variants=[sut.mutate_instrumented(original,family=family,delta=d)
              for d in sut.DELTAS]
    assert len({v.manifest["source_cpp_sha256"] for v in variants})==4
    for v in variants:
        assert sut._hash(v.cpp)==v.manifest["source_cpp_sha256"]
        assert v.cpp!=original.cpp
        assert "FPGAI_SLOT" in v.cpp


def test_full_36_mutant_family_matrix_is_GCC_executed_and_not_hidden(gcc_cohort):
    root,summary=gcc_cohort
    assert summary["original_graphs"]==1
    assert summary["original_mutant_count"]==36
    assert summary["planned_full_mutant_population"]==216
    assert summary["original_nonmutated_GCC_baselines_qualified"]==1
    assert summary["original_gcc_attempts"]==37
    assert sum(summary["mutant_status_counts"].values())==36
    assert summary["surviving_mutants"]==0
    assert summary["compile_invalid_mutants"]==0
    assert summary["preflight_invalid_mutants"]==0
    assert summary["unknown_runtime_failures"]==0
    assert summary["true_unsafe_mutants_detected"]==36
    assert summary["original_vitis_calls"]==0
    assert summary["original_vivado_calls"]==0
    assert summary["rl_policies_trained"]==0
    receipts=[json.loads(p.read_text()) for p in
              sorted((root/"graph_000").glob("mutant_*.json"))]
    assert len(receipts)==36
    assert set(row["family"] for row in receipts)==set(sut.MUTATION_FAMILIES)
    assert any(row["status"]=="DETECTED_DYNAMIC_SHADOW" for row in receipts)
    assert any(row["status"]=="DETECTED_R2H_CHECKER" for row in receipts)
    audit=sut.audit(root)
    assert audit["original_mutants"]==36
    assert audit["status_counts"]==summary["mutant_status_counts"]
    assert audit["original_GCC_compilations_during_audit"]==0


def test_original_mutant_study_create_only_and_read_only_audit(gcc_cohort):
    root,summary=gcc_cohort
    before={p.name:p.read_bytes() for p in root.rglob("*") if p.is_file()}
    sut.audit(root)
    assert before=={p.name:p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError,match="create-only"):
        sut.run(output=root,graphs=1,mutants_per_graph=36)


def test_tampered_mutant_source_is_rejected_by_audit(gcc_cohort,tmp_path):
    source=gcc_cohort[0]
    copy=tmp_path/"forged"
    shutil.copytree(source,copy)
    p=copy/"graph_000"/"mutant_000.cpp"
    p.write_text(p.read_text()+"\n// hidden source change\n")
    with pytest.raises(ValueError,match="mismatched source mutant evidence"):
        sut.audit(copy)


def test_forged_detection_outcome_without_shadow_error_is_rejected(
        gcc_cohort,tmp_path):
    copy=tmp_path/"forged"
    shutil.copytree(gcc_cohort[0],copy)
    path=copy/"graph_000"/"mutant_000.json"
    row=json.loads(path.read_text())
    row["status"]="DETECTED_DYNAMIC_SHADOW"
    row["error_type"]="RuntimeError"
    row["reason"]="not actually observed"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match="shadow classification unsupported"):
        sut.audit(copy)


def test_no_unbounded_graphs_or_unknown_mutation_classes():
    arch=sut._arch(sut.CASES[0])
    source=sut.shadow_instrument(
        arch,emit_allocated_fanout_cpp(arch,layout="per_element_optimal"))
    with pytest.raises(ValueError,match="prespecified"):
        sut.mutate_instrumented(source,family="none",delta=1)
    with pytest.raises(ValueError,match="graphs"):
        sut.run(output="/tmp/r2r_never",graphs=0)
    with pytest.raises(ValueError,match="mutants_per_graph"):
        sut.run(output="/tmp/r2r_never",mutants_per_graph=0)
