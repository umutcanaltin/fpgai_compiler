"""R2t exact tensor comparator: solver proof checks and source research gates.

Independent small brute-force offset oracle enumerates *all bases*,
not the R2t spatial-permutation algorithm. No Vitis/Vivado/RL execution.
"""
from __future__ import annotations

import itertools
import json
import shutil

import pytest

from fpgai_research.model_ir.source_allocation_baselines import (
    WholeTensorBlock, whole_tensor_first_fit,
)
from fpgai_research.model_ir.tensor_exact_allocation import (
    exact_tensor_bases,
)
from fpgai_research.experiments import cst_r2t_tensor_optimum as study


def _independent_brute(blocks):
    """Enumerate bases 0..disjoint, with an independently coded overlap."""
    cap=sum(b.size for b in blocks)
    for limit in range(max(b.size for b in blocks),cap+1):
        options=[range(limit-b.size+1) for b in blocks]
        for bases in itertools.product(*options):
            safe=True
            for i,left in enumerate(blocks):
                for j in range(i+1,len(blocks)):
                    right=blocks[j]
                    lifetime_overlap=not (
                        left.last_read < right.first_commit or
                        right.last_read < left.first_commit
                    )
                    space_overlap=not (
                        bases[i]+left.size <= bases[j] or
                        bases[j]+right.size <= bases[i]
                    )
                    if lifetime_overlap and space_overlap:
                        safe=False
                        break
                if not safe:
                    break
            if safe:
                return limit
    raise AssertionError("Independent disjoint reference should always fit")


def test_global_tensor_exact_improves_greedy_with_real_counterexample():
    blocks=(
        WholeTensorBlock("0",1,1,2),
        WholeTensorBlock("1",3,2,8),
        WholeTensorBlock("2",6,4,7),
        WholeTensorBlock("3",5,3,8),
    )
    greedy,_=whole_tensor_first_fit(blocks)
    exact=exact_tensor_bases(blocks)
    assert greedy==15
    assert exact.status=="OPTIMAL_PROVED"
    assert exact.exact_feasible_slots==14
    assert exact.exact_feasible_slots==_independent_brute(blocks)
    assert exact.greedy_feasible_slots==15
    assert len(exact.bases)==4


@pytest.mark.parametrize("lengths",(
    ((1,0,2),(3,1,4),(2,3,5)),
    ((2,0,3),(1,1,2),(3,4,5)),
    ((1,0,4),(1,0,4),(1,0,4)),
    ((3,0,1),(4,2,3),(5,4,5)),
))
def test_exact_tensor_spatial_order_matches_independent_small_bruteforce(lengths):
    blocks=tuple(
        WholeTensorBlock(f"t{i}",size,start,end)
        for i,(size,start,end) in enumerate(lengths)
    )
    exact=exact_tensor_bases(blocks)
    assert exact.status=="OPTIMAL_PROVED"
    assert exact.exact_feasible_slots==_independent_brute(blocks)
    assert exact.lower_bound <= exact.exact_feasible_slots


def test_unknown_budget_is_not_falsely_called_proved_optimum():
    blocks=(
        WholeTensorBlock("0",1,1,2),
        WholeTensorBlock("1",3,2,8),
        WholeTensorBlock("2",6,4,7),
        WholeTensorBlock("3",5,3,8),
    )
    result=exact_tensor_bases(blocks,max_search_nodes=1)
    assert result.status=="UNKNOWN_BUDGET"
    assert result.exact_feasible_slots is None
    assert result.bases is None
    assert result.greedy_feasible_slots==15
    with pytest.raises(ValueError,match="node budget"):
        exact_tensor_bases(blocks,max_search_nodes=0)


def test_fail_closed_on_duplicate_tensor_and_incorrect_lifetimes():
    with pytest.raises(ValueError,match="invalid tensor"):
        exact_tensor_bases((
            WholeTensorBlock("foo",2,0,3),
            WholeTensorBlock("foo",3,1,2),
        ))
    with pytest.raises(ValueError,match="invalid tensor"):
        exact_tensor_bases((WholeTensorBlock("foo",1,2,1),))
    with pytest.raises(ValueError,match="1..9"):
        exact_tensor_bases(())


@pytest.fixture(scope="module")
def original_source_36():
    return study.calculate_catalog()


def test_all_36_r2q_source_cases_are_checked_against_exact_whole_tensor(
        original_source_36):
    assert len(original_source_36)==36
    for row in original_source_36:
        assert row["tensor_optimum_status"] in (
            "OPTIMAL_PROVED","UNKNOWN_BUDGET")
        assert (
            row["committed_element_exact_slots"]
            <= row["proven_tensor_lower_bound_slots"]
            <= row["greedy_whole_tensor_slots"]
            <= row["disjoint_source_slots"]
        )
        if row["tensor_optimum_status"]=="OPTIMAL_PROVED":
            assert (
                row["committed_element_exact_slots"]
                <= row["proven_optimal_whole_tensor_slots"]
                <= row["greedy_whole_tensor_slots"]
            )
            assert row["tensor_layout_bases"] is not None
            assert len(row["source_event_sha256"])==64
        else:
            assert row["proven_optimal_whole_tensor_slots"] is None
            assert row["tensor_layout_bases"] is None
    summary=study.summarize(original_source_36,"a"*40,250_000)
    assert summary["original_source_cases"]==36
    assert (summary["optimality_proved_cases"]+
            summary["budget_unknown_cases"]==36)
    assert summary["original_Vitis_calls"]==0
    assert summary["original_GCC_calls"]==0


def test_original_tensor_optimum_output_create_only_and_audit(tmp_path):
    root=tmp_path/"original"
    summary=study.run(output=root)
    assert summary["original_source_cases"]==36
    before={p.name:p.read_bytes() for p in root.iterdir()}
    checked=study.audit(root)
    assert checked["original_cases"]==36
    assert checked["original_GCC_calls_during_audit"]==0
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError,match="create-only"):
        study.run(output=root)


def test_tampered_case_and_csv_detected(tmp_path):
    root=tmp_path/"original"
    study.run(output=root)
    fp=root/"cases.json"
    original=fp.read_bytes()
    cases=json.loads(original)
    cases[0]["greedy_whole_tensor_slots"]+=1
    fp.write_text(json.dumps(cases))
    with pytest.raises(ValueError,match="recalculation"):
        study.audit(root)
    fp.write_bytes(original)
    csvfile=root/"comparisons.csv"
    csvfile.write_text(csvfile.read_text().replace(
        "case_000","forged_000",1))
    with pytest.raises(ValueError,match="CSV source measurement"):
        study.audit(root)
