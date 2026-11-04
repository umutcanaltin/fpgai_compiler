"""R2s prospective bounded source Phi scaling and adversarial upper-bound tests.

The original R2r and R2q result folders are never read or modified.
Unit tests use only one of seven fixed source shapes; the full paper
cohort runs by its own create-only CLI on the author's workstation.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import json

import pytest

from fpgai_research.experiments import cst_r2s_phi_scaling as study


@pytest.fixture(scope="module")
def first_shape():
    return study.calculate_rows(shape_limit=1)


def test_exact_prespecified_family_and_widths():
    assert study.SHAPES[-1] == (16,2,16,16)
    assert study.FANOUTS == (2,3,4)
    assert len(study.POLICIES) == 7
    assert study.MAX_CASES == 147
    assert study.widths_for(2,"scalar") == (1,1)
    assert study.widths_for(3,"first_pes2") == (2,1,1)
    assert study.widths_for(3,"last_pes4") == (1,1,4)
    assert study.widths_for(4,"all_pes4") == (4,4,4,4)
    with pytest.raises(ValueError,match="bounded readers"):
        study.widths_for(5,"scalar")
    with pytest.raises(ValueError,match="known PE policy"):
        study.widths_for(2,"pes_unsupported")
    with pytest.raises(ValueError,match="shape_limit"):
        study.calculate_rows(shape_limit=0)
    with pytest.raises(ValueError,match="shape_limit"):
        study.calculate_rows(shape_limit=True)


def test_each_actual_source_layout_has_expected_typed_population(first_shape):
    assert len(first_shape) == 21
    assert [(x["fanout_readers"],x["policy"]) for x in first_shape] == [
        (r,p) for r in (2,3,4) for p in study.POLICIES
    ]
    for case in first_shape:
        r=case["fanout_readers"]
        m,k,n,q=case["shape"]
        assert (m,k,n,q)==(2,2,3,4)
        assert case["source_value_count"] == (
            m*n + (2*r-1)*m*q
        )
        assert case["source_disjoint_slots"] == case["source_value_count"]
        assert case["original_r2h_separate_accumulator_elements"] == r*m*q
        assert case["original_r2h_separate_counter_elements"] == r*m*q
        assert case["original_r2h_declared_mutable_elements"] == (
            case["source_element_exact_committed_slots"]+2*r*m*q
        )
        assert (
            case["source_element_exact_committed_slots"]
            <= case["source_tensor_first_fit_slots"]
            <= case["source_disjoint_slots"]
        )
        assert (
            case["source_element_exact_committed_slots"]
            <= case["hypothetical_first_mac_inplace_peak_slots"]
        )
        # Configuration evidence uses the original emitted-R2h source
        # declaration name; per-intervention CSV intentionally uses its
        # shorter, separately typed comparison-row field.
        assert case["original_r2h_scratch_nominal_16bit_bits"] == (
            case["source_element_exact_committed_slots"] * 16
        )
        assert len(case["source_event_sha256"]) == 64
        assert case["new_gcc_executions"] == 0
        assert case["new_vendor_calls"] == 0


def test_controlled_contrasts_separate_phi_from_memory_convention(first_shape):
    comparisons=study.compare_rows(first_shape,shape_limit=1)
    assert len(comparisons)==18
    assert set(v["policy"] for v in comparisons) == set(study.POLICIES[1:])
    assert all(v["fanout_readers"] in (2,3,4) for v in comparisons)
    by_identity = {
        (r["fanout_readers"], r["policy"]): r for r in first_shape
    }
    for v in comparisons:
        source_case = by_identity[(v["fanout_readers"], v["policy"])]
        assert v["source_scratch_nominal_16bit_bits"] == (
            source_case["original_r2h_scratch_nominal_16bit_bits"]
        )
        assert (v["proposed_reader_width_bound"] ==
                v["fanout_readers"]*(max(v["reader_pes"])-1))
        assert (v["committed_saved_slots"] ==
                v["scalar_committed_slots"]-v["intervened_committed_slots"])
        assert (v["hypothetical_inplace_saved_slots"] ==
                v["scalar_hypothetical_inplace_slots"]-
                v["intervened_hypothetical_inplace_slots"])
        assert (v["committed_savings_share_of_scalar_total_declared_pct"] ==
                100.0*v["committed_saved_slots"]/
                v["scalar_total_declared_mutable_elements"])
    result=study.summarize(first_shape,comparisons,
                           shape_limit=1,revision="a"*40)
    assert result["source_configuration_count"]==21
    assert result["phi_intervention_count"]==18
    assert sum(result["committed_phi_effects"][name] for name in (
        "reduced","tied","increased")) == 18
    assert sum(result["hypothetical_inplace_phi_effects"][name] for name in (
        "reduced","tied","increased")) == 18
    assert result["original_gcc_executions"] == 0
    assert result["new_vitis_calls"] == 0
    assert result["new_vivado_calls"] == 0


def test_conjectured_phi_upper_bound_is_falsifiable_not_asserted(first_shape):
    forged=deepcopy(first_shape)
    scalar=forged[0]
    intervention=forged[1]
    assert intervention["policy"]=="first_pes2"
    intervention["source_element_exact_committed_slots"]=(
        scalar["source_element_exact_committed_slots"]-10
    )
    measured=study.compare_rows(forged,shape_limit=1)
    affected=next(v for v in measured if v["fanout_readers"]==2 and
                  v["policy"]=="first_pes2")
    assert affected["committed_saved_slots"]==10
    assert affected["proposed_reader_width_bound"]==2
    assert affected["committed_observed_satisfies_proposed_upper_bound"] is False
    summary=study.summarize(forged,measured,shape_limit=1,revision="b"*40)
    assert summary["proposed_bound_all_tested_interventions_hold"] is False
    assert len(summary["proposed_bound_violations"]) >= 1


def test_not_called_an_hardware_bram_or_general_graph_theorem(first_shape):
    report=study.summarize(
        first_shape,study.compare_rows(first_shape,shape_limit=1),
        shape_limit=1,revision="c"*40
    )
    assert "tested conjecture" in " ".join(report["limits"])
    assert "not exhaustive" in " ".join(report["limits"])
    assert report["rl_training_calls"]==0
    assert report["new_vivado_calls"]==0


def test_source_pinned_create_only_and_read_only_recomputation(
        first_shape,tmp_path):
    root=tmp_path/"original"
    result=study.run(output=root,shape_limit=1)
    assert result["source_configuration_count"]==21
    assert result["phi_intervention_count"]==18
    before={p.name:p.read_bytes() for p in root.iterdir()}
    verified=study.audit(study=root)
    assert verified["recomputed_source_cases"]==21
    assert verified["recomputed_phi_interventions"]==18
    assert verified["new_original_gcc_calls"]==0
    assert verified["new_vendor_calls"]==0
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError,match="create-only"):
        study.run(output=root,shape_limit=1)


def test_tampered_per_case_value_rejected_even_if_summary_unchanged(tmp_path):
    root=tmp_path/"original"
    study.run(output=root,shape_limit=1)
    p=root/"cases.json"
    rows=json.loads(p.read_text())
    rows[0]["source_disjoint_slots"] += 1
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="capacity rows modified"):
        study.audit(study=root)


def test_tampered_csv_phi_delta_rejected(tmp_path):
    root=tmp_path/"original"
    study.run(output=root,shape_limit=1)
    p=root/"comparisons.csv"
    original=p.read_text()
    first=p.read_text().splitlines()
    assert len(first)>2
    # Duplicate a real row, rather than adding a blank CSV record
    # (which Python's csv parser legitimately ignores).
    p.write_text(original+first[1]+"\n")
    with pytest.raises(ValueError,match="CSV incomplete"):
        study.audit(study=root)


def test_complete_csv_roundtrip_and_single_capacity_tamper_detected(tmp_path):
    root = tmp_path / "original"
    study.run(output=root, shape_limit=1)
    csv_path = root / "comparisons.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    assert len(rows) == 18
    assert fields is not None
    for key in (
        "scalar_committed_slots", "intervened_committed_slots",
        "scalar_hypothetical_inplace_slots",
        "intervened_hypothetical_inplace_slots",
    ):
        assert key in fields
        assert int(rows[0][key]) >= 0
    rows[0]["intervened_committed_slots"] = str(
        int(rows[0]["intervened_committed_slots"]) + 1
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="original CSV changed"):
        study.audit(study=root)


def test_csv_schema_preflight_refuses_partial_original_output(
        monkeypatch, tmp_path):
    original_compare = study.compare_rows

    def extra_field(rows, *, shape_limit):
        compared = original_compare(rows, shape_limit=shape_limit)
        compared[0]["unexpected_unversioned_capacity"] = 42
        return compared

    monkeypatch.setattr(study, "compare_rows", extra_field)
    path = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="CSV schema"):
        study.run(output=path, shape_limit=1)
    assert not path.exists()


def test_missing_csv_column_fails_read_only_audit(tmp_path):
    root = tmp_path / "original"
    study.run(output=root, shape_limit=1)
    path = root / "comparisons.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == list(study.CSV_COLUMNS)
    rows[0][-1] = "changed_source_event_sha256"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CSV columns differ"):
        study.audit(study=root)


def test_symlinked_original_study_receipt_refused(tmp_path):
    root=tmp_path/"original"
    study.run(output=root,shape_limit=1)
    path=root/"summary.json"
    path.unlink()
    path.symlink_to(root/"cases.json")
    with pytest.raises(ValueError,match="missing or symlinked"):
        study.audit(study=root)
