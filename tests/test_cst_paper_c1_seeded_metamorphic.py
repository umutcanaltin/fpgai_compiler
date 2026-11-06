"""C1 seeded independent-oracle metamorphic source tests.

All test data are bounded-source examples and synthetic negative
mutations. Passing tests is not formal proof of general C++/FPGA soundness.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_paper_c1_seeded_metamorphic as sut


def test_new_shapes_do_not_duplicate_prespecified_r3j_holdout():
    cases = sut.configurations(seeds=(111,222), shapes_per_group=2)
    assert len(cases) == 24
    assert len(set(cases)) == 24
    assert len({(c[1], tuple(c[2:6]), c[-1]) for c in cases}) == 24
    assert all(tuple(c[2:6]) not in sut.SHAPES for c in cases)
    assert sut.configurations(seeds=(111,222), shapes_per_group=2) == cases
    assert sut.configurations(seeds=(111,333), shapes_per_group=2) != cases
    assert {c[1] for c in cases} == {2,3,4}
    assert {c[-1] for c in cases} == {"scalar", "mixed_phi"}


def test_new_mapped_event_streams_have_independently_verified_controls():
    one = sut.configurations(seeds=(111,),shapes_per_group=1)
    assert len(one) == 6
    for case in one:
        result = sut.evaluate(case)
        assert result["seed"] == 111
        assert tuple(x["mutation"] for x in result["mutations"]) == sut.MUTATIONS
        assert result["mutations"][0]["full_contract_admissible"] is True
        assert result["mutations"][1]["value_safe"] is True
        assert result["mutations"][1]["issue_conformant"] is False
        assert result["mutations"][1]["full_contract_admissible"] is False
        assert all(not item["value_safe"] for item in result["mutations"][2:])
        assert all(len(x["event_trace_sha256"]) == 64
                   for x in result["mutations"])


def test_source_only_quick_cohort_has_exact_denominators_and_limits():
    report = sut.run(seeds=(111,), shapes_per_group=1)
    assert report["profiles"] == 6
    assert report["traces"] == 54
    assert report["value_safe_wrong_issue_controls"] == 6
    assert report["full_contract_accepts_wrong_issue_controls"] == 0
    assert report["canonical_valid_controls"] == 6
    assert report["other_selected_unsafe_controls"] == 42
    assert report["value_oracle_disagreements"] == 0
    assert report["issue_oracle_disagreements"] == 0
    assert report["original_gcc_calls"] == 0
    assert report["original_vitis_calls"] == 0
    assert report["original_vivado_calls"] == 0
    assert report["hardware_latency"] is None


@pytest.mark.parametrize("opts",[
    {"seeds":(111,111)},
    {"seeds":()},
    {"seeds":(111.0,)},
    {"shapes_per_group":0},
    {"shapes_per_group":4},
])
def test_invalid_or_changed_seed_protocol_fails_closed(opts):
    with pytest.raises(ValueError):
        sut.configurations(**opts)


def test_genuine_oracle_disagreement_not_allowed(monkeypatch):
    case = sut.configurations(seeds=(111,),shapes_per_group=1)[0]
    actual = sut.reference_event_oracle
    def contradictory(arch, events, offsets):
        safe, order = actual(arch,events,offsets)
        return not safe, order
    monkeypatch.setattr(sut, "reference_event_oracle", contradictory)
    with pytest.raises(AssertionError,match="independent model disagreement"):
        sut.evaluate(case)


def test_no_overwriting_original_receipts_and_sha_drift_rejected(tmp_path):
    report = sut.run(seeds=(111,),shapes_per_group=1)
    snapshot = json.dumps(report,sort_keys=True)
    out = tmp_path/"new_source_only"
    sut.create_only(report,output=out,source_revision="f"*40)
    a = sut.audit(out)
    assert a["profiles"] == 6 and a["traces"] == 54
    assert len(a["report_sha256"]) == 64
    assert json.dumps(report,sort_keys=True) == snapshot
    assert sha256((out/"report.json").read_bytes()).hexdigest() == a["report_sha256"]
    with pytest.raises(FileExistsError):
        sut.create_only(report,output=out,source_revision="f"*40)
    (out/"report.json").write_text((out/"report.json").read_text()+" ")
    with pytest.raises(ValueError,match="SHA drift"):
        sut.audit(out)
