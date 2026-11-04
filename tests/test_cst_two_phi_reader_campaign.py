"""Prespecified bounded two-independent-phi MatMul readers experiment, original GCC."""
import csv
import json
import shutil

import pytest

from fpgai_research.experiments.cst_two_phi_reader_campaign import (
    POLICIES,SHAPES,run,
)


@pytest.mark.skipif(shutil.which("g++") is None,reason="Original host GCC required")
def test_two_phi_reader_study_retains_original_complete_denominator(tmp_path):
    target=tmp_path/"two_readers"
    summary=run(output=target)
    count=2*len(POLICIES)*len(SHAPES)
    assert count==24
    assert summary["cases"]==count
    assert summary["numeric_passes"]==count
    assert summary["source_trace_passes"]==count
    assert summary["source_scratch_saving_pairs"]==count//2
    assert summary["native_vitis_calls"]==0
    assert json.loads((target/"summary.json").read_text())==summary
    with (target/"cases.csv").open() as reader:
        rows=list(csv.DictReader(reader))
    assert len(rows)==count
    by_graph={}
    for row in rows:
        key=row["shape"]
        by_graph.setdefault(key,[]).append(row)
        assert row["numeric_pass"]=="True"
        assert row["executed_trace_pass"]=="True"
        assert int(row["original_vitis_calls"])==0
        # An event certificate includes Phi read-owner fields. Holding the
        # Functional graph fixed does NOT make mapping-specific event hashes
        # identical; only a paired layout with unchanged mapping must match.
    for group in by_graph.values():
        assert len({r["functional_sha256"] for r in group})==1
        assert len({r["producer_phi_sha256"] for r in group})==1
        assert len({r["reader0_phi_sha256"] for r in group})>1
        assert len({r["reader1_phi_sha256"] for r in group})>1
        assert len({r["source_event_sha256"] for r in group})>1
        assert len({r["architecture_sha256"] for r in group})==len(POLICIES)
        for first,second in zip(group[::2],group[1::2]):
            assert int(first["source_scratch_slots"])>int(second["source_scratch_slots"])
            assert first["source_event_sha256"]==second["source_event_sha256"]
    with pytest.raises(FileExistsError):
        run(output=target)
