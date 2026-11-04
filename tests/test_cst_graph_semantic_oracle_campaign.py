"""Prespecified correlated-map falsification must preserve every case."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_graph_semantic_oracle_campaign import (
    CASES,correlated_map_mutation,run,
)


def test_independent_semantic_oracle_falsification_denominator(tmp_path):
    root=tmp_path/"semantic_oracle"
    summary=run(output=root)
    assert summary["cases"]==2*len(CASES)
    assert summary["positive_controls"]==len(CASES)
    assert summary["negative_correlated_map_mutants"]==len(CASES)
    assert summary["lifetime_only_false_acceptances"]==len(CASES)
    assert summary["full_independent_false_acceptances"]==0
    assert summary["original_vitis_calls"]==0
    with (root/"cases.csv").open() as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==2*len(CASES)
    assert sum(r["independent_semantics_accepted"]=="True" for r in rows)==len(CASES)
    assert all(r["source_lifetime_accepted"]=="True" for r in rows)
    assert json.loads((root/"summary.json").read_text())==summary
    with pytest.raises(FileExistsError):
        run(output=root)
