"""Prespecified abstract bank-port evidence counts are not hardware results."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_branch_bank_port_campaign import run,SHAPES


def test_prespecified_schedule_unknown_and_port_reject_counts(tmp_path):
    folder=tmp_path/"ports"
    summary=run(output=folder)
    assert summary["cases"]==3*len(SHAPES)
    assert summary["unknown_unscheduled"]==len(SHAPES)
    assert summary["abstract_feasible_only"]==len(SHAPES)
    assert summary["declared_port_oversubscriptions"]==len(SHAPES)
    assert summary["hardware_qualified_cases"]==0
    assert summary["original_vitis_calls"]==0
    assert json.loads((folder/"summary.json").read_text())==summary
    with (folder/"cases.csv").open() as reader:
        rows=list(csv.DictReader(reader))
    assert len(rows)==3*len(SHAPES)
    assert all(row["hardware_qualified"]=="False" for row in rows)
    with pytest.raises(FileExistsError):
        run(output=folder)
