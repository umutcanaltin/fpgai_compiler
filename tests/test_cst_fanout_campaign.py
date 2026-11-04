"""Prespecified fan-out E0 controls preserve honest source/RTL boundary."""
import csv
import json
import pytest
from fpgai_research.experiments.cst_fanout_campaign import run

def test_branch_consumer_study_preserves_full_controls(tmp_path):
    root=tmp_path/"study"
    summary=run(output=root)
    assert summary["cases"]==2
    assert summary["valid_controls"]==2
    assert summary["early_alias_rejections"]==1
    assert summary["executed_source_cases"]==0
    assert summary["native_vitis_calls"]==0
    with (root/"cases.csv").open() as fp:
        cases=list(csv.DictReader(fp))
    assert len(cases)==2
    assert cases[0]["early_commit_reason"]=="OVERWRITE_BEFORE_LAST_READER"
    assert cases[1]["early_commit_reason"]=="DUPLICATE_COMMIT"
    assert json.loads((root/"summary.json").read_text())==summary
    with pytest.raises(FileExistsError):
        run(output=root)
