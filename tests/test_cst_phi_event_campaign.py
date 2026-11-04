"""Fixed R2 candidate-projection experiment must retain complete denominators."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_phi_event_campaign import run

def test_phi_conditioned_event_campaign(tmp_path):
    dest=tmp_path/"phi_study"
    result=run(output=dest)
    assert result["candidate_mappings"]==3
    assert result["canonical_verified"]==3
    assert result["stale_mapping_rejections"]==2
    assert result["missing_reduction_rejections"]==3
    assert result["early_read_rejections"]==3
    assert result["vitis_calls"]==0
    assert len(list(csv.DictReader((dest/"cases.csv").open())))==3
    assert json.loads((dest/"summary.json").read_text())==result
    with pytest.raises(FileExistsError,match="Create-only"):
        run(output=dest)
