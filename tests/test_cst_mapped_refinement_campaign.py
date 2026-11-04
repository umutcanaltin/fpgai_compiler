"""E0 logical-to-E1 actual source empirical denominators."""
import csv,json,pytest
from fpgai_research.experiments.cst_mapped_refinement_campaign import run

def test_mapped_refinement_study(tmp_path):
    root=tmp_path/"refinement"
    r=run(output=root)
    assert r["matched_source_cases"]==3
    assert r["qualified_source_cases"]==3
    assert r["all_source_last_reads_released"]
    assert r["original_vitis_calls"]==0
    assert len(list(csv.DictReader((root/"cases.csv").open())))==3
    assert json.loads((root/"summary.json").read_text())==r
    with pytest.raises(FileExistsError):
        run(output=root)
