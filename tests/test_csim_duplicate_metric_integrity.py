"""Duplicate numeric records in a vendor log cannot determine a unique result."""
import pytest
from fpgai_research.toolchains.vitis_hls.report import discover_csim_success

OK="FPGAI_METRICS samples=2 outputs=3 cosine=1 rmse=0 maxabs=0 failures=0"
BAD="FPGAI_METRICS samples=1 outputs=3 cosine=0.5 rmse=3 maxabs=4 failures=4"
STATUS="FPGAI_CSIM_STATUS=completed"

def _parse(tmp_path,value):
    directory=tmp_path/"logs"
    directory.mkdir()
    (directory/"vitis.stdout.log").write_text(value)
    return discover_csim_success(tmp_path)

@pytest.mark.parametrize("log",[
    OK+"\n"+BAD+"\n"+STATUS,
    BAD+"\n"+OK+"\n"+STATUS,
    OK+"\n"+OK+"\n"+STATUS,
    OK+"\nFPGAI_CLASSIFICATION samples=2 compiled_correct=2\n"+STATUS,
    OK+"\nFPGAI_CLASSIFICATION samples=2 compiled_correct=2 reference_correct=2 class_disagreements=0\n"
      +"FPGAI_CLASSIFICATION samples=2 compiled_correct=0 reference_correct=2 class_disagreements=2\n"+STATUS,
])
def test_ambiguous_or_partial_metrics_not_accepted(tmp_path,log):
    record=_parse(tmp_path,log)
    assert record["passed"] is False
    assert record["metrics_missing"] is True

def test_single_complete_record_is_supported(tmp_path):
    record=_parse(tmp_path,OK+"\n"+STATUS)
    assert record["passed"] is True
    assert record["samples"]==2
    assert record["outputs_per_sample"]==3
