from fpgai_research.toolchains.vitis_hls.report import discover_csim_success

def test_csim_classification_is_separate_metric(tmp_path):
    logs=tmp_path/"logs"
    logs.mkdir()
    (logs/"vitis.stdout.log").write_text(
      "FPGAI_METRICS samples=2 outputs=3 cosine=0.9 rmse=0.02 maxabs=0.1 failures=0\n"
      "FPGAI_CLASSIFICATION samples=2 compiled_correct=1 reference_correct=2 class_disagreements=1\n"
      "FPGAI_CSIM_STATUS=completed\n")
    record=discover_csim_success(tmp_path)
    assert record["passed"] is True
    assert record["classification"]["compiled_top1_accuracy"]==0.5
    assert record["classification"]["reference_top1_accuracy"]==1.0
    assert record["classification"]["class_disagreement_rate"]==0.5

def test_csim_inconsistent_classification_is_rejected(tmp_path):
    logs=tmp_path/"logs"
    logs.mkdir()
    (logs/"vitis.stdout.log").write_text(
      "FPGAI_METRICS samples=2 outputs=3 cosine=0.9 rmse=0.02 maxabs=0.1 failures=0\n"
      "FPGAI_CLASSIFICATION samples=3 compiled_correct=3 reference_correct=2 class_disagreements=0\n"
      "FPGAI_CSIM_STATUS=completed\n")
    assert discover_csim_success(tmp_path)["metrics_missing"] is True
