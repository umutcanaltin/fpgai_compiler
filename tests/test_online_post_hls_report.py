"""Report-only structural tests: all cases are synthetic fixtures, NO vendor runs.

Synthetic fixture values test the CSV/plot contract, not scientific hardware
results. Real qualification remains the separate original-case audit.
"""
import json
import pytest

from rl_environment.report import export


def _new_fake_report_study(tmp_path, *, normalized):
    root=tmp_path/"fixture_study"
    root.mkdir()
    (root/"campaign.json").write_text(json.dumps({
        "source_proxy_reward_used":False,
    }))
    (root/"summary.json").write_text(json.dumps({
        "heldout_comparison":[
            {"task_id":"test_a","method":"online_drl",
             "best_hls_worst_case_cycles":40,
             **({"normalized_hls_improvement":.20} if normalized else {})},
            {"task_id":"test_a","method":"random",
             "best_hls_worst_case_cycles":45,
             **({"normalized_hls_improvement":.10} if normalized else {})},
        ],
    }))
    rows=[{
        "training_networks_completed":n,
        "validation_networks":2,
        "qualified_validation_networks":2,
        "mean_best_qualified_hls_cycles":c,
        "online_training_updates":n*2,
        "random_qualified_validation_networks":2,
        "random_mean_best_qualified_hls_cycles":50,
        **({
            "paired_normalized_valid_networks":2,
            "drl_mean_normalized_hls_improvement":p,
            "random_mean_normalized_hls_improvement":.1
        } if normalized else {})
    } for n,c,p in [(25,47,.08),(50,43,.16)]]
    (root/"learning_curve.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in rows))
    (root/"native_steps.jsonl").write_text(json.dumps({
        "source_proxy_reward":False,
        "native_receipt_sha256":"a"*64,
        "native_vendor_seconds":12.5,
    })+"\n")
    return root


def test_report_uses_matched_baseline_normalized_curve_when_available(tmp_path):
    root=_new_fake_report_study(tmp_path,normalized=True)
    out=tmp_path/"report"
    result=export(root,out)
    assert result["paired_test_tasks_with_qualified_baselines"]==1
    assert result["paired_test_mean_drl_normalized_improvement"]==.20
    assert result["paired_test_mean_random_normalized_improvement"]==.10
    assert result["total_original_vitis_seconds_recorded"]==12.5
    csv=(out/"native_learning_curve.csv").read_text()
    assert "drl_mean_normalized_hls_improvement" in csv
    assert "paired_normalized_valid_networks" in csv
    if result["graph_created"]:
        assert result["validation_curve_metric"]=="paired_normalized_hls_improvement"
        assert (out/"native_validation_learning_curve.png").is_file()
    with pytest.raises(FileExistsError):
        export(root,out)


def test_report_legacy_original_hls_smoke_without_normalized_fields(tmp_path):
    root=_new_fake_report_study(tmp_path,normalized=False)
    out=tmp_path/"legacy_report"
    result=export(root,out)
    assert result["paired_test_tasks_with_qualified_baselines"]==0
    assert result["paired_test_mean_drl_normalized_improvement"] is None
    if result["graph_created"]:
        assert result["validation_curve_metric"]=="mean_qualified_hls_worst_case_cycles"
    assert (out/"report.json").is_file()


def test_report_rejects_incomplete_or_source_proxy_study(tmp_path):
    root=_new_fake_report_study(tmp_path,normalized=True)
    (root/"summary.json").unlink()
    with pytest.raises(ValueError,match="incomplete"):
        export(root,tmp_path/"incomplete")
    (root/"summary.json").write_text(json.dumps({"heldout_comparison":[]}))
    (root/"campaign.json").write_text(json.dumps({"source_proxy_reward_used":True}))
    with pytest.raises(ValueError,match="Native-only"):
        export(root,tmp_path/"proxy")
