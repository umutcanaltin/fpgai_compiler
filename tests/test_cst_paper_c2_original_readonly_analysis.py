"""Read-only C2 signed research result interpretation tests (synthetic fixtures).

The fixture numbers below are NOT the author's measured original wall times.
No compiler, HLS or Vivado executes in this test module.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import (
    cst_paper_c2_original_readonly_analysis as sut,
)


def original_format_synthetic(tmp_path):
    root = tmp_path / "synthetic_original"
    root.mkdir()
    rows = []
    for name, width in sorted(sut.ORIGINAL_COHORTS):
        for stream in sut.STREAMS:
            for seed in sut.SEEDS:
                for queries in sut.QUERIES:
                    indices = ([0] * queries if stream == "repeat_certified"
                               else [0, 2, 5][:queries] * (queries // 3) +
                                    [0, 2, 5][:queries % 3])
                    assert len(indices) == queries
                    result_sha = "b"*64
                    for order, method in enumerate(sut.METHODS):
                        calls = (queries if method == "direct" else
                                 len(set(indices)) if method == "lazy" else 8)
                        rows.append({
                            "cohort": name, "width": width, "stream": stream,
                            "seed": seed, "queries": queries,
                            "method": method, "request_indices": indices,
                            "distinct_requested": len(set(indices)),
                            "execution_order": order,
                            "source_compiler_calls": calls,
                            "wall_ns": calls*1000+order*10,
                            "matched_compiled_results": True,
                            "source_parent_sha256": "a"*64,
                            "result_sha256": result_sha,
                        })
    obj = {
        "schema": sut.INPUT_SCHEMA,
        "experiment_source_revision": sut.ORIGINAL_SOURCE_REVISION,
        "cohorts": 6, "measured_cases": 324,
        "seeds": list(sut.SEEDS), "query_counts": list(sut.QUERIES),
        "streams": list(sut.STREAMS),
        "original_source_preflight_compiler_calls_excluded": 48,
        "all_compiled_source_results_matched": True,
        "native_gcc_calls": 0, "native_vitis_calls": 0,
        "native_vivado_calls": 0, "trained_rl_agents": 0,
        "hardware_latency": None, "records": rows,
    }
    report = root/"report.json"
    report.write_text(json.dumps(obj, sort_keys=True) + "\n")
    digest = sha256(report.read_bytes()).hexdigest()
    (root/"artifact_sha256.json").write_text(
        json.dumps({"report.json":digest}) + "\n")
    return root, digest


def test_original_style_324_case_paired_analysis_and_eager_negative(tmp_path):
    root,digest = original_format_synthetic(tmp_path)
    result = sut.analyze(root, pinned_report_sha256=digest)
    assert result["schema"] == sut.SCHEMA
    assert result["original_measured_method_rows"] == 324
    assert result["paired_trial_count"] == 108
    assert result["new_source_compiler_calls_during_analysis"] == 0
    assert result["hardware_latency"] is None
    assert len(result["by_condition"]) == 6
    assert result["original_source_preflight_calls_excluded"] == 48
    assert all(c["paired_workloads"] == 18 for c in result["by_condition"])
    one = next(c for c in result["by_condition"]
               if c["queries"] == 1 and c["stream"] == "repeat_certified")
    assert one["strategies"]["eager"]["median_compiler_calls"] == 8
    assert one["strategies"]["lazy"]["median_compiler_calls"] == 1
    assert (one["paired_wall_time_comparisons"]["lazy_vs_eager"]
            ["left_faster_pairs"]) == 18
    assert result["total_measured_compiler_calls"]["eager"] == 108*8
    assert result["total_measured_compiler_calls"]["lazy"] <= 108*8
    assert result["all_compiler_results_matched"] is True


def test_author_pinned_sha_fail_closed_if_wrong(tmp_path):
    root,digest = original_format_synthetic(tmp_path)
    with pytest.raises(ValueError,match="SHA differs"):
        sut.analyze(root)
    assert digest != sut.ORIGINAL_REPORT_SHA256


@pytest.mark.parametrize("mutator",[
    lambda x: x.update(experiment_source_revision="z"*40),
    lambda x: x["records"].pop(),
    lambda x: x["records"][0].update(source_compiler_calls=999),
    lambda x: x["records"][0].update(result_sha256="f"*64),
    lambda x: x["records"][0].update(request_indices=[8]),
    lambda x: x["records"][0].update(native_vivado_calls=1),
    lambda x: x.update(all_compiled_source_results_matched=False),
])
def test_tampered_or_incomplete_original_data_rejected(tmp_path,mutator):
    root,_ = original_format_synthetic(tmp_path)
    path = root/"report.json"
    data = json.loads(path.read_text())
    mutator(data)
    path.write_text(json.dumps(data,sort_keys=True) + "\n")
    digest = sha256(path.read_bytes()).hexdigest()
    (root/"artifact_sha256.json").write_text(
        json.dumps({"report.json":digest}) + "\n")
    if "native_vivado_calls" in data["records"][0]:
        # This row-level field is not part of the original fixed schema;
        # reject synthetic extra columns by requiring provenance to match.
        with pytest.raises(ValueError):
            sut.analyze(root,pinned_report_sha256=digest)
    else:
        with pytest.raises(ValueError):
            sut.analyze(root,pinned_report_sha256=digest)


def test_original_file_mutation_fails_when_manifest_still_old(tmp_path):
    root,digest = original_format_synthetic(tmp_path)
    with (root/"report.json").open("a") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError,match="SHA differs"):
        sut.analyze(root,pinned_report_sha256=digest)


def test_cannot_accept_symlinked_original_study(tmp_path):
    root,digest = original_format_synthetic(tmp_path)
    alias = tmp_path/"alias"
    alias.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError,match="symlinked"):
        sut.analyze(alias,pinned_report_sha256=digest)
