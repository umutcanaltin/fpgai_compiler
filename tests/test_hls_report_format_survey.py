"""Exploratory Vitis report layout cannot become HLS scheduling evidence."""
import json
from hashlib import sha256

import pytest

from fpgai_research.experiments.hls_report_format_survey import (
    _survey_text, survey_schedule_report_formats, export_schedule_report_formats,
)
from fpgai_research.experiments.native_schedule_reports import (
    analyze_native_schedule_inventory,
)
from fpgai_research.toolchains.workflow import main


def _fixture(tmp_path):
    root=tmp_path/"source_suite"
    root.mkdir(parents=True)
    (root/"suite_results.json").write_text('{"schema":"fpgai-research/full-paper-suite/v1"}\n')
    cases=[]
    original=[]
    for number in range(2):
        case=f"case_{number:04d}"
        directory=root/"mnist_mlp"/case
        directory.mkdir(parents=True)
        source=directory/"model.cpp"
        source.write_text("float private_weight_123456789=0.1;\n")
        original.append(source.read_bytes())
        if number==0:
            report=directory/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.rpt"
            report.parent.mkdir(parents=True)
            report.write_text(
                "Vitis HLS v2023.2\n"
                "Loop Summary\n"
                "| Loop | Latency | Interval | Pipeline | Trip Count |\n"
                "| secret_operator_77777 | 128 | 2 | yes | 512 |\n"
                "| Loop Name | Iteration Latency | II | other |\n"
                "source:/sensitive/folder/secret_run_8675309\n"
            )
        cases.append({
            "model":"mnist_mlp","case":case,"axis":"baseline" if number==0 else "simd",
            "evidence":"E1-HLS-source",
            "source_sha256":sha256(source.read_bytes()).hexdigest(),
        })
    study={
        "schema":"fpgai-research/scientific-ir-mechanism-study/v1",
        "source_suite_sha256":sha256((root/"suite_results.json").read_bytes()).hexdigest(),
        "case_count":2,"cases":cases
    }
    inventory=analyze_native_schedule_inventory(root,study)
    assert inventory["status_counts"]=={
        "PRESENT_NOT_ATTESTED":1,"REPORT_NOT_PRESENT":1}
    file=tmp_path/"native_schedule_inventory.json"
    file.write_text(json.dumps(inventory))
    return root,file,inventory,original


def test_legacy_survey_recognizes_headers_without_leaking_source_or_numbers(tmp_path):
    root,_,inventory,original=_fixture(tmp_path)
    report=survey_schedule_report_formats(root,inventory)
    assert report["case_count"]==2
    assert report["surveyed_report_count"]==1
    assert report["distinct_format_count"]==1
    assert not report["native_tools_executed"]
    assert not report["per_loop_schedule_evidence_available"]
    assert not report["native_qor_or_evidence_upgraded"]
    entry=report["cases"][0]
    assert entry["provenance_status"]=="PRESENT_NOT_ATTESTED"
    assert entry["surveyed"] is True
    assert entry["possible_loop_heading_lines"]==1
    assert entry["possible_schedule_table_headers"]==2
    assert entry["possible_loop_or_iteration_table_headers"]==2
    assert len(entry["candidate_table_row_profiles"])==2
    assert all(p["candidate_role"]=="POSSIBLY_LOOP_OR_ITERATION_TABLE"
               for p in entry["candidate_table_row_profiles"])
    assert entry["recognized_line_counts"]["loop"]>=2
    assert entry["per_loop_ii_cycles"] is None
    assert entry["claims_about_achieved_loop_ii"] is False
    assert report["cases"][1]["surveyed"] is False
    rendered=json.dumps(report,sort_keys=True)
    for private in ("private_weight_123456789","secret_operator_77777",
                    "secret_run_8675309","/sensitive/folder"):
        assert private not in rendered
    for number in range(2):
        assert (root/"mnist_mlp"/f"case_{number:04d}"/"model.cpp").read_bytes()==original[number]


def test_layout_survey_maintains_column_vocabulary_only():
    raw=(b"| Loop Name | latency (cycles) | Interval (cycles) | II | z_987654 |\n"
         b"other private intellectual property 12345\n")
    result=_survey_text(raw)
    assert result["possible_schedule_table_headers"]==1
    assert len(result["possible_column_layouts"])==1
    assert result["possible_column_layouts"][0]["columns"]==[
        "loop","latency","interval","ii","other"]
    for token in result["possible_column_layouts"][0]["columns"]:
        assert token in ("loop","latency","interval","ii","other")
    assert "12345" not in json.dumps(result)


def test_survey_profiles_adjacent_rows_without_exposing_achieved_ii():
    raw=(b"| Loop | Latency | Interval | Pipeline | Trip Count |\n"
         b"| ---- | ------- | -------- | -------- | ---------- |\n"
         b"| secret_operator_A77 | 128 | 2 | yes | 512 |\n"
         b"| secret_operator_B99 | 256 | 4 | yes | 1024 |\n"
         b"| Loop Name | Iteration Latency | II | other |\n"
         b"| secret_C | 33 | 7 | no |\n")
    result=_survey_text(raw)
    assert result["possible_schedule_table_headers"]==2
    assert result["possible_loop_or_iteration_table_headers"]==2
    profiles=result["candidate_table_row_profiles"]
    first=next(p for p in profiles if len(p["columns"])==5)
    assert first["candidate_role"]=="POSSIBLY_LOOP_OR_ITERATION_TABLE"
    assert first["possible_numeric_data_rows"]==2
    assert first["separator_rows"]==1
    assert first["truncated_tables"]==0
    assert result["extracted_numerical_schedule_values"] is False
    assert result["claims_about_achieved_loop_ii"] is False
    emitted=json.dumps(result)
    for secret in ("secret_operator_A77","secret_operator_B99",
                   "secret_C"):
        assert secret not in emitted


def test_within_model_layout_contrasts_are_not_scheduling_conclusions(tmp_path):
    root,_,inventory,_=_fixture(tmp_path)
    preliminary=survey_schedule_report_formats(root,inventory)
    contrast=preliminary["within_model_layout_contrasts"]
    assert len(contrast)==1
    assert contrast[0]["format_comparison_status"]=="REPORT_LAYOUT_UNAVAILABLE"
    assert contrast[0]["matching_format_fingerprint"] is None
    assert not contrast[0]["scheduling_equivalence_or_cause_proven"]

    source=root/"mnist_mlp"/"case_0000"/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.rpt"
    second=root/"mnist_mlp"/"case_0001"/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.rpt"
    second.parent.mkdir(parents=True)
    second.write_bytes(source.read_bytes())
    def inventory_now():
        study={
            "schema":"fpgai-research/scientific-ir-mechanism-study/v1",
            "source_suite_sha256":inventory["source_suite_sha256"],
            "case_count":2,
            "cases":[{
                "model":c["model"],"case":c["case"],"axis":c["axis"],
                "evidence":c["case_evidence"],"source_sha256":c["source_sha256"]
            } for c in inventory["cases"]],
        }
        return analyze_native_schedule_inventory(root,study)
    same=survey_schedule_report_formats(root,inventory_now())
    row=same["within_model_layout_contrasts"][0]
    assert row["format_comparison_status"]=="SAME_SURVEY_FORMAT_FINGERPRINT"
    assert row["matching_format_fingerprint"] is True
    assert not row["scheduling_equivalence_or_cause_proven"]
    second.write_text(second.read_text()+"| Loop | Latency | II | Pipeline |\n")
    diff=survey_schedule_report_formats(root,inventory_now())
    row=diff["within_model_layout_contrasts"][0]
    assert row["format_comparison_status"]=="DIFFERENT_SURVEY_FORMAT_FINGERPRINT"
    assert row["matching_format_fingerprint"] is False
    assert not row["scheduling_equivalence_or_cause_proven"]


def test_read_only_export_and_cli_leave_original_reports_unchanged(tmp_path,capsys):
    root,inventory_path,inventory,_=_fixture(tmp_path)
    report_file=root/"mnist_mlp"/"case_0000"/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.rpt"
    before=sha256(report_file.read_bytes()).hexdigest()
    destination=tmp_path/"survey"
    result=export_schedule_report_formats(root,inventory_path,destination)
    assert result["surveyed_report_count"]==1
    assert {x.name for x in destination.iterdir()}=={
        "hls_report_format_survey.json","hls_report_format_survey.md"}
    assert sha256(report_file.read_bytes()).hexdigest()==before
    assert "No old report was re-attested" in (
        destination/"hls_report_format_survey.md").read_text()
    assert "secret_operator_77777" not in (
        destination/"hls_report_format_survey.md").read_text()
    with pytest.raises(FileExistsError):
        export_schedule_report_formats(root,inventory_path,destination)
    cli_result=main([
        "survey-hls-reports","--directory",str(root),
        "--inventory",str(inventory_path),
        "--output",str(tmp_path/"survey_cli")
    ])
    assert cli_result==0
    parsed=json.loads(capsys.readouterr().out)
    assert parsed["surveyed_report_count"]==1
    assert parsed["per_loop_schedule_evidence_available"] is False
    assert parsed["status_counts"]==inventory["status_counts"]


def test_rejects_stale_report_after_inventory_created(tmp_path):
    root,_,inventory,_=_fixture(tmp_path)
    report=root/"mnist_mlp"/"case_0000"/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.rpt"
    report.write_text(report.read_text()+"tampered\n")
    with pytest.raises(ValueError,match="changed after inventory"):
        survey_schedule_report_formats(root,inventory)


def test_rejects_changed_source_or_suite_manifest(tmp_path):
    root,_,inventory,_=_fixture(tmp_path)
    (root/"mnist_mlp"/"case_0000"/"model.cpp").write_text("altered\n")
    with pytest.raises(ValueError,match="source hash"):
        survey_schedule_report_formats(root,inventory)
    root,_,inventory,_=_fixture(tmp_path/"another")
    (root/"suite_results.json").write_text("{}\n")
    with pytest.raises(ValueError,match="pinned suite"):
        survey_schedule_report_formats(root,inventory)


@pytest.mark.parametrize("bad",["duplicate","path_traversal","wrong_schema","removed_case"])
def test_rejects_bad_inventory(tmp_path,bad):
    root,_,inventory,_=_fixture(tmp_path)
    changed=json.loads(json.dumps(inventory))
    if bad=="duplicate":
        changed["cases"][1]=dict(changed["cases"][0])
    elif bad=="path_traversal":
        changed["cases"][0]["model"]="../outside"
    elif bad=="wrong_schema":
        changed["schema"]="wrong"
    elif bad=="removed_case":
        changed["cases"].pop()
    with pytest.raises(ValueError):
        survey_schedule_report_formats(root,changed)


def test_rejects_symlink_to_report(tmp_path):
    root,_,inventory,_=_fixture(tmp_path)
    path=root/"mnist_mlp"/"case_0000"/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.rpt"
    original=path.parent/"original.rpt"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ValueError,match="changed after inventory"):
        survey_schedule_report_formats(root,inventory)


def test_rejects_oversized_inventory_and_report_text(tmp_path):
    root,inventory_path,inventory,_=_fixture(tmp_path)
    with pytest.raises(ValueError,match="outside safe survey size"):
        _survey_text(b"")
    inventory_path.write_bytes(b"x"*(16*1024*1024+1))
    with pytest.raises(ValueError,match="size exceeds"):
        export_schedule_report_formats(root,inventory_path,tmp_path/"bad")
