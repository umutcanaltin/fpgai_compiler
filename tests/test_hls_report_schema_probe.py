"""Redacted HLS grammar probes never turn legacy text into schedule evidence."""
import json
from hashlib import sha256
from pathlib import Path
import pytest

from fpgai_research.experiments.native_schedule_reports import (
    analyze_native_schedule_inventory, REPORT_REL,
)
from fpgai_research.experiments.hls_report_schema_probe import (
    _examine, _role, _type, probe_hls_report_grammar,
    export_hls_report_grammar,
)
from fpgai_research.toolchains.workflow import main


def _prepared(tmp_path):
    root=tmp_path/"suite"
    root.mkdir(parents=True)
    (root/"suite_results.json").write_text('{"schema":"fpgai-research/full-paper-suite/v1"}\n')
    cases=[]
    for model,case in (
        ("mnist_mlp","case_0000"),("mnist_mlp","case_0006"),
        ("mnist_cnn","case_0000")):
        folder=root/model/case
        folder.mkdir(parents=True)
        src=folder/"model.cpp"
        src.write_text('const float PRIVATE_TENSOR_WEIGHT_675309 = 0.125f;\n')
        rpt=folder/REPORT_REL
        if (model,case)!=("mnist_cnn","case_0000"):
            rpt.parent.mkdir(parents=True)
            rpt.write_text(
                "Private project root: /home/research/private_lab_34567\n"
                "+----+----+----+----+----+----+\n"
                "| Loop Name | Latency (cycles) | Iteration Latency | Interval | Trip Count | Pipelined |\n"
                "| ---- | ---- | ---- | ---- | ---- | ---- |\n"
                "| private_Operator_ALPHA | 123456 | 5 | 1 | 20 | yes |\n"
                "| private_Operator_BETA | 234567 | ? | - | 40 | no |\n"
                "Some additional unrelated secret text dataset_998877\n")
        cases.append({
            "model":model,"case":case,
            "axis":"simd" if case=="case_0006" else "baseline",
            "evidence":"E1-HLS-source",
            "source_sha256":sha256(src.read_bytes()).hexdigest(),
        })
    manifest={
        "schema":"fpgai-research/scientific-ir-mechanism-study/v1",
        "source_suite_sha256":sha256((root/"suite_results.json").read_bytes()).hexdigest(),
        "case_count":len(cases),"cases":cases
    }
    inv=analyze_native_schedule_inventory(root,manifest)
    path=tmp_path/"inventory.json"
    path.write_text(json.dumps(inv))
    return root,path,inv


def test_safe_header_and_cell_type_classifiers_are_closed_vocabulary():
    assert _role("Iteration Latency")=="ITERATION_LATENCY_HEADER"
    assert _role("Loop Name")=="LOOP_HEADER"
    assert _role("Initiation Interval")=="INTERVAL_HEADER"
    assert _role("II")=="II_HEADER"
    assert _role("ii")=="II_HEADER"
    assert _role("private_Operator_ALPHA")=="OTHER_HEADER"
    assert _type("3.5")=="NUMBER"
    assert _type("1 ~ 24")=="NUMBER_RANGE"
    assert _type("?")=="UNKNOWN"
    assert _type("-")=="UNAVAILABLE"
    assert _type("----")=="SEPARATOR"
    assert _type("private_operator_XYZ")=="TEXT"
    assert _type("yes")=="BOOLEAN"


def test_redacted_four_column_candidate_has_ii_role_without_values():
    result=_examine((
        "| Loop | Iteration Latency | II | Pipelined |\n"
        "| PRIVATE_HLS_LOOP_426 | 3 | 1 | yes |\n"
    ).encode())
    assert result["table_probe_count"]==1
    assert result["table_grammars"][0]["sanitized_header_roles"]==[
        "LOOP_HEADER","ITERATION_LATENCY_HEADER","II_HEADER","PIPELINED_HEADER"]
    assert "PRIVATE_HLS_LOOP" not in json.dumps(result)
    assert result["numeric_schedule_values_exported"] is False


def test_probed_schemas_contain_no_raw_values_operator_names_or_paths(tmp_path):
    root,_,inventory=_prepared(tmp_path)
    result=probe_hls_report_grammar(root,inventory)
    assert result["case_count"]==3
    assert result["probed_report_count"]==2
    assert not result["native_tools_executed"]
    assert not result["existing_evidence_upgraded"]
    assert not result["per_loop_schedule_evidence_available"]
    mlp=result["cases"][0]
    assert mlp["provenance_status"]=="PRESENT_NOT_ATTESTED"
    assert not mlp["per_loop_schedule_attested"]
    assert mlp["grammars_probed"] is True
    assert mlp["table_probe_count"]>=1
    shapes=mlp["table_grammars"]
    assert any("ITERATION_LATENCY_HEADER" in x["sanitized_header_roles"]
               for x in shapes)
    one=next(x for x in shapes
             if "ITERATION_LATENCY_HEADER" in x["sanitized_header_roles"])
    assert one["sample_row_grammars"][0]["cell_types"]==[
        "TEXT","NUMBER","NUMBER","NUMBER","NUMBER","BOOLEAN"]
    assert one["sample_row_grammars"][1]["cell_types"]==[
        "TEXT","NUMBER","UNKNOWN","UNAVAILABLE","NUMBER","BOOLEAN"]
    assert result["cases"][2]["grammars_probed"] is False
    serialized=json.dumps(result)
    for secret in ("PRIVATE_TENSOR_WEIGHT_675309",
                   "private_Operator_ALPHA","private_Operator_BETA",
                   "private_lab_34567","dataset_998877"):
        assert secret not in serialized
    assert '"operator_names_exported": false' in serialized.lower() or (
        all(item.get("operator_names_exported") is False
            for item in result["cases"] if item["grammars_probed"]))


def test_selection_and_cli_make_immutable_sanitized_output(tmp_path,capsys):
    root,path,inventory=_prepared(tmp_path)
    dest=tmp_path/"redacted"
    old_digest=sha256((root/"mnist_mlp"/"case_0006"/REPORT_REL).read_bytes()).hexdigest()
    report=export_hls_report_grammar(
        root,path,dest,selected_models=("mnist_mlp",),
        selected_cases=("case_0006",))
    assert report["case_count"]==1
    assert report["probed_report_count"]==1
    assert {x.name for x in dest.iterdir()}=={
        "hls_table_grammar.json","hls_table_grammar.md"}
    assert sha256((root/"mnist_mlp"/"case_0006"/REPORT_REL).read_bytes()).hexdigest()==old_digest
    view=(dest/"hls_table_grammar.md").read_text()
    assert "redacted row" in view
    assert "No report was re-attested" in view
    assert "private_Operator_ALPHA" not in view
    assert "123456" not in view
    with pytest.raises(FileExistsError):
        export_hls_report_grammar(root,path,dest)
    assert main([
        "probe-hls-table-grammar",
        "--directory",str(root),
        "--inventory",str(path),
        "--output",str(tmp_path/"cli"),
        "--model","mnist_mlp","--case","case_0000",
    ])==0
    status=json.loads(capsys.readouterr().out)
    assert status["case_count"]==1
    assert status["probed_report_count"]==1
    assert not status["per_loop_schedule_evidence_available"]


def test_rejects_stale_report_after_inventory(tmp_path):
    root,_,inventory=_prepared(tmp_path)
    report=root/"mnist_mlp"/"case_0000"/REPORT_REL
    report.write_text(report.read_text()+"changed after collection\n")
    with pytest.raises(ValueError,match="changed after inventory"):
        probe_hls_report_grammar(root,inventory)


def test_rejects_modified_source_and_pinned_suite(tmp_path):
    root,_,inventory=_prepared(tmp_path)
    source=root/"mnist_mlp"/"case_0000"/"model.cpp"
    source.write_text("replaced model\n")
    with pytest.raises(ValueError,match="source hash"):
        probe_hls_report_grammar(root,inventory)
    root2,_,inventory2=_prepared(tmp_path/"second")
    (root2/"suite_results.json").write_text("{}")
    with pytest.raises(ValueError,match="pinned suite"):
        probe_hls_report_grammar(root2,inventory2)


@pytest.mark.parametrize("models,cases",[
    (("../outside",),()), ((),("../outside",)),
    (("unknown_model",),()), ((),("case_9988",)),
])
def test_reject_invalid_filters(tmp_path,models,cases):
    root,_,inventory=_prepared(tmp_path)
    with pytest.raises(ValueError):
        probe_hls_report_grammar(root,inventory,
                                 selected_models=models,selected_cases=cases)


def test_strict_report_bounds_and_missing_cases(tmp_path):
    with pytest.raises(ValueError,match="empty or unbounded"):
        _examine(b"")
    root,_,inventory=_prepared(tmp_path)
    target=root/"mnist_mlp"/"case_0000"/REPORT_REL
    old=target.parent/"safe_original.rpt"
    target.rename(old)
    target.symlink_to(old)
    with pytest.raises(ValueError,match="changed after inventory"):
        probe_hls_report_grammar(root,inventory)


def test_does_not_infer_achieved_ii_from_numeric_table(tmp_path):
    root,_,inventory=_prepared(tmp_path)
    result=probe_hls_report_grammar(root,inventory)
    for row in result["cases"]:
        assert row["per_loop_schedule_attested"] is False
        if row["grammars_probed"]:
            assert row["numeric_schedule_values_exported"] is False
            assert row["operator_names_exported"] is False
            assert row["source_text_exported"] is False
            assert "achieved_ii" not in json.dumps(row).lower()
