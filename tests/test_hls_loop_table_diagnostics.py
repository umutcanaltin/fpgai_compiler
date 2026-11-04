"""Vitis 2023.2 loop-table diagnostics are shape-only and never attest II."""
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments.hls_loop_table_diagnostics import (
    _interval_kind, inspect_vitis_2023_loop_tables,
    classify_vitis_loop_tables, export_vitis_loop_table_diagnostics,
)
from fpgai_research.experiments.native_schedule_reports import (
    analyze_native_schedule_inventory, REPORT_REL,
)
from fpgai_research.toolchains.workflow import main

GROUP = "|     | Latency (cycles) |      | Initiation Interval |    |   |\n"
HEADER = "| Loop Name | min | max | Latency | achieved | target | Count | Pipelined |\n"
SEP = "+-------+------+-----+---------+----------+--------+-------+-----------+\n"
PRIVATE = "PRIVATE_LOOP_MATMUL_424242"
SAMPLE = (
    "Project root /home/PRIVATE_LAB_1199\n" + GROUP + HEADER + SEP +
    "| " + PRIVATE + " | 17 | 21 | 6 | - | - | 64 | yes |\n"
    "| + SECRET_CONV_OPERATOR | 4 | 11 | 3 | ? | ? | 2 | no |\n"
    "end unrelated material\n"
)


def test_vitis_ascii_divider_detection_is_not_a_regex_escape_regression():
    from fpgai_research.experiments.hls_loop_table_diagnostics import _is_ascii_divider
    assert _is_ascii_divider(SEP.rstrip("\n"))
    assert _is_ascii_divider("| ----- | ----- |")
    assert not _is_ascii_divider("   ")
    assert not _is_ascii_divider("| SECRET | 17 | 21 | yes |")


def test_recognize_paired_header_without_exporting_private_data():
    record = inspect_vitis_2023_loop_tables(SAMPLE.encode())
    assert record["paired_loop_table_count"] == 1
    assert record["candidate_rows_scanned"] == 2
    assert record["row_status_counts"]["ACHIEVED_UNAVAILABLE"] == 1
    assert record["row_status_counts"]["TARGET_UNAVAILABLE"] == 1
    assert record["row_status_counts"]["ACHIEVED_UNKNOWN"] == 1
    assert record["row_status_counts"]["SHAPE_VALID_ROWS"] == 2
    assert not record["per_loop_schedule_evidence_available"]
    serialized = json.dumps(record)
    for secret in ("PRIVATE_LOOP", "PRIVATE_LAB", "SECRET_CONV", "424242"):
        assert secret not in serialized


def test_numeric_cells_stay_unattributed_and_not_exported():
    report = (GROUP + HEADER + SEP +
              "| PRIVATE_MAC_LOOP | 10 | 20 | 3 | 2 | 1 | 32 | yes |\n")
    result = inspect_vitis_2023_loop_tables(report.encode())
    assert result["row_status_counts"]["NUMERIC_ACHIEVED_FIELD_UNATTRIBUTED"] == 1
    assert result["row_status_counts"]["NUMERIC_TARGET_FIELD_UNATTRIBUTED"] == 1
    assert "PRIVATE_MAC_LOOP" not in json.dumps(result)
    assert '"achieved_ii"' not in json.dumps(result)
    assert result["numeric_schedule_values_exported"] is False


@pytest.mark.parametrize("bad", ("0", "-2", "2.5", "1 ~ 8", "nan", "private"))
def test_ambiguous_ii_values_are_not_numbers(bad):
    assert _interval_kind(bad) == "INVALID_OR_AMBIGUOUS"


def test_malformed_row_not_accidentally_interpreted():
    report = GROUP + HEADER + SEP + "| SECRET | 1 | 2 | 3 | 0 | 1 | 4 | yes |\n"
    result = inspect_vitis_2023_loop_tables(report.encode())
    assert result["row_status_counts"]["MALFORMED_OR_AMBIGUOUS_ROW"] == 1
    assert "NUMERIC_ACHIEVED_FIELD_UNATTRIBUTED" not in result["row_status_counts"]


def test_unpaired_or_summary_headers_are_not_loop_schedule_evidence():
    orphan = HEADER + SEP + "| PRIVATE | 1 | 2 | 3 | 1 | 1 | 4 | yes |\n"
    result = inspect_vitis_2023_loop_tables(orphan.encode())
    assert result["paired_loop_table_count"] == 0
    assert result["row_status_counts"]["HEADER_WITHOUT_GROUPING"] == 1
    assert result["candidate_rows_scanned"] == 0
    assert inspect_vitis_2023_loop_tables(b"| min | max | latency | interval |\n")["paired_loop_table_count"] == 0


def test_bad_utf8_empty_and_bounds_rejected():
    with pytest.raises(ValueError):
        inspect_vitis_2023_loop_tables(b"")
    with pytest.raises(ValueError):
        inspect_vitis_2023_loop_tables(b"\xff\xfe")
    with pytest.raises(ValueError):
        inspect_vitis_2023_loop_tables(b"x" * (12 * 1024 * 1024 + 1))


def _suite(tmp_path):
    root = tmp_path / "suite"
    folder = root / "mnist_mlp" / "case_0000"
    folder.mkdir(parents=True)
    (root / "suite_results.json").write_text('{"schema":"fpgai-research/full-paper-suite/v1"}\n')
    src = folder / "model.cpp"
    src.write_text("const float SECRET_TENSOR_515151=0.125;\n")
    report = folder / REPORT_REL
    report.parent.mkdir(parents=True)
    report.write_text(SAMPLE)
    study = {
        "schema":"fpgai-research/scientific-ir-mechanism-study/v1",
        "source_suite_sha256":sha256((root / "suite_results.json").read_bytes()).hexdigest(),
        "case_count":1,
        "cases":[{
            "model":"mnist_mlp","case":"case_0000","axis":"baseline",
            "evidence":"E1-HLS-source",
            "source_sha256":sha256(src.read_bytes()).hexdigest(),
        }],
    }
    inventory = analyze_native_schedule_inventory(root, study)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    return root, inventory_path, inventory, report


def test_inventory_bound_classification_is_redacted_and_immutable(tmp_path, capsys):
    root, inventory_path, inventory, _ = _suite(tmp_path)
    destination = tmp_path / "diagnostics"
    result = export_vitis_loop_table_diagnostics(root, inventory_path, destination)
    assert result["classified_report_count"] == 1
    assert result["cases"][0]["provenance_status"] == "PRESENT_NOT_ATTESTED"
    assert result["cases"][0]["paired_loop_table_count"] == 1
    assert result["cases"][0]["candidate_rows_scanned"] == 2
    assert result["cases"][0]["row_status_counts"]["ACHIEVED_UNAVAILABLE"] == 1
    assert result["status_counts"]["SHAPE_VALID_ROWS"] == 2
    assert result["per_loop_schedule_evidence_available"] is False
    assert {p.name for p in destination.iterdir()} == {
        "loop_table_diagnostics.json", "loop_table_diagnostics.md"}
    published = "".join(p.read_text() for p in destination.iterdir())
    for secret in ("SECRET_TENSOR", "PRIVATE_LOOP", "PRIVATE_LAB", "424242", "515151"):
        assert secret not in published
    with pytest.raises(FileExistsError):
        export_vitis_loop_table_diagnostics(root, inventory_path, destination)
    assert main([
        "classify-hls-loop-tables", "--directory", str(root),
        "--inventory",str(inventory_path), "--output", str(tmp_path / "cli"),
        "--model","mnist_mlp","--case","case_0000",
    ]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["classified_report_count"] == 1
    assert not receipt["per_loop_schedule_evidence_available"]


def test_report_changes_and_symlinks_fail_closed(tmp_path):
    root, _, inventory, report = _suite(tmp_path)
    report.write_text(SAMPLE + "modified after inventory\n")
    with pytest.raises(ValueError, match="changed after inventory"):
        classify_vitis_loop_tables(root, inventory)
    root2, _, inventory2, report2 = _suite(tmp_path / "second")
    saved = report2.parent / "original.rpt"
    report2.rename(saved)
    report2.symlink_to(saved)
    with pytest.raises(ValueError, match="changed after inventory"):
        classify_vitis_loop_tables(root2, inventory2)
