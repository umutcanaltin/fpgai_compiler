from __future__ import annotations

import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    assert path.exists(), f"missing expected paper result artifact: {path}"
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_paper_artifact_index_coverage_if_generated() -> None:
    p = Path("paper_results/index/paper_artifact_index.csv")
    if not p.exists():
        return

    rows = _read_rows(p)
    assert len(rows) == 20

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["paper_status"]] = status_counts.get(row["paper_status"], 0) + 1

    assert status_counts.get("vivado_impl_bitstream_ready") == 4
    assert status_counts.get("vivado_board_capacity_rejected") == 1
    assert status_counts.get("hls_only") == 15


def test_paper_numeric_joined_coverage_if_generated() -> None:
    p = Path("paper_results/parsed/paper_numeric_joined.csv")
    if not p.exists():
        return

    rows = _read_rows(p)
    assert len(rows) == 20

    assert sum(bool(r.get("prediction_lut")) for r in rows) == 20
    assert sum(bool(r.get("prediction_dsp")) for r in rows) == 20
    assert sum(bool(r.get("prediction_bram18")) for r in rows) == 20

    assert sum(bool(r.get("hls_lut")) for r in rows) == 20
    assert sum(bool(r.get("hls_latency_worst_cycles")) for r in rows) == 20

    assert sum(bool(r.get("vivado_lut")) for r in rows) == 4
    assert sum(bool(r.get("vivado_total_on_chip_power_w")) for r in rows) == 4

    failures = [r for r in rows if r.get("vivado_failure_class")]
    assert len(failures) == 1
    failure = failures[0]
    assert failure["design"] == "training_kv260_safe_fx16_6"
    assert failure["vivado_failure_class"] == "vivado_impl_failed_board_capacity_lut_overutilized"
    assert failure["failure_slice_lut_required"] == "133729"
    assert failure["failure_slice_lut_available"] == "117120"
    assert failure["failure_slice_lut_util_pct"] == "114.18"


def test_paper_tables_coverage_if_generated() -> None:
    base = Path("paper_results/tables")
    if not base.exists():
        return

    expected = {
        "table_1_artifact_coverage.csv": 8,
        "table_2_prediction_vs_hls.csv": 20,
        "table_3_hls_vs_vivado.csv": 5,
        "table_4_knob_effects.csv": 16,
        "table_5_training_capacity.csv": 2,
    }

    for name, expected_rows in expected.items():
        rows = _read_rows(base / name)
        assert len(rows) == expected_rows

    training_rows = _read_rows(base / "table_5_training_capacity.csv")
    safe = next(r for r in training_rows if r["design"] == "training_kv260_safe_fx16_6")
    assert safe["vivado_status"] == "vivado_board_capacity_rejected"
    assert safe["failure_class"] == "vivado_impl_failed_board_capacity_lut_overutilized"
    assert safe["required_slice_luts"] == "133729"
    assert safe["available_slice_luts"] == "117120"
    assert safe["slice_lut_util_pct"] == "114.18"


def test_paper_figures_if_generated() -> None:
    base = Path("paper_results/figures")
    if not base.exists():
        return

    expected = [
        "artifact_coverage.pdf",
        "hls_vs_vivado_lut.pdf",
        "knob_latency_effect.pdf",
        "prediction_vs_hls_bram18.pdf",
        "prediction_vs_hls_dsp.pdf",
        "prediction_vs_hls_lut.pdf",
        "training_capacity.pdf",
    ]

    for name in expected:
        path = base / name
        assert path.exists(), f"missing generated figure: {path}"
        assert path.stat().st_size > 1000, f"generated figure is unexpectedly small: {path}"


def test_arxiv_tables_if_generated() -> None:
    base = Path("paper_results/tables")
    if not base.exists():
        return

    expected = [
        "table_arxiv_artifact_coverage.tex",
        "table_arxiv_prediction_vs_hls.tex",
        "table_arxiv_vivado_subset.tex",
        "table_arxiv_knob_summary.tex",
        "table_arxiv_training_capacity.tex",
    ]

    for name in expected:
        path = base / name
        assert path.exists(), f"missing arXiv table: {path}"
        text = path.read_text()
        assert "\\begin{table}" in text
        assert "\\end{table}" in text
        assert "textbackslash" not in text

    training = (base / "table_arxiv_training_capacity.tex").read_text()
    assert "cap reject" in training
    assert "114.18\\%" in training
