from __future__ import annotations

import csv
import json
from pathlib import Path

from fpgai.paper.plots import generate_paper_plot_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _base_out(tmp_path: Path, name: str, mode: str) -> Path:
    out = tmp_path / name
    _write_json(
        out / "manifest.json",
        {
            "pipeline_mode": mode,
            "top_kernel_name": "deeplearn",
            "hls_ran": True,
            "hls_ok": True,
            "hls_returncode": 0,
            "build_stages": {"hls_synthesis": True, "vivado_implementation": True, "bitstream": True},
            "vivado_bridge": {
                "board": "kv260",
                "vivado_bridge_generated": True,
                "vivado_impl_requested": True,
                "bitstream_requested": True,
                "bitstream_exists": True,
                "xsa_exists": True,
                "ok": True,
                "failed_rows": [],
            },
        },
    )
    _write_json(out / "reports/model_profile.json", {"graph_name": name, "pipeline_mode": mode})
    _write_json(out / "reports/resource_prediction.json", {"totals": {"lut": 1200, "dsp": 8, "bram18": 4}})
    _write_json(out / "reports/timing_prediction.json", {"predicted_latency_ms": 0.2, "predicted_parallel_macs": 4})
    _write_json(out / "reports/board_fit.json", {"status": "fits", "vivado_implementation_allowed": True, "bitstream_allowed": True})
    _write_json(out / "reports/generated_cpp_validation.json", {"status": "passed"})
    _write_json(out / "reports/movement_contract_validation.json", {"status": "passed", "passed": True})
    _write_json(out / "reports/vivado_bd_validation.json", {"status": "passed"})
    _write_json(out / "reports/hls_synthesis_report.json", {"status": "passed", "lut": 1300, "dsp": 7, "bram18": 5, "latency_max": 1234})
    _write_json(out / "reports/vivado_implementation_report.json", {"status": "passed", "lut": 1500, "dsp": 7, "bram18": 5, "wns": 1.2, "power_w": 2.0})
    _write_json(out / "reports/hardware_knob_contract.json", {"knobs": [
        {"path": "optimization.parallel.pe", "source": "manual_yaml", "requested": 4, "effective": 4, "status": "applied", "applied_to": ["HLS unroll", "Vivado implementation"]},
        {"path": "precision.default", "source": "manual_yaml", "requested": "fx16", "effective": "fx16", "status": "applied", "applied_to": ["HLS types"]},
    ]})
    _write_json(out / "runtime_package/package_manifest.json", {"board": "kv260", "pipeline_mode": mode, "files": {}})
    return out


def test_paper_plot_generator_creates_inference_and_training_sections(tmp_path: Path) -> None:
    inference = _base_out(tmp_path, "inference_model", "inference")
    training = _base_out(tmp_path, "training_model", "training_on_device")
    _write_json(inference / "reports/runtime_results.json", {"status": "passed", "board": "kv260", "latency_ms_mean": 0.45, "throughput_fps": 2222, "accuracy": 0.98})
    (training / "reports/board_training_curve.csv").write_text(
        "step,epoch,batch,loss,accuracy,runtime_seconds,status\n"
        "0,0,0,1.0,0.4,0.01,ok\n"
        "1,0,1,0.7,0.6,0.02,ok\n",
        encoding="utf-8",
    )

    manifest = generate_paper_plot_artifacts([inference, training], output_dir=tmp_path / "paper_plots")

    assert manifest["sections"]["inference"]["design_count"] == 1
    assert manifest["sections"]["training"]["design_count"] == 1
    assert manifest["sections"]["training"]["real_training_curve_rows"] == 2
    assert "figure_03_inference_real_latency_ms" in manifest["created_figures"]
    assert "figure_08_training_curve_loss" in manifest["created_figures"]
    assert (tmp_path / "paper_plots/figures/figure_01_inference_hls_latency.svg").exists()
    assert (tmp_path / "paper_plots/figures/figure_08_training_curve_loss.svg").exists()

    with (tmp_path / "paper_plots/data/hardware_knob_settings.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert rows[0]["yaml_path"] == "optimization.parallel.pe"


def test_paper_plot_generator_marks_missing_real_runtime_as_pending(tmp_path: Path) -> None:
    inference = _base_out(tmp_path, "inference_model", "inference")
    training = _base_out(tmp_path, "training_model", "training_on_device")

    manifest = generate_paper_plot_artifacts([tmp_path], output_dir=tmp_path / "paper_plots")

    assert manifest["sections"]["inference"]["design_count"] == 1
    assert manifest["sections"]["inference"]["real_runtime_rows"] == 0
    assert manifest["sections"]["training"]["real_training_curve_rows"] == 0
    assert manifest["pending_figures"]["figure_03_inference_real_latency_ms"] == "pending real inference board-runtime measurements"
    assert manifest["pending_figures"]["figure_08_training_curve_loss"] == "pending real FPGA training-curve rows"
    assert (tmp_path / "paper_plots/paper_plot_manifest.md").exists()


def test_paper_plot_generator_extracts_metrics_from_real_report_files(tmp_path: Path) -> None:
    out = _base_out(tmp_path, "training_real_reports", "training_on_device")
    # Keep JSON reports as path-only/minimal, matching real compile outputs where numbers live in tool reports.
    _write_json(out / "reports/hls_synthesis_report.json", {"status": "passed", "report_path": "csynth.rpt"})
    _write_json(out / "reports/vivado_implementation_report.json", {"status": "passed"})

    hls_report = out / "hls/fpgai_hls_proj/sol1/syn/report/deeplearn_csynth.xml"
    hls_report.parent.mkdir(parents=True, exist_ok=True)
    hls_report.write_text(
        """
<profile>
  <PerformanceEstimates>
    <SummaryOfTimingAnalysis>
      <EstimatedClockPeriod>7.12</EstimatedClockPeriod>
    </SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency>
      <Best-caseLatency>100</Best-caseLatency>
      <Worst-caseLatency>250</Worst-caseLatency>
      <Interval-min>12</Interval-min>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources>
      <LUT>4239</LUT>
      <FF>5751</FF>
      <DSP>21</DSP>
      <BRAM_18K>19</BRAM_18K>
    </Resources>
  </AreaEstimates>
</profile>
""".strip(),
        encoding="utf-8",
    )
    reports_dir = out / "vivado_bridge/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "utilization_impl.rpt").write_text(
        "| CLB LUTs                   | 22397 |     0 |  0 |    117120 | 19.12 |\n"
        "| DSPs                       |    15 |     0 |  0 |      1248 |  1.20 |\n",
        encoding="utf-8",
    )
    (reports_dir / "timing_impl.rpt").write_text(
        "Setup :            0  Failing Endpoints,  Worst Slack   4.045ns,  Total Violation        0.000ns\n",
        encoding="utf-8",
    )
    (reports_dir / "power_impl.rpt").write_text(
        "| Total On-Chip Power (W)  | 2.904        |\n",
        encoding="utf-8",
    )

    manifest = generate_paper_plot_artifacts([out], output_dir=tmp_path / "paper_plots")

    assert "figure_05_training_hls_latency" in manifest["created_figures"]
    assert "figure_06_training_vivado_lut" in manifest["created_figures"]
    with (tmp_path / "paper_plots/data/paper_plot_rows.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["hls_latency_max"] == "250"
    assert rows[0]["hls_lut"] == "4239"
    assert rows[0]["vivado_lut"] == "22397"
    assert rows[0]["vivado_dsp"] == "15"
    assert rows[0]["vivado_wns"] == "4.045"
    assert rows[0]["vivado_power_w"] == "2.904"


def test_paper_plot_generator_writes_paper_tables_and_extra_resource_figures(tmp_path: Path) -> None:
    training = _base_out(tmp_path, "training_model", "training_on_device")
    reports_dir = training / "vivado_bridge/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "utilization_impl.rpt").write_text(
        "| CLB LUTs                   | 22397 |     0 |  0 |    117120 | 19.12 |\n"
        "| DSPs                       |    15 |     0 |  0 |      1248 |  1.20 |\n",
        encoding="utf-8",
    )
    (reports_dir / "timing_impl.rpt").write_text(
        "Setup :            0  Failing Endpoints,  Worst Slack   4.045ns,  Total Violation        0.000ns\n",
        encoding="utf-8",
    )
    (reports_dir / "power_impl.rpt").write_text("| Total On-Chip Power (W)  | 2.904        |\n", encoding="utf-8")

    manifest = generate_paper_plot_artifacts([training], output_dir=tmp_path / "paper_plots")

    assert "table_files" in manifest
    assert "table_01_experiment_overview" in manifest["table_files"]
    assert "table_03_hls_results" in manifest["table_files"]
    assert "table_04_vivado_results" in manifest["table_files"]
    assert (tmp_path / "paper_plots/tables/table_01_experiment_overview.md").exists()
    assert (tmp_path / "paper_plots/tables/table_04_vivado_results.csv").exists()
    assert "figure_10_training_vivado_power" in manifest["created_figures"]
    assert "figure_11_training_vivado_dsp" in manifest["created_figures"]
    assert "figure_13_all_vivado_power" in manifest["created_figures"]

    table_md = (tmp_path / "paper_plots/paper_plot_manifest.md").read_text(encoding="utf-8")
    assert "## Paper tables" in table_md
    assert "table_06_pending_measurements" in table_md


def test_paper_plot_generator_prefers_frozen_paper_experiments_and_skips_runtime_packages(tmp_path: Path) -> None:
    old = _base_out(tmp_path / "build" / "examples", "old_example", "inference")
    paper = _base_out(tmp_path / "build" / "paper_experiments" / "inference", "I0_baseline_fx16_embedded", "inference")
    runtime_pkg = paper / "runtime_package"
    _write_json(runtime_pkg / "manifest.json", {"pipeline_mode": "inference"})
    _write_json(runtime_pkg / "reports/vivado_implementation_report.json", {"status": "passed", "lut": 99})

    manifest = generate_paper_plot_artifacts([tmp_path / "build"], output_dir=tmp_path / "paper_plots")

    assert manifest["compile_output_count"] == 1
    with (tmp_path / "paper_plots/tables/table_01_experiment_overview.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["design_id"] for row in rows] == ["I0_baseline_fx16_embedded"]


def test_paper_table_uses_compact_precision_label(tmp_path: Path) -> None:
    out = _base_out(tmp_path / "paper_experiments" / "inference", "I1_precision_fx8_embedded", "inference")
    _write_json(out / "reports/hardware_knob_contract.json", {"knobs": [
        {
            "path": "precision.default",
            "source": "manual_yaml",
            "requested": "fx8",
            "effective": {
                "defaults": {
                    "activation": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
                    "weight": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
                    "accum": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                },
                "training": {},
            },
            "status": "applied",
            "applied_to": ["HLS types"],
        }
    ]})

    generate_paper_plot_artifacts([tmp_path / "paper_experiments"], output_dir=tmp_path / "paper_plots")
    text = (tmp_path / "paper_plots/tables/table_01_experiment_overview.md").read_text(encoding="utf-8")
    assert "fx8_3" in text
    assert "total_bits" not in text


def test_paper_plot_generator_writes_captions_summary_and_gallery(tmp_path: Path) -> None:
    inference = _base_out(tmp_path / "build" / "paper_experiments" / "inference", "I0_baseline_fx16_embedded", "inference")
    variant = _base_out(tmp_path / "build" / "paper_experiments" / "inference", "I1_precision_fx8_embedded", "inference")
    _write_json(variant / "reports/hls_synthesis_report.json", {"status": "passed", "lut": 800, "dsp": 4, "bram18": 2, "latency_max": 1000})
    _write_json(variant / "reports/vivado_implementation_report.json", {"status": "passed", "lut": 900, "dsp": 4, "bram18": 2, "wns": 1.1, "power_w": 1.8})

    manifest = generate_paper_plot_artifacts([tmp_path / "build"], output_dir=tmp_path / "paper_plots")

    assert "narrative_files" in manifest
    for rel in [
        "paper_results_summary.md",
        "figure_captions.md",
        "table_captions.md",
        "paper_claims_from_artifacts.md",
        "plot_gallery.md",
        "plot_gallery.html",
        "tables/table_08_result_comparisons.md",
        "tables/table_08_result_comparisons.csv",
    ]:
        assert (tmp_path / "paper_plots" / rel).exists()

    summary = (tmp_path / "paper_plots/paper_results_summary.md").read_text(encoding="utf-8")
    assert "Key computed comparisons" in summary
    assert "I1_precision_fx8_embedded" in summary
    manifest_md = (tmp_path / "paper_plots/paper_plot_manifest.md").read_text(encoding="utf-8")
    assert "## Narrative files" in manifest_md
    assert "table_08_result_comparisons" in manifest_md


def test_paper_plot_generator_writes_expanded_comparisons_classifications_and_subsection(tmp_path: Path) -> None:
    root = tmp_path / "build" / "paper_experiments"
    baseline = _base_out(root / "inference", "I0_baseline_fx16_embedded", "inference")
    fx24 = _base_out(root / "inference", "I2_precision_fx24_embedded", "inference")
    pe4 = _base_out(root / "inference", "I4_parallel_pe4", "inference")
    weight_import = _base_out(root / "inference", "I7_weight_import_m_axi", "inference")
    sgd = _base_out(root / "training", "T0_sgd_tiled_m_axi", "training_on_device")
    adam = _base_out(root / "training", "T2_adam_tiled_m_axi", "training_on_device")
    tile128 = _base_out(root / "training", "T5_tile128_m_axi", "training_on_device")
    bitstream = _base_out(root / "training", "T7_deployable_training_bitstream", "training_on_device")

    _write_json(fx24 / "reports/hls_synthesis_report.json", {"status": "passed", "lut": 1800, "dsp": 13, "bram18": 8, "latency_max": 1240})
    _write_json(fx24 / "reports/vivado_implementation_report.json", {"status": "passed", "lut": 2200, "dsp": 13, "bram18": 8, "wns": 0.9, "power_w": 2.2})
    _write_json(pe4 / "reports/hls_synthesis_report.json", {"status": "passed", "lut": 3000, "dsp": 40, "bram18": 12, "latency_max": 700})
    _write_json(pe4 / "reports/vivado_implementation_report.json", {"status": "passed", "lut": 3500, "dsp": 35, "bram18": 12, "wns": 0.7, "power_w": 2.5})
    _write_json(weight_import / "reports/hls_synthesis_report.json", {"status": "passed", "lut": 1600, "dsp": 7, "bram18": 6, "latency_max": 1000})
    _write_json(weight_import / "reports/vivado_implementation_report.json", {"status": "passed", "lut": 2100, "dsp": 7, "bram18": 7, "wns": 0.8, "power_w": 2.1})
    _write_json(adam / "reports/hls_synthesis_report.json", {"status": "passed", "lut": 1700, "dsp": 7, "bram18": 7, "latency_max": 1234})
    _write_json(adam / "reports/vivado_implementation_report.json", {"status": "passed", "lut": 1900, "dsp": 7, "bram18": 7, "wns": 1.0, "power_w": 2.1})
    _write_json(tile128 / "reports/hls_synthesis_report.json", {"status": "passed", "lut": 1400, "dsp": 7, "bram18": 5, "latency_max": 1500})
    _write_json(tile128 / "reports/vivado_implementation_report.json", {"status": "passed", "lut": 1600, "dsp": 7, "bram18": 5, "wns": 1.4, "power_w": 2.05})

    manifest = generate_paper_plot_artifacts([tmp_path / "build"], output_dir=tmp_path / "paper_plots")

    assert "paper_results_subsection_md" in manifest["narrative_files"]
    comparison_path = tmp_path / "paper_plots" / "tables" / "table_08_result_comparisons.csv"
    with comparison_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    comparisons = {row["comparison"] for row in rows}
    assert "precision_fx24_vs_fx16" in comparisons
    assert "parallel_pe4_vs_pe1" in comparisons
    assert "weight_import_m_axi_vs_embedded" in comparisons
    assert "training_adam_vs_sgd" in comparisons
    assert "training_tile128_vs_tile64" in comparisons
    assert "training_bitstream_vs_sgd" in comparisons
    assert {"result_classification", "artifact_status", "interpretation"}.issubset(rows[0].keys())
    assert any(row["result_classification"] == "expected_tradeoff" for row in rows)
    assert any(row["result_classification"] == "deployability_result" for row in rows)

    summary_text = (tmp_path / "paper_plots" / "tables" / "table_09_result_classification_summary.md").read_text(encoding="utf-8")
    assert "Result classification summary" in summary_text
    assert "expected_tradeoff" in summary_text
    subsection_text = (tmp_path / "paper_plots" / "paper_results_subsection.md").read_text(encoding="utf-8")
    assert "Paper-ready results subsection draft" in subsection_text
    assert "Claim boundary" in subsection_text


def test_paper_plot_generator_writes_numeric_validation_tables_and_figures(tmp_path: Path) -> None:
    root = tmp_path / "build" / "paper_experiments"
    fx16 = _base_out(root / "inference", "I0_baseline_fx16_embedded", "inference")
    fx8 = _base_out(root / "inference", "I1_precision_fx8_embedded", "inference")
    sgd = _base_out(root / "training", "T0_sgd_tiled_m_axi", "training_on_device")

    _write_json(
        fx16 / "reports" / "numeric_validation.json",
        {
            "status": "passed",
            "passed": True,
            "reason": "inference output comparison artifacts were compared successfully",
            "inference": {
                "status": "passed",
                "output_compare": {
                    "status": "compared",
                    "passed": True,
                    "num_compared": 10,
                    "mse": 1.0e-8,
                    "mae": 2.0e-4,
                    "max_abs_error": 8.0e-4,
                    "cosine_similarity": 0.9999,
                },
            },
            "paper_claim_allowed": {"numeric_correctness": True},
        },
    )
    _write_json(
        fx8 / "reports" / "numeric_validation.json",
        {
            "status": "passed",
            "passed": True,
            "reason": "inference output comparison artifacts were compared successfully",
            "inference": {
                "status": "passed",
                "output_compare": {
                    "status": "compared",
                    "passed": True,
                    "num_compared": 10,
                    "mse": 4.0e-6,
                    "mae": 1.2e-3,
                    "max_abs_error": 3.4e-3,
                    "cosine_similarity": 0.998,
                },
            },
            "paper_claim_allowed": {"numeric_correctness": True},
        },
    )
    _write_json(
        sgd / "reports" / "numeric_validation.json",
        {
            "status": "passed",
            "passed": True,
            "reason": "training reference and generated/testbench comparison artifacts are available",
            "training": {
                "status": "passed",
                "comparison": {
                    "status": "compared",
                    "grad_cosine": 0.997,
                    "weight_after_cosine": 0.999,
                    "weight_delta_cosine": 0.996,
                    "grad_mae": 0.001,
                    "grad_max_abs": 0.004,
                    "weight_after_mae": 0.0005,
                    "weight_after_max_abs": 0.002,
                },
            },
            "gradient_export": {"status": "compared"},
            "optimizer_state_validation": {"status": "not_requested"},
            "batch_accumulation": {"status": "not_requested"},
            "loss_validation": {"status": "compared", "initial_loss": 1.0, "final_loss": 0.8},
            "training_tiled_io": {"status": "compared"},
            "paper_claim_allowed": {"numeric_correctness": True},
        },
    )

    manifest = generate_paper_plot_artifacts([tmp_path / "build"], output_dir=tmp_path / "paper_plots")

    assert "figure_15_precision_numeric_error" in manifest["created_figures"]
    assert "figure_16_precision_resource_numeric_tradeoff" in manifest["created_figures"]
    assert "figure_17_training_gradient_fidelity" in manifest["created_figures"]
    for rel in [
        "data/numeric_validation_rows.csv",
        "tables/table_10_numeric_validation_summary.csv",
        "tables/table_11_inference_precision_numeric_tradeoff.md",
        "tables/table_12_training_numeric_validation.md",
        "tables/table_13_training_precision_numeric_tradeoff.md",
    ]:
        assert (tmp_path / "paper_plots" / rel).exists()

    with (tmp_path / "paper_plots" / "data" / "numeric_validation_rows.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_design = {row["design_id"]: row for row in rows}
    assert by_design["I1_precision_fx8_embedded"]["max_abs_error"] == "0.0034"
    assert by_design["I1_precision_fx8_embedded"]["numeric_quality"] == "passed"
    assert by_design["T0_sgd_tiled_m_axi"]["gradient_cosine"] == "0.997"
    assert by_design["T0_sgd_tiled_m_axi"]["loss_delta"] == "-0.19999999999999996"

    summary = (tmp_path / "paper_plots" / "paper_results_summary.md").read_text(encoding="utf-8")
    assert "Numeric validation summary" in summary
    subsection = (tmp_path / "paper_plots" / "paper_results_subsection.md").read_text(encoding="utf-8")
    assert "Numeric behavior validation" in subsection
