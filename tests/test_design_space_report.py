from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import fpgai.analysis.design_space_report as design_space_module
from fpgai.analysis.design_space_report import (
    run_design_space_report,
)
from fpgai.engine.models import LayerDescriptor


def _spec(
    total_bits: int,
    int_bits: int,
) -> dict[str, object]:
    return {
        "type": "ap_fixed",
        "total_bits": total_bits,
        "int_bits": int_bits,
    }


def _config() -> dict:
    return {
        "targets": {
            "platform": {
                "clocks": [
                    {
                        "name": "pl_clk0",
                        "target_mhz": 200,
                    }
                ]
            }
        },
        "numerics": {
            "defaults": {
                "activation": _spec(16, 6),
                "weight": _spec(16, 6),
                "bias": _spec(24, 10),
                "accum": _spec(24, 10),
            },
            "layers": [
                {
                    "match": {
                        "op_type": "Dense",
                    },
                    "weight": _spec(12, 4),
                }
            ],
        },
        "optimization": {
            "parallel": {
                "pe": 1,
                "simd": 1,
                "unroll_factor": 1,
                "partition_factor": 1,
                "pipeline_ii": 1,
            }
        },
        "memory": {
            "storage": {
                "weights": "bram",
                "activations": "bram",
            }
        },
        "analysis": {
            "precision_sweep": {
                "layer_overrides": "clear",
                "candidates": [
                    {
                        "name": "fx8",
                        "defaults": {
                            "activation": _spec(8, 3),
                            "weight": _spec(8, 3),
                            "bias": _spec(16, 6),
                            "accum": _spec(16, 6),
                        },
                    },
                    {
                        "name": "fx16",
                        "defaults": {
                            "activation": _spec(16, 6),
                            "weight": _spec(16, 6),
                            "bias": _spec(24, 10),
                            "accum": _spec(24, 10),
                        },
                    },
                ],
            },
            "design_space": {
                "recommendation": {
                    "require_prediction_match": True,
                    "min_cosine": 0.99,
                },
                "performance": {
                    "baseline_cpu_latency_ms": 1.6,
                },
                "estimator": {
                    "minimum_bram_elements": 512,
                },
            },
        },
    }


def _dense_descriptor() -> LayerDescriptor:
    return LayerDescriptor(
        node_name="dense0",
        op_type="Dense",
        inputs=[
            "input",
            "weights",
            "bias",
        ],
        outputs=["dense_output"],
        input_shapes=[
            (1, 32),
            (16, 32),
            (16,),
        ],
        output_shapes=[(1, 16)],
        param_names=[
            "weights",
            "bias",
        ],
        param_bytes=(
            16 * 32 * 4
            + 16 * 4
        ),
        activation_bytes_in=32 * 4,
        activation_bytes_out=16 * 4,
        macs=32 * 16,
        attrs={
            "in_features": 32,
            "out_features": 16,
        },
        compute_hint="compute_bound",
        backend_kernel="dense",
    )


def _relu_descriptor() -> LayerDescriptor:
    return LayerDescriptor(
        node_name="relu0",
        op_type="Relu",
        inputs=["dense_output"],
        outputs=["output"],
        input_shapes=[(1, 16)],
        output_shapes=[(1, 16)],
        param_names=[],
        param_bytes=0,
        activation_bytes_in=16 * 4,
        activation_bytes_out=16 * 4,
        macs=0,
        attrs={},
        compute_hint="memory_bound",
        backend_kernel="relu",
    )


def _sweep_rows() -> list[dict]:
    common = {
        "activation_int_bits": 3,
        "weight_int_bits": 3,
        "bias_int_bits": 6,
        "accum_int_bits": 6,
        "output_mae": 0.001,
        "output_max_abs": 0.003,
        "float_top1": 2,
        "quant_top1": 2,
        "prediction_match": True,
        "worst_layer_name": "dense0",
        "worst_layer_type": "Dense",
        "worst_layer_mse": 0.001,
        "quant_metrics_json": "metrics.json",
        "quant_summary_txt": "summary.txt",
        "quant_layerwise_csv": "layerwise.csv",
    }

    return [
        {
            **common,
            "name": "fx8",
            "activation_bits": 8,
            "weight_bits": 8,
            "bias_bits": 16,
            "accum_bits": 16,
            "output_mse": 0.002,
            "output_cosine": 0.995,
        },
        {
            **common,
            "name": "fx16",
            "activation_bits": 16,
            "activation_int_bits": 6,
            "weight_bits": 16,
            "weight_int_bits": 6,
            "bias_bits": 24,
            "bias_int_bits": 10,
            "accum_bits": 24,
            "accum_int_bits": 10,
            "output_mse": 0.0001,
            "output_cosine": 0.9999,
        },
    ]


def _install_test_doubles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_precision_sweep(
        *,
        model_path,
        raw_cfg,
        out_dir,
    ):
        del model_path
        del raw_cfg

        sweep_dir = Path(out_dir) / "precision_sweep"
        sweep_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_json = sweep_dir / "results.json"
        results_json.write_text(
            json.dumps(
                {
                    "results": _sweep_rows(),
                }
            ),
            encoding="utf-8",
        )

        return SimpleNamespace(
            results_json=results_json,
        )

    def fake_analyze_graph(graph):
        del graph
        return [
            _dense_descriptor(),
            _relu_descriptor(),
        ]

    monkeypatch.setattr(
        design_space_module,
        "run_precision_sweep",
        fake_precision_sweep,
    )
    monkeypatch.setattr(
        design_space_module,
        "analyze_graph",
        fake_analyze_graph,
    )


def test_design_space_report_writes_analytical_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_test_doubles(
        monkeypatch,
        tmp_path,
    )

    result = run_design_space_report(
        graph=object(),
        model_path=tmp_path / "model.onnx",
        raw_cfg=_config(),
        out_dir=tmp_path,
    )

    assert result.passed is True
    assert result.results_json.is_file()
    assert result.results_csv.is_file()
    assert result.summary_txt.is_file()

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8",
        )
    )

    assert payload["format"] == (
        "fpgai.design_space.v2"
    )
    assert payload["analytical_models"] == {
        "resources": "operator_structural_v2",
        "performance": (
            "operator_execution_schedule_v2"
        ),
    }

    assert len(payload["results"]) == 2
    assert len(payload["detailed_results"]) == 2

    for row in payload["results"]:
        assert row[
            "resource_estimation_mode"
        ] == "analytical"
        assert row[
            "performance_estimation_mode"
        ] == "analytical"
        assert row["predicted_lut"] > 0
        assert row["predicted_cycles"] > 0

    layer_csv = Path(
        payload["layer_breakdown_csv"]
    )

    assert layer_csv.is_file()

    with layer_csv.open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        layer_rows = list(
            csv.DictReader(input_file)
        )

    assert len(layer_rows) == 4
    assert {
        row["candidate"]
        for row in layer_rows
    } == {
        "fx8",
        "fx16",
    }
    assert {
        row["op_type"]
        for row in layer_rows
    } == {
        "Dense",
        "Relu",
    }

    assert all(
        float(row["predicted_cycles"]) > 0
        for row in layer_rows
    )


def test_design_space_candidates_are_isolated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_test_doubles(
        monkeypatch,
        tmp_path,
    )

    result = run_design_space_report(
        graph=object(),
        model_path=tmp_path / "model.onnx",
        raw_cfg=_config(),
        out_dir=tmp_path,
    )

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8",
        )
    )

    detailed = {
        row["name"]: row
        for row in payload["detailed_results"]
    }

    fx8_dense = detailed["fx8"][
        "resource_estimate"
    ]["layers"][0]
    fx16_dense = detailed["fx16"][
        "resource_estimate"
    ]["layers"][0]

    assert fx8_dense["activation_bits"] == 8
    assert fx8_dense["weight_bits"] == 8
    assert fx16_dense["activation_bits"] == 16
    assert fx16_dense["weight_bits"] == 16

    assert (
        fx16_dense["predicted_lut"]
        > fx8_dense["predicted_lut"]
    )
    assert (
        fx16_dense["predicted_ff"]
        > fx8_dense["predicted_ff"]
    )


def test_smallest_valid_candidate_is_selected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_test_doubles(
        monkeypatch,
        tmp_path,
    )

    result = run_design_space_report(
        graph=object(),
        model_path=tmp_path / "model.onnx",
        raw_cfg=_config(),
        out_dir=tmp_path,
    )

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8",
        )
    )

    assert payload[
        "recommended_smallest_valid"
    ]["name"] == "fx8"

    assert payload[
        "recommended_best_accuracy"
    ]["name"] == "fx16"

    assert "operator_structural_v2" in (
        result.terminal_summary
    )
    assert "operator_execution_schedule_v2" in (
        result.terminal_summary
    )


def test_design_space_payload_marks_recommendations_as_configured_candidate_only() -> None:
    from fpgai.analysis.design_space_report import _annotate_design_space_payload

    payload = {
        "format": "fpgai.design_space.v2",
        "recommendation_policy": {
            "mode": "balanced",
        },
        "results": [
            {
                "name": "fx8",
                "valid": True,
            },
            {
                "name": "fx16",
                "valid": True,
            },
        ],
        "recommended_smallest_valid": {
            "name": "fx8",
            "valid": True,
        },
        "recommended_balanced": {
            "name": "fx8",
            "valid": True,
        },
        "recommended_best_accuracy": {
            "name": "fx16",
            "valid": True,
        },
    }

    annotated = _annotate_design_space_payload(payload)

    assert annotated["recommendation_scope"] == "configured_candidates_only"
    assert annotated["search_enabled"] is False
    assert annotated["recommendation_kind"] == "estimate_based_recommendation"
    assert annotated["dse_truth"]["configured_candidates_only"] is True
    assert annotated["dse_truth"]["search_enabled"] is False
    assert annotated["dse_truth"]["estimate_based"] is True

    for row in annotated["results"]:
        assert row["compile_ready"] is True
        assert row["recommendation_scope"] == "configured_candidates_only"
        assert row["search_enabled"] is False
        assert row["materialization"]["unsupported_knobs"] == []
        assert "optimization.parallel.pe" in row["materialization"]["materialized_knobs"]
        assert "optimization.pipeline.ii" in row["materialization"]["materialized_knobs"]
        assert "optimization.tiling.dense" in row["materialization"]["materialized_knobs"]
        assert "data_movement.ps_pl.input" in row["materialization"]["materialized_knobs"]
        assert "resource_prediction" in row["materialization"]["estimate_only_outputs"]

    assert annotated["recommended_balanced"]["compile_ready"] is True
    assert annotated["recommended_balanced"]["search_enabled"] is False
    assert annotated["recommendation_policy"]["scope"] == "configured_candidates_only"
