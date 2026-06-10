from __future__ import annotations

import csv
import json
from pathlib import Path

from fpgai.analysis.hls_layer_validation import (
    build_hls_layer_validation,
    run_hls_layer_validation,
)


def _resource_estimate() -> dict:
    return {
        "analytical_model": (
            "operator_structural_v2"
        ),
        "top_level": {
            "predicted_lut": 500,
            "predicted_ff": 600,
            "predicted_dsp": 0,
            "predicted_bram18": 2,
        },
        "layers": [
            {
                "layer_index": 0,
                "layer_name": "conv0",
                "op_type": "Conv",
                "predicted_lut": 1000,
                "predicted_ff": 1200,
                "predicted_dsp": 2,
                "predicted_bram18": 4,
            },
            {
                "layer_index": 1,
                "layer_name": "relu0",
                "op_type": "Relu",
                "predicted_lut": 200,
                "predicted_ff": 100,
                "predicted_dsp": 0,
                "predicted_bram18": 0,
            },
            {
                "layer_index": 2,
                "layer_name": "dense0",
                "op_type": "Dense",
                "predicted_lut": 400,
                "predicted_ff": 600,
                "predicted_dsp": 2,
                "predicted_bram18": 3,
            },
            {
                "layer_index": 3,
                "layer_name": "softmax0",
                "op_type": "Softmax",
                "predicted_lut": 500,
                "predicted_ff": 600,
                "predicted_dsp": 2,
                "predicted_bram18": 0,
            },
        ],
    }


def _performance_estimate() -> dict:
    return {
        "analytical_performance_model": (
            "operator_execution_schedule_v2"
        ),
        "layer_cycles": [
            {
                "layer_index": 0,
                "layer_name": "conv0",
                "op_type": "Conv",
                "predicted_cycles": 12000,
            },
            {
                "layer_index": 1,
                "layer_name": "relu0",
                "op_type": "Relu",
                "predicted_cycles": 2704,
            },
            {
                "layer_index": 2,
                "layer_name": "dense0",
                "op_type": "Dense",
                "predicted_cycles": 6768,
            },
            {
                "layer_index": 3,
                "layer_name": "softmax0",
                "op_type": "Softmax",
                "predicted_cycles": 66,
            },
        ],
    }


def _module(
    *,
    name: str,
    op_type: str,
    lut: int,
    ff: int,
    dsp: int,
    bram18: int,
    latency_cycles: int,
) -> dict:
    return {
        "module": name,
        "op_type": op_type,
        "hierarchy_role": "primary",
        "is_top": False,
        "is_generated_helper": False,
        "lut": lut,
        "ff": ff,
        "dsp": dsp,
        "bram18": bram18,
        "latency_cycles": latency_cycles,
        "report_path": (
            f"/reports/{name}_csynth.xml"
        ),
    }


def _module_breakdown() -> dict:
    return {
        "format": (
            "fpgai.hls_module_breakdown.v2"
        ),
        "available": True,
        "primary_modules": [
            _module(
                name="conv2d",
                op_type="Conv",
                lut=10385,
                ff=14264,
                dsp=154,
                bram18=0,
                latency_cycles=3045,
            ),
            _module(
                name="relu_typed",
                op_type="Relu",
                lut=221,
                ff=54,
                dsp=0,
                bram18=0,
                latency_cycles=2706,
            ),
            _module(
                name="dense_out_in",
                op_type="Dense",
                lut=416,
                ff=624,
                dsp=2,
                bram18=3,
                latency_cycles=6768,
            ),
            _module(
                name="softmax_typed",
                op_type="Softmax",
                lut=2732,
                ff=3046,
                dsp=2,
                bram18=0,
                latency_cycles=81,
            ),
        ],
        "unassigned_top_resources": {
            "lut": 4691,
            "ff": 834,
            "dsp": 7,
            "bram18": 27,
        },
    }


def test_matches_layers_to_primary_modules() -> None:
    result = build_hls_layer_validation(
        resource_estimate=(
            _resource_estimate()
        ),
        performance_estimate=(
            _performance_estimate()
        ),
        module_breakdown=(
            _module_breakdown()
        ),
    )

    assert result["available"] is True
    assert result["format"] == (
        "fpgai.hls_layer_validation.v1"
    )
    assert result["resource_model"] == (
        "operator_structural_v2"
    )
    assert result["performance_model"] == (
        "operator_execution_schedule_v2"
    )

    assert len(
        result["matched_layers"]
    ) == 4

    rows = {
        row["layer_name"]: row
        for row in result[
            "matched_layers"
        ]
    }

    assert rows["conv0"]["module"] == (
        "conv2d"
    )
    assert rows["relu0"]["module"] == (
        "relu_typed"
    )
    assert rows["dense0"]["module"] == (
        "dense_out_in"
    )
    assert rows["softmax0"]["module"] == (
        "softmax_typed"
    )


def test_conv_model_error_is_reported() -> None:
    result = build_hls_layer_validation(
        resource_estimate=(
            _resource_estimate()
        ),
        performance_estimate=(
            _performance_estimate()
        ),
        module_breakdown=(
            _module_breakdown()
        ),
    )

    conv = result[
        "matched_layers"
    ][0]
    comparisons = conv["comparisons"]

    assert comparisons["lut"][
        "direction"
    ] == "underestimated"

    assert comparisons["dsp"][
        "direction"
    ] == "underestimated"

    assert comparisons[
        "latency_cycles"
    ]["direction"] == "overestimated"

    assert comparisons["lut"][
        "quality"
    ] == "poor"

    assert comparisons["dsp"][
        "quality"
    ] == "poor"

    assert result[
        "requires_model_revision"
    ] is True


def test_accurate_dense_fields_are_detected() -> None:
    result = build_hls_layer_validation(
        resource_estimate=(
            _resource_estimate()
        ),
        performance_estimate=(
            _performance_estimate()
        ),
        module_breakdown=(
            _module_breakdown()
        ),
    )

    dense = next(
        row
        for row in result[
            "matched_layers"
        ]
        if row["layer_name"] == "dense0"
    )
    comparisons = dense["comparisons"]

    assert comparisons["dsp"][
        "absolute_percentage_error"
    ] == 0.0
    assert comparisons["dsp"][
        "quality"
    ] == "excellent"

    assert comparisons["bram18"][
        "absolute_percentage_error"
    ] == 0.0
    assert comparisons["bram18"][
        "quality"
    ] == "excellent"

    assert comparisons[
        "latency_cycles"
    ][
        "absolute_percentage_error"
    ] == 0.0


def test_top_level_resources_are_compared() -> None:
    result = build_hls_layer_validation(
        resource_estimate=(
            _resource_estimate()
        ),
        performance_estimate=(
            _performance_estimate()
        ),
        module_breakdown=(
            _module_breakdown()
        ),
    )

    top = result[
        "top_level_comparison"
    ]

    assert top["lut"]["predicted"] == 500
    assert top["lut"]["actual"] == 4691
    assert top["lut"]["direction"] == (
        "underestimated"
    )

    assert top["bram18"]["predicted"] == 2
    assert top["bram18"]["actual"] == 27


def test_missing_primary_module_is_reported() -> None:
    breakdown = _module_breakdown()
    breakdown["primary_modules"] = [
        module
        for module in breakdown[
            "primary_modules"
        ]
        if module["op_type"] != "Softmax"
    ]

    result = build_hls_layer_validation(
        resource_estimate=(
            _resource_estimate()
        ),
        performance_estimate=(
            _performance_estimate()
        ),
        module_breakdown=breakdown,
    )

    softmax = next(
        row
        for row in result[
            "matched_layers"
        ]
        if row["layer_name"] == "softmax0"
    )

    assert softmax["module"] is None
    assert softmax["comparisons"] is None
    assert softmax["match_method"] == (
        "no_primary_module_report"
    )


def test_extra_primary_module_is_reported() -> None:
    breakdown = _module_breakdown()
    breakdown[
        "primary_modules"
    ].append(
        _module(
            name="softmax_typed_extra",
            op_type="Softmax",
            lut=100,
            ff=100,
            dsp=1,
            bram18=0,
            latency_cycles=20,
        )
    )

    result = build_hls_layer_validation(
        resource_estimate=(
            _resource_estimate()
        ),
        performance_estimate=(
            _performance_estimate()
        ),
        module_breakdown=breakdown,
    )

    assert len(
        result[
            "unmatched_primary_modules"
        ]
    ) == 1

    assert result[
        "unmatched_primary_modules"
    ][0]["module"] == (
        "softmax_typed_extra"
    )

    softmax = next(
        row
        for row in result[
            "matched_layers"
        ]
        if row["layer_name"] == "softmax0"
    )

    assert softmax[
        "ambiguous_match"
    ] is True


def test_multiple_same_type_layers_are_ambiguous() -> None:
    resources = _resource_estimate()
    performance = _performance_estimate()
    breakdown = _module_breakdown()

    resources["layers"].append(
        {
            "layer_index": 4,
            "layer_name": "relu1",
            "op_type": "Relu",
            "predicted_lut": 200,
            "predicted_ff": 100,
            "predicted_dsp": 0,
            "predicted_bram18": 0,
        }
    )
    performance[
        "layer_cycles"
    ].append(
        {
            "layer_index": 4,
            "layer_name": "relu1",
            "op_type": "Relu",
            "predicted_cycles": 100,
        }
    )
    breakdown[
        "primary_modules"
    ].append(
        _module(
            name="relu_typed_second",
            op_type="Relu",
            lut=210,
            ff=90,
            dsp=0,
            bram18=0,
            latency_cycles=102,
        )
    )

    result = build_hls_layer_validation(
        resource_estimate=resources,
        performance_estimate=performance,
        module_breakdown=breakdown,
    )

    relu_rows = [
        row
        for row in result[
            "matched_layers"
        ]
        if row["op_type"] == "Relu"
    ]

    assert len(relu_rows) == 2
    assert all(
        row["ambiguous_match"]
        for row in relu_rows
    )


def test_writes_validation_artifacts(
    tmp_path: Path,
) -> None:
    result = run_hls_layer_validation(
        out_dir=tmp_path,
        resource_estimate=(
            _resource_estimate()
        ),
        performance_estimate=(
            _performance_estimate()
        ),
        module_breakdown=(
            _module_breakdown()
        ),
    )

    assert result.available is True
    assert result.results_json.is_file()
    assert result.results_csv.is_file()
    assert result.summary_txt.is_file()

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8",
        )
    )

    assert len(
        payload["matched_layers"]
    ) == 4

    with result.results_csv.open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        rows = list(
            csv.DictReader(input_file)
        )

    assert len(rows) == 4

    dense = next(
        row
        for row in rows
        if row["layer_name"] == "dense0"
    )

    assert float(
        dense["predicted_dsp"]
    ) == 2.0
    assert float(
        dense["actual_dsp"]
    ) == 2.0
    assert float(
        dense["cycle_error_percent"]
    ) == 0.0

    assert "FPGAI Layer vs HLS Validation" in (
        result.terminal_summary
    )
    assert "Operator models requiring revision" in (
        result.terminal_summary
    )


def test_accepts_json_file_inputs(
    tmp_path: Path,
) -> None:
    resources_path = (
        tmp_path / "resources.json"
    )
    performance_path = (
        tmp_path / "performance.json"
    )
    modules_path = (
        tmp_path / "modules.json"
    )

    resources_path.write_text(
        json.dumps(
            _resource_estimate()
        ),
        encoding="utf-8",
    )
    performance_path.write_text(
        json.dumps(
            _performance_estimate()
        ),
        encoding="utf-8",
    )
    modules_path.write_text(
        json.dumps(
            _module_breakdown()
        ),
        encoding="utf-8",
    )

    result = run_hls_layer_validation(
        out_dir=tmp_path / "build",
        resource_estimate=resources_path,
        performance_estimate=(
            performance_path
        ),
        module_breakdown=modules_path,
    )

    assert result.available is True
    assert result.results_json.is_file()


def test_unavailable_breakdown_produces_valid_report(
    tmp_path: Path,
) -> None:
    breakdown = {
        "available": False,
        "primary_modules": [],
        "unassigned_top_resources": {},
    }

    result = run_hls_layer_validation(
        out_dir=tmp_path,
        resource_estimate=(
            _resource_estimate()
        ),
        performance_estimate=(
            _performance_estimate()
        ),
        module_breakdown=breakdown,
    )

    assert result.available is False
    assert result.results_json.is_file()
    assert result.results_csv.is_file()
    assert result.summary_txt.is_file()

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8",
        )
    )

    assert payload["available"] is False
    assert "No HLS module breakdown" in (
        result.terminal_summary
    )