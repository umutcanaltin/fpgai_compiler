from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from fpgai.analysis.performance_estimator import (
    estimate_performance,
)
from fpgai.analysis.precision_sweep import (
    run_precision_sweep,
)
from fpgai.analysis.resource_estimator import (
    estimate_resources_from_descriptors,
)
from fpgai.config.access import get_path
from fpgai.engine.analysis import analyze_graph
from fpgai.numerics.precision_policy import (
    DEFAULT_PRECISION,
    PRECISION_ROLES,
    normalize_precision_spec,
    sweep_layer_override_mode,
)


_cfg_get = get_path


def _set_cfg(
    data: Dict[str, Any],
    path: str,
    value: Any,
) -> None:
    parts = path.split(".")
    current = data

    for key in parts[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[parts[-1]] = value


def _candidate_defaults(
    candidate: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    defaults = candidate.get("defaults", {}) or {}

    if not isinstance(defaults, Mapping):
        raise RuntimeError(
            "Precision sweep candidate defaults must be a mapping"
        )

    return {
        role: normalize_precision_spec(
            defaults.get(role),
            fallback=DEFAULT_PRECISION[role],
            path=(
                "analysis.precision_sweep."
                f"candidates[].defaults.{role}"
            ),
        )
        for role in PRECISION_ROLES
    }


def _candidate_config(
    raw_cfg: Dict[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], str]:
    candidate_cfg = copy.deepcopy(raw_cfg)
    defaults = _candidate_defaults(candidate)

    for role in PRECISION_ROLES:
        _set_cfg(
            candidate_cfg,
            f"numerics.defaults.{role}",
            defaults[role],
        )

    sweep_cfg = _cfg_get(
        raw_cfg,
        "analysis.precision_sweep",
        {},
    ) or {}

    override_mode = sweep_layer_override_mode(
        sweep_cfg,
        dict(candidate),
    )

    if override_mode == "clear":
        _set_cfg(
            candidate_cfg,
            "numerics.layers",
            [],
        )

    return candidate_cfg, defaults, override_mode


def _recommend_smallest_valid(
    rows: List[Dict[str, Any]],
    require_match: bool,
    minimum_cosine: float,
) -> Dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if (
            (row["prediction_match"] or not require_match)
            and float(row["output_cosine"]) >= minimum_cosine
        )
    ]

    if not valid:
        return None

    return min(
        valid,
        key=lambda row: (
            int(row["activation_bits"]),
            int(row["weight_bits"]),
            int(row["bias_bits"]),
            int(row["accum_bits"]),
            int(row["predicted_lut"]),
            float(row["predicted_latency_ms"]),
            float(row["output_mse"]),
        ),
    )


def _recommend_balanced(
    rows: List[Dict[str, Any]],
    require_match: bool,
) -> Dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if row["prediction_match"] or not require_match
    ]

    if not valid:
        valid = list(rows)

    if not valid:
        return None

    max_mse = max(
        float(row["output_mse"])
        for row in valid
    ) or 1.0
    max_lut = max(
        float(row["predicted_lut"])
        for row in valid
    ) or 1.0
    max_latency = max(
        float(row["predicted_latency_ms"])
        for row in valid
    ) or 1.0
    max_dsp = max(
        float(row["predicted_dsp"])
        for row in valid
    ) or 1.0

    scored = []

    for row in valid:
        score = (
            0.45
            * float(row["output_mse"])
            / max_mse
            + 0.20
            * float(row["predicted_lut"])
            / max_lut
            + 0.20
            * float(row["predicted_latency_ms"])
            / max_latency
            + 0.15
            * float(row["predicted_dsp"])
            / max_dsp
        )

        scored.append((score, row))

    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def _recommend_best_accuracy(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    if not rows:
        return None

    return min(
        rows,
        key=lambda row: (
            float(row["output_mse"]),
            -float(row["output_cosine"]),
        ),
    )


def _terminal_summary(
    rows: List[Dict[str, Any]],
    recommended_smallest: Dict[str, Any] | None,
    recommended_balanced: Dict[str, Any] | None,
    recommended_accuracy: Dict[str, Any] | None,
) -> str:
    lines = [
        "=============== FPGAI Design Space Summary ===============",
        (
            "Candidate  Match  Cosine    MSE         LUT      "
            "DSP   BRAM  Cycles      Lat(ms)  Verdict"
        ),
    ]

    for row in rows:
        verdict = ""

        if not row["prediction_match"]:
            verdict = "too aggressive"
        elif (
            recommended_smallest is not None
            and row["name"] == recommended_smallest["name"]
        ):
            verdict = "smallest valid"
        elif (
            recommended_balanced is not None
            and row["name"] == recommended_balanced["name"]
        ):
            verdict = "balanced"
        elif (
            recommended_accuracy is not None
            and row["name"] == recommended_accuracy["name"]
        ):
            verdict = "best accuracy"

        lines.append(
            f"{row['name']:<10} "
            f"{str(row['prediction_match']):<5} "
            f"{float(row['output_cosine']):<9.6f} "
            f"{float(row['output_mse']):<11.6g} "
            f"{int(row['predicted_lut']):<8} "
            f"{int(row['predicted_dsp']):<5} "
            f"{int(row['predicted_bram18']):<5} "
            f"{float(row['predicted_cycles']):<11.0f} "
            f"{float(row['predicted_latency_ms']):<8.4f} "
            f"{verdict}"
        )

    lines.append(
        "----------------------------------------------------------"
    )
    lines.append(
        "Resource model : operator_structural_v2"
    )
    lines.append(
        "Schedule model : operator_execution_schedule_v2"
    )

    if recommended_smallest is not None:
        lines.append(
            "Recommended (smallest valid): "
            f"{recommended_smallest['name']}"
        )

    if recommended_balanced is not None:
        lines.append(
            "Recommended (balanced)      : "
            f"{recommended_balanced['name']}"
        )

    if recommended_accuracy is not None:
        lines.append(
            "Recommended (best accuracy) : "
            f"{recommended_accuracy['name']}"
        )

    lines.append(
        "=========================================================="
    )

    return "\n".join(lines)


def _write_layer_breakdown(
    path: Path,
    detailed_results: List[Dict[str, Any]],
) -> None:
    fields = [
        "candidate",
        "layer_index",
        "layer_name",
        "op_type",
        "activation_bits",
        "weight_bits",
        "bias_bits",
        "accumulator_bits",
        "pe",
        "simd",
        "multiplier_lanes",
        "macs",
        "predicted_lut",
        "predicted_ff",
        "predicted_dsp",
        "predicted_bram18",
        "predicted_cycles",
        "parameter_bram18",
        "bias_bram18",
        "activation_bram18",
        "line_buffer_bram18",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()

        for candidate in detailed_results:
            cycle_map = {
                int(row["layer_index"]): row
                for row in candidate[
                    "performance_estimate"
                ]["layer_cycles"]
            }

            for layer in candidate[
                "resource_estimate"
            ]["layers"]:
                components = layer.get(
                    "resource_components",
                    {},
                )
                cycle_row = cycle_map.get(
                    int(layer["layer_index"]),
                    {},
                )

                writer.writerow(
                    {
                        "candidate": candidate["name"],
                        "layer_index": layer[
                            "layer_index"
                        ],
                        "layer_name": layer[
                            "layer_name"
                        ],
                        "op_type": layer["op_type"],
                        "activation_bits": layer[
                            "activation_bits"
                        ],
                        "weight_bits": layer[
                            "weight_bits"
                        ],
                        "bias_bits": layer["bias_bits"],
                        "accumulator_bits": layer[
                            "accumulator_bits"
                        ],
                        "pe": layer["pe"],
                        "simd": layer["simd"],
                        "multiplier_lanes": layer[
                            "multiplier_lanes"
                        ],
                        "macs": layer["macs"],
                        "predicted_lut": layer[
                            "predicted_lut"
                        ],
                        "predicted_ff": layer[
                            "predicted_ff"
                        ],
                        "predicted_dsp": layer[
                            "predicted_dsp"
                        ],
                        "predicted_bram18": layer[
                            "predicted_bram18"
                        ],
                        "predicted_cycles": cycle_row.get(
                            "predicted_cycles",
                            0,
                        ),
                        "parameter_bram18": components.get(
                            "parameter_bram18",
                            0,
                        ),
                        "bias_bram18": components.get(
                            "bias_bram18",
                            0,
                        ),
                        "activation_bram18": components.get(
                            "activation_bram18",
                            0,
                        ),
                        "line_buffer_bram18": components.get(
                            "line_buffer_bram18",
                            0,
                        ),
                    }
                )


@dataclass(frozen=True)
class DesignSpaceReportResult:
    out_dir: Path
    results_json: Path
    summary_txt: Path
    results_csv: Path
    terminal_summary: str
    passed: bool


def run_design_space_report(
    *,
    graph: Any,
    model_path: str | Path,
    raw_cfg: Dict[str, Any],
    out_dir: str | Path,
) -> DesignSpaceReportResult:
    output_root = Path(out_dir).resolve()
    design_dir = output_root / "design_space"

    if design_dir.exists():
        for path in design_dir.glob("**/*"):
            if path.is_file():
                path.unlink()

    design_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sweep_cfg = _cfg_get(
        raw_cfg,
        "analysis.precision_sweep",
        {},
    ) or {}
    candidates = sweep_cfg.get(
        "candidates",
        [],
    ) or []

    if not candidates:
        raise RuntimeError(
            "Design space report requires "
            "analysis.precision_sweep.candidates"
        )

    if not isinstance(candidates, list):
        raise RuntimeError(
            "analysis.precision_sweep.candidates "
            "must be a list"
        )

    candidate_map: Dict[str, Dict[str, Any]] = {}

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise RuntimeError(
                f"Precision candidate {index} must be a mapping"
            )

        name = str(
            candidate.get(
                "name",
                f"candidate_{index}",
            )
        ).strip()

        if not name:
            raise RuntimeError(
                "Precision candidate names cannot be empty"
            )

        if name in candidate_map:
            raise RuntimeError(
                f"Duplicate precision candidate name: {name}"
            )

        candidate_map[name] = candidate

    sweep_result = run_precision_sweep(
        model_path=model_path,
        raw_cfg=raw_cfg,
        out_dir=output_root,
    )
    sweep_payload = json.loads(
        sweep_result.results_json.read_text(
            encoding="utf-8",
        )
    )
    sweep_rows = sweep_payload["results"]

    descriptors = analyze_graph(graph)

    rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []

    for quant_row in sweep_rows:
        name = str(quant_row["name"])

        if name not in candidate_map:
            raise RuntimeError(
                "Precision sweep returned unknown candidate: "
                f"{name}"
            )

        candidate = candidate_map[name]
        candidate_cfg, defaults, override_mode = (
            _candidate_config(
                raw_cfg,
                candidate,
            )
        )

        resource_estimate = (
            estimate_resources_from_descriptors(
                descriptors,
                candidate_cfg,
            )
        )
        performance_estimate = estimate_performance(
            resource_estimate=resource_estimate,
            raw_cfg=candidate_cfg,
        )

        row = dict(quant_row)
        row.update(resource_estimate["totals"])
        row.update(performance_estimate)
        row["resource_estimation_mode"] = (
            resource_estimate["estimation_mode"]
        )
        row["performance_estimation_mode"] = (
            performance_estimate["estimation_mode"]
        )
        row["layer_overrides"] = override_mode

        rows.append(row)

        detailed_results.append(
            {
                "name": name,
                "defaults": defaults,
                "layer_overrides": override_mode,
                "quantization": quant_row,
                "resource_estimate": resource_estimate,
                "performance_estimate": performance_estimate,
            }
        )

    require_match = bool(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.recommendation."
            "require_prediction_match",
            True,
        )
    )
    minimum_cosine = float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.recommendation."
            "min_cosine",
            0.999,
        )
    )

    recommended_smallest = _recommend_smallest_valid(
        rows,
        require_match,
        minimum_cosine,
    )
    recommended_balanced = _recommend_balanced(
        rows,
        require_match,
    )
    recommended_accuracy = _recommend_best_accuracy(
        rows
    )

    terminal_summary = _terminal_summary(
        rows,
        recommended_smallest,
        recommended_balanced,
        recommended_accuracy,
    )

    results_json = design_dir / "results.json"
    results_csv = design_dir / "results.csv"
    layer_results_csv = (
        design_dir / "layer_breakdown.csv"
    )
    summary_txt = design_dir / "summary.txt"

    payload = {
        "format": "fpgai.design_space.v2",
        "model_path": str(model_path),
        "analytical_models": {
            "resources": "operator_structural_v2",
            "performance": (
                "operator_execution_schedule_v2"
            ),
        },
        "recommendation_policy": {
            "require_prediction_match": require_match,
            "minimum_cosine": minimum_cosine,
        },
        "recommended_smallest_valid": (
            recommended_smallest
        ),
        "recommended_balanced": recommended_balanced,
        "recommended_best_accuracy": (
            recommended_accuracy
        ),
        "results": rows,
        "detailed_results": detailed_results,
        "layer_breakdown_csv": str(
            layer_results_csv
        ),
        "terminal_summary": terminal_summary,
    }

    results_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "name",
        "layer_overrides",
        "activation_bits",
        "activation_int_bits",
        "weight_bits",
        "weight_int_bits",
        "bias_bits",
        "bias_int_bits",
        "accum_bits",
        "accum_int_bits",
        "output_mse",
        "output_mae",
        "output_max_abs",
        "output_cosine",
        "float_top1",
        "quant_top1",
        "prediction_match",
        "worst_layer_name",
        "worst_layer_type",
        "worst_layer_mse",
        "predicted_lut",
        "predicted_ff",
        "predicted_dsp",
        "predicted_bram18",
        "total_macs",
        "total_multiplier_lanes",
        "clock_mhz",
        "predicted_parallel_macs",
        "predicted_compute_cycles",
        "predicted_transfer_cycles",
        "predicted_control_cycles",
        "predicted_cycles",
        "predicted_latency_ms",
        "predicted_throughput_fps",
        "predicted_speedup_vs_cpu",
        "resource_estimation_mode",
        "performance_estimation_mode",
        "quant_metrics_json",
        "quant_summary_txt",
        "quant_layerwise_csv",
    ]

    with results_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=csv_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    _write_layer_breakdown(
        layer_results_csv,
        detailed_results,
    )

    summary_txt.write_text(
        terminal_summary + "\n",
        encoding="utf-8",
    )

    return DesignSpaceReportResult(
        out_dir=design_dir,
        results_json=results_json,
        summary_txt=summary_txt,
        results_csv=results_csv,
        terminal_summary=terminal_summary,
        passed=True,
    )
