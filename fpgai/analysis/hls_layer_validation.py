from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping


RESOURCE_NAMES = (
    "lut",
    "ff",
    "dsp",
    "bram18",
)


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def _comparison(
    predicted: float,
    actual: float,
) -> Dict[str, Any]:
    if actual <= 0.0:
        percentage_error = None
        relative_error = None
    else:
        relative_error = (
            predicted - actual
        ) / actual
        percentage_error = (
            abs(relative_error) * 100.0
        )

    if percentage_error is None:
        quality = "unavailable"
    elif percentage_error <= 10.0:
        quality = "excellent"
    elif percentage_error <= 25.0:
        quality = "good"
    elif percentage_error <= 50.0:
        quality = "rough"
    else:
        quality = "poor"

    if actual <= 0.0:
        direction = "unavailable"
    elif math.isclose(
        predicted,
        actual,
    ):
        direction = "matched"
    elif predicted > actual:
        direction = "overestimated"
    else:
        direction = "underestimated"

    return {
        "predicted": predicted,
        "actual": actual,
        "signed_error": (
            predicted - actual
        ),
        "signed_relative_error": (
            relative_error
        ),
        "absolute_percentage_error": (
            percentage_error
        ),
        "direction": direction,
        "quality": quality,
    }


def _load_json(
    source: str | Path | Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)

    path = Path(source)

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def _layer_cycle_map(
    performance_estimate: Mapping[str, Any],
) -> Dict[int, Mapping[str, Any]]:
    rows = performance_estimate.get(
        "layer_cycles",
        [],
    )

    if not isinstance(rows, list):
        return {}

    result = {}

    for row in rows:
        if not isinstance(row, Mapping):
            continue

        try:
            index = int(
                row.get(
                    "layer_index",
                    -1,
                )
            )
        except (TypeError, ValueError):
            continue

        if index >= 0:
            result[index] = row

    return result


def _primary_modules_by_operator(
    module_breakdown: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    primary_modules = module_breakdown.get(
        "primary_modules",
        [],
    )

    if not isinstance(
        primary_modules,
        list,
    ):
        return {}

    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for module in primary_modules:
        if not isinstance(module, Mapping):
            continue

        op_type = str(
            module.get(
                "op_type",
                "Unknown",
            )
        )

        grouped.setdefault(
            op_type,
            [],
        ).append(
            dict(module)
        )

    # Report-path ordering is stable when multiple instances of the same
    # operator exist. Exact layer names are not always retained by Vitis.
    for modules in grouped.values():
        modules.sort(
            key=lambda module: (
                str(
                    module.get(
                        "report_path",
                        "",
                    )
                ),
                str(
                    module.get(
                        "module",
                        "",
                    )
                ),
            )
        )

    return grouped


def _estimated_layers_by_operator(
    resource_estimate: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    layers = resource_estimate.get(
        "layers",
        [],
    )

    if not isinstance(layers, list):
        return {}

    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for layer in layers:
        if not isinstance(layer, Mapping):
            continue

        op_type = str(
            layer.get(
                "op_type",
                "Unknown",
            )
        )

        grouped.setdefault(
            op_type,
            [],
        ).append(
            dict(layer)
        )

    for operator_layers in grouped.values():
        operator_layers.sort(
            key=lambda layer: int(
                layer.get(
                    "layer_index",
                    0,
                )
            )
        )

    return grouped


def _match_layers_to_modules(
    resource_estimate: Mapping[str, Any],
    performance_estimate: Mapping[str, Any],
    module_breakdown: Mapping[str, Any],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    estimated = _estimated_layers_by_operator(
        resource_estimate
    )
    actual = _primary_modules_by_operator(
        module_breakdown
    )
    cycle_map = _layer_cycle_map(
        performance_estimate
    )

    matched: List[Dict[str, Any]] = []
    unmatched_modules: List[
        Dict[str, Any]
    ] = []

    operator_names = sorted(
        set(estimated)
        | set(actual)
    )

    for op_type in operator_names:
        layers = estimated.get(
            op_type,
            [],
        )
        modules = actual.get(
            op_type,
            [],
        )

        pair_count = min(
            len(layers),
            len(modules),
        )

        ambiguous = (
            len(layers) > 1
            or len(modules) > 1
        )

        for index in range(pair_count):
            layer = layers[index]
            module = modules[index]

            layer_index = int(
                layer.get(
                    "layer_index",
                    index,
                )
            )
            cycle_row = cycle_map.get(
                layer_index,
                {},
            )

            comparisons = {
                resource: _comparison(
                    _safe_float(
                        layer.get(
                            f"predicted_{resource}",
                            0,
                        )
                    ),
                    _safe_float(
                        module.get(
                            resource,
                            0,
                        )
                    ),
                )
                for resource
                in RESOURCE_NAMES
            }

            comparisons[
                "latency_cycles"
            ] = _comparison(
                _safe_float(
                    cycle_row.get(
                        "predicted_cycles",
                        0,
                    )
                ),
                _safe_float(
                    module.get(
                        "latency_cycles",
                        0,
                    )
                ),
            )

            matched.append(
                {
                    "layer_index": (
                        layer_index
                    ),
                    "layer_name": str(
                        layer.get(
                            "layer_name",
                            f"layer_{layer_index}",
                        )
                    ),
                    "op_type": op_type,
                    "module": str(
                        module.get(
                            "module",
                            "",
                        )
                    ),
                    "report_path": str(
                        module.get(
                            "report_path",
                            "",
                        )
                    ),
                    "match_method": (
                        "operator_type_and_order"
                    ),
                    "ambiguous_match": (
                        ambiguous
                    ),
                    "comparisons": (
                        comparisons
                    ),
                }
            )

        for layer in layers[pair_count:]:
            layer_index = int(
                layer.get(
                    "layer_index",
                    0,
                )
            )

            matched.append(
                {
                    "layer_index": (
                        layer_index
                    ),
                    "layer_name": str(
                        layer.get(
                            "layer_name",
                            f"layer_{layer_index}",
                        )
                    ),
                    "op_type": op_type,
                    "module": None,
                    "report_path": None,
                    "match_method": (
                        "no_primary_module_report"
                    ),
                    "ambiguous_match": False,
                    "comparisons": None,
                }
            )

        for module in modules[pair_count:]:
            unmatched_modules.append(
                module
            )

    matched.sort(
        key=lambda row: int(
            row["layer_index"]
        )
    )

    return matched, unmatched_modules


def _top_level_comparison(
    resource_estimate: Mapping[str, Any],
    module_breakdown: Mapping[str, Any],
) -> Dict[str, Any]:
    predicted = resource_estimate.get(
        "top_level",
        {},
    )
    actual = module_breakdown.get(
        "unassigned_top_resources",
        {},
    )

    if not isinstance(predicted, Mapping):
        predicted = {}

    if not isinstance(actual, Mapping):
        actual = {}

    return {
        resource: _comparison(
            _safe_float(
                predicted.get(
                    f"predicted_{resource}",
                    0,
                )
            ),
            _safe_float(
                actual.get(
                    resource,
                    0,
                )
            ),
        )
        for resource in RESOURCE_NAMES
    }


def build_hls_layer_validation(
    *,
    resource_estimate: Mapping[str, Any],
    performance_estimate: Mapping[str, Any],
    module_breakdown: Mapping[str, Any],
) -> Dict[str, Any]:
    matched, unmatched_modules = (
        _match_layers_to_modules(
            resource_estimate,
            performance_estimate,
            module_breakdown,
        )
    )

    poor_rows = []

    for row in matched:
        comparisons = row.get(
            "comparisons"
        )

        if not isinstance(
            comparisons,
            Mapping,
        ):
            continue

        poor_resources = [
            name
            for name, comparison
            in comparisons.items()
            if comparison.get(
                "quality"
            ) == "poor"
        ]

        if poor_resources:
            poor_rows.append(
                {
                    "layer_index": row[
                        "layer_index"
                    ],
                    "layer_name": row[
                        "layer_name"
                    ],
                    "op_type": row[
                        "op_type"
                    ],
                    "module": row[
                        "module"
                    ],
                    "poor_fields": (
                        poor_resources
                    ),
                }
            )

    return {
        "format": (
            "fpgai.hls_layer_validation.v1"
        ),
        "available": bool(
            module_breakdown.get(
                "available",
                False,
            )
        ),
        "resource_model": (
            resource_estimate.get(
                "analytical_model",
                resource_estimate.get(
                    "model",
                ),
            )
        ),
        "performance_model": (
            performance_estimate.get(
                "analytical_performance_model",
                performance_estimate.get(
                    "performance_model",
                ),
            )
        ),
        "match_note": (
            "Vitis function names do not always preserve graph layer names. "
            "Multiple instances of the same operator are matched by stable "
            "layer order and report-path order and are marked ambiguous."
        ),
        "matched_layers": matched,
        "unmatched_primary_modules": (
            unmatched_modules
        ),
        "top_level_comparison": (
            _top_level_comparison(
                resource_estimate,
                module_breakdown,
            )
        ),
        "poor_layers": poor_rows,
        "requires_model_revision": bool(
            poor_rows
        ),
    }


def _format_error(
    comparison: Mapping[str, Any],
) -> str:
    error = comparison.get(
        "absolute_percentage_error"
    )

    if error is None:
        return "n/a"

    return f"{float(error):.2f}%"


def _terminal_summary(
    payload: Mapping[str, Any],
) -> str:
    lines = [
        (
            "=============== FPGAI Layer vs "
            "HLS Validation ==============="
        )
    ]

    if not payload.get(
        "available",
        False,
    ):
        lines.extend(
            [
                (
                    "No HLS module breakdown "
                    "is available."
                ),
                (
                    "============================"
                    "=========================="
                ),
            ]
        )
        return "\n".join(lines)

    lines.append(
        "Layer                Type       "
        "Module                       "
        "LUT err   FF err    DSP err   "
        "BRAM err  Cycle err"
    )

    for row in payload[
        "matched_layers"
    ]:
        comparisons = row.get(
            "comparisons"
        )

        if not isinstance(
            comparisons,
            Mapping,
        ):
            lines.append(
                f"{row['layer_name'][:20]:<20} "
                f"{row['op_type'][:10]:<10} "
                "no primary module report"
            )
            continue

        module = str(
            row.get(
                "module",
                "",
            )
        )

        lines.append(
            f"{str(row['layer_name'])[:20]:<20} "
            f"{str(row['op_type'])[:10]:<10} "
            f"{module[:28]:<28} "
            f"{_format_error(comparisons['lut']):<9} "
            f"{_format_error(comparisons['ff']):<9} "
            f"{_format_error(comparisons['dsp']):<9} "
            f"{_format_error(comparisons['bram18']):<9} "
            f"{_format_error(comparisons['latency_cycles'])}"
        )

    lines.append(
        "----------------------------------------------------------"
    )

    if payload[
        "requires_model_revision"
    ]:
        lines.append(
            "Operator models requiring revision:"
        )

        for row in payload[
            "poor_layers"
        ]:
            fields = ", ".join(
                row["poor_fields"]
            )
            lines.append(
                f" - {row['layer_name']} "
                f"({row['op_type']}): "
                f"{fields}"
            )
    else:
        lines.append(
            "No poor matched operator estimates."
        )

    lines.append(
        "=========================================================="
    )

    return "\n".join(lines)


@dataclass(frozen=True)
class HlsLayerValidationResult:
    out_dir: Path
    results_json: Path
    results_csv: Path
    summary_txt: Path
    terminal_summary: str
    available: bool


def run_hls_layer_validation(
    *,
    out_dir: str | Path,
    resource_estimate: (
        str
        | Path
        | Mapping[str, Any]
    ),
    performance_estimate: (
        str
        | Path
        | Mapping[str, Any]
    ),
    module_breakdown: (
        str
        | Path
        | Mapping[str, Any]
    ),
) -> HlsLayerValidationResult:
    output_root = Path(
        out_dir
    ).resolve()
    validation_dir = (
        output_root
        / "estimate_vs_hls"
        / "layer_validation"
    )
    validation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = build_hls_layer_validation(
        resource_estimate=_load_json(
            resource_estimate
        ),
        performance_estimate=_load_json(
            performance_estimate
        ),
        module_breakdown=_load_json(
            module_breakdown
        ),
    )
    terminal_summary = _terminal_summary(
        payload
    )

    results_json = (
        validation_dir / "results.json"
    )
    results_csv = (
        validation_dir / "layers.csv"
    )
    summary_txt = (
        validation_dir / "summary.txt"
    )

    results_json.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fields = [
        "layer_index",
        "layer_name",
        "op_type",
        "module",
        "match_method",
        "ambiguous_match",
        "predicted_lut",
        "actual_lut",
        "lut_error_percent",
        "predicted_ff",
        "actual_ff",
        "ff_error_percent",
        "predicted_dsp",
        "actual_dsp",
        "dsp_error_percent",
        "predicted_bram18",
        "actual_bram18",
        "bram18_error_percent",
        "predicted_cycles",
        "actual_cycles",
        "cycle_error_percent",
    ]

    with results_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in payload[
            "matched_layers"
        ]:
            comparisons = row.get(
                "comparisons"
            )

            if not isinstance(
                comparisons,
                Mapping,
            ):
                writer.writerow(
                    {
                        "layer_index": row[
                            "layer_index"
                        ],
                        "layer_name": row[
                            "layer_name"
                        ],
                        "op_type": row[
                            "op_type"
                        ],
                        "module": row[
                            "module"
                        ],
                        "match_method": row[
                            "match_method"
                        ],
                        "ambiguous_match": row[
                            "ambiguous_match"
                        ],
                    }
                )
                continue

            writer.writerow(
                {
                    "layer_index": row[
                        "layer_index"
                    ],
                    "layer_name": row[
                        "layer_name"
                    ],
                    "op_type": row[
                        "op_type"
                    ],
                    "module": row["module"],
                    "match_method": row[
                        "match_method"
                    ],
                    "ambiguous_match": row[
                        "ambiguous_match"
                    ],
                    "predicted_lut": comparisons[
                        "lut"
                    ]["predicted"],
                    "actual_lut": comparisons[
                        "lut"
                    ]["actual"],
                    "lut_error_percent": comparisons[
                        "lut"
                    ][
                        "absolute_percentage_error"
                    ],
                    "predicted_ff": comparisons[
                        "ff"
                    ]["predicted"],
                    "actual_ff": comparisons[
                        "ff"
                    ]["actual"],
                    "ff_error_percent": comparisons[
                        "ff"
                    ][
                        "absolute_percentage_error"
                    ],
                    "predicted_dsp": comparisons[
                        "dsp"
                    ]["predicted"],
                    "actual_dsp": comparisons[
                        "dsp"
                    ]["actual"],
                    "dsp_error_percent": comparisons[
                        "dsp"
                    ][
                        "absolute_percentage_error"
                    ],
                    "predicted_bram18": comparisons[
                        "bram18"
                    ]["predicted"],
                    "actual_bram18": comparisons[
                        "bram18"
                    ]["actual"],
                    "bram18_error_percent": comparisons[
                        "bram18"
                    ][
                        "absolute_percentage_error"
                    ],
                    "predicted_cycles": comparisons[
                        "latency_cycles"
                    ]["predicted"],
                    "actual_cycles": comparisons[
                        "latency_cycles"
                    ]["actual"],
                    "cycle_error_percent": comparisons[
                        "latency_cycles"
                    ][
                        "absolute_percentage_error"
                    ],
                }
            )

    summary_txt.write_text(
        terminal_summary + "\n",
        encoding="utf-8",
    )

    return HlsLayerValidationResult(
        out_dir=validation_dir,
        results_json=results_json,
        results_csv=results_csv,
        summary_txt=summary_txt,
        terminal_summary=terminal_summary,
        available=bool(
            payload["available"]
        ),
    )