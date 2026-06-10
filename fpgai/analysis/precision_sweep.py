from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from fpgai.analysis.quantization_report import (
    run_quantization_report,
)
from fpgai.numerics.precision_policy import (
    DEFAULT_PRECISION,
    PRECISION_ROLES,
    normalize_precision_spec,
    sweep_layer_override_mode,
)


def _cfg_get(
    data: Dict[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in path.split("."):
        if (
            not isinstance(current, dict)
            or key not in current
        ):
            return default

        current = current[key]

    return current


def _set_cfg(
    data: Dict[str, Any],
    path: str,
    value: Any,
) -> None:
    parts = path.split(".")
    current = data

    for key in parts[:-1]:
        if (
            key not in current
            or not isinstance(current[key], dict)
        ):
            current[key] = {}

        current = current[key]

    current[parts[-1]] = value


def _candidate_defaults(
    candidate: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    defaults = candidate.get(
        "defaults",
        {},
    ) or {}

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


def _format_spec(
    spec: Dict[str, Any],
) -> str:
    return (
        f"ap_fixed<"
        f"{int(spec['total_bits'])},"
        f"{int(spec['int_bits'])}>"
    )


def _recommend_candidate(
    rows: List[Dict[str, Any]],
    *,
    require_prediction_match: bool,
    minimum_cosine: float,
) -> Dict[str, Any] | None:
    if not rows:
        return None

    valid = []

    for row in rows:
        if (
            require_prediction_match
            and not row["prediction_match"]
        ):
            continue

        if float(row["output_cosine"]) < minimum_cosine:
            continue

        valid.append(row)

    if valid:
        return min(
            valid,
            key=lambda row: (
                int(row["activation_bits"]),
                int(row["weight_bits"]),
                int(row["bias_bits"]),
                int(row["accum_bits"]),
                float(row["output_mse"]),
            ),
        )

    prediction_matches = [
        row
        for row in rows
        if row["prediction_match"]
    ]

    if prediction_matches:
        return min(
            prediction_matches,
            key=lambda row: (
                -float(row["output_cosine"]),
                float(row["output_mse"]),
                int(row["activation_bits"]),
                int(row["weight_bits"]),
            ),
        )

    return min(
        rows,
        key=lambda row: (
            float(row["output_mse"]),
            -float(row["output_cosine"]),
        ),
    )


@dataclass(frozen=True)
class PrecisionSweepResult:
    out_dir: Path
    results_json: Path
    summary_txt: Path
    results_csv: Path
    passed: bool


def run_precision_sweep(
    *,
    model_path: str | Path,
    raw_cfg: Dict[str, Any],
    out_dir: str | Path,
) -> PrecisionSweepResult:
    output_root = Path(out_dir).resolve()
    sweep_dir = output_root / "precision_sweep"

    if sweep_dir.exists():
        for path in sweep_dir.glob("**/*"):
            if path.is_file():
                path.unlink()

    sweep_dir.mkdir(
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
            "Precision sweep enabled but no "
            "analysis.precision_sweep.candidates "
            "were provided"
        )

    if not isinstance(candidates, list):
        raise RuntimeError(
            "analysis.precision_sweep.candidates "
            "must be a list"
        )

    seen_names: set[str] = set()
    rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise RuntimeError(
                "Precision sweep candidate "
                f"{index} must be a mapping"
            )

        name = str(
            candidate.get(
                "name",
                f"candidate_{index}",
            )
        ).strip()

        if not name:
            raise RuntimeError(
                "Precision sweep candidate names "
                "cannot be empty"
            )

        if name in seen_names:
            raise RuntimeError(
                "Duplicate precision sweep "
                f"candidate name: {name}"
            )

        seen_names.add(name)

        defaults = _candidate_defaults(candidate)
        override_mode = sweep_layer_override_mode(
            sweep_cfg,
            candidate,
        )

        candidate_cfg = copy.deepcopy(raw_cfg)

        for role in PRECISION_ROLES:
            _set_cfg(
                candidate_cfg,
                f"numerics.defaults.{role}",
                defaults[role],
            )

        if override_mode == "clear":
            _set_cfg(
                candidate_cfg,
                "numerics.layers",
                [],
            )

        _set_cfg(
            candidate_cfg,
            "analysis.quantization_report.enabled",
            True,
        )

        candidate_dir = sweep_dir / name
        candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        quant_result = run_quantization_report(
            model_path=model_path,
            raw_cfg=candidate_cfg,
            out_dir=candidate_dir,
        )

        quant_metrics = json.loads(
            quant_result.metrics_json.read_text(
                encoding="utf-8"
            )
        )

        final = quant_metrics["final"]
        worst = quant_metrics.get(
            "worst_layer",
        ) or {}

        row = {
            "name": name,
            "layer_overrides": override_mode,
            "activation_bits": int(
                defaults["activation"]["total_bits"]
            ),
            "activation_int_bits": int(
                defaults["activation"]["int_bits"]
            ),
            "weight_bits": int(
                defaults["weight"]["total_bits"]
            ),
            "weight_int_bits": int(
                defaults["weight"]["int_bits"]
            ),
            "bias_bits": int(
                defaults["bias"]["total_bits"]
            ),
            "bias_int_bits": int(
                defaults["bias"]["int_bits"]
            ),
            "accum_bits": int(
                defaults["accum"]["total_bits"]
            ),
            "accum_int_bits": int(
                defaults["accum"]["int_bits"]
            ),
            "output_mse": float(
                final["output_mse"]
            ),
            "output_mae": float(
                final["output_mae"]
            ),
            "output_max_abs": float(
                final["output_max_abs"]
            ),
            "output_cosine": float(
                final["output_cosine"]
            ),
            "float_top1": int(
                final["float_top1"]
            ),
            "quant_top1": int(
                final["quant_top1"]
            ),
            "prediction_match": bool(
                final["prediction_match"]
            ),
            "worst_layer_name": str(
                worst.get(
                    "layer_name",
                    "",
                )
            ),
            "worst_layer_type": str(
                worst.get(
                    "op_type",
                    "",
                )
            ),
            "worst_layer_mse": float(
                worst.get(
                    "mse",
                    0.0,
                )
            ),
            "quant_metrics_json": str(
                quant_result.metrics_json
            ),
            "quant_summary_txt": str(
                quant_result.summary_txt
            ),
            "quant_layerwise_csv": str(
                quant_result.layerwise_csv
            ),
        }

        rows.append(row)

        detailed_results.append(
            {
                "name": name,
                "defaults": defaults,
                "layer_overrides": override_mode,
                "effective_layer_rules": (
                    candidate_cfg
                    .get("numerics", {})
                    .get("layers", [])
                ),
                "quant_report": quant_metrics,
            }
        )

    require_prediction_match = bool(
        sweep_cfg.get(
            "require_prediction_match",
            True,
        )
    )

    minimum_cosine = float(
        sweep_cfg.get(
            "minimum_cosine",
            0.99,
        )
    )

    recommended = _recommend_candidate(
        rows,
        require_prediction_match=(
            require_prediction_match
        ),
        minimum_cosine=minimum_cosine,
    )

    results_json = sweep_dir / "results.json"
    results_csv = sweep_dir / "results.csv"
    summary_txt = sweep_dir / "summary.txt"

    payload = {
        "model_path": str(model_path),
        "settings": {
            "require_prediction_match": (
                require_prediction_match
            ),
            "minimum_cosine": minimum_cosine,
            "default_layer_overrides": (
                sweep_cfg.get(
                    "layer_overrides",
                    "clear",
                )
            ),
        },
        "recommended": recommended,
        "results": rows,
        "detailed_results": detailed_results,
    }

    results_json.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
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
        "quant_metrics_json",
        "quant_summary_txt",
        "quant_layerwise_csv",
    ]

    with results_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=csv_fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    lines: List[str] = [
        "=============== FPGAI Precision Sweep ===============",
        f"Model path         : {model_path}",
        f"Candidates         : {len(rows)}",
        (
            "Layer overrides    : "
            f"{sweep_cfg.get('layer_overrides', 'clear')}"
        ),
        (
            "Minimum cosine     : "
            f"{minimum_cosine}"
        ),
        (
            "Require pred match : "
            f"{require_prediction_match}"
        ),
        "-----------------------------------------------------",
        (
            "Name       Overrides Act          Wgt          "
            "Bias         Acc          MSE         "
            "Cosine      Match"
        ),
    ]

    for row in rows:
        activation = _format_spec(
            {
                "total_bits": row["activation_bits"],
                "int_bits": (
                    row["activation_int_bits"]
                ),
            }
        )
        weight = _format_spec(
            {
                "total_bits": row["weight_bits"],
                "int_bits": row["weight_int_bits"],
            }
        )
        bias = _format_spec(
            {
                "total_bits": row["bias_bits"],
                "int_bits": row["bias_int_bits"],
            }
        )
        accum = _format_spec(
            {
                "total_bits": row["accum_bits"],
                "int_bits": row["accum_int_bits"],
            }
        )

        lines.append(
            f"{row['name']:<10} "
            f"{row['layer_overrides']:<9} "
            f"{activation:<12} "
            f"{weight:<12} "
            f"{bias:<12} "
            f"{accum:<12} "
            f"{row['output_mse']:<11.8f} "
            f"{row['output_cosine']:<11.8f} "
            f"{row['prediction_match']}"
        )

    lines.append(
        "-----------------------------------------------------"
    )

    if recommended is not None:
        lines.append(
            f"Recommended        : {recommended['name']}"
        )
        lines.append(
            "Recommendation     : "
            f"prediction_match="
            f"{recommended['prediction_match']}, "
            f"cosine="
            f"{recommended['output_cosine']:.8f}, "
            f"mse="
            f"{recommended['output_mse']:.8f}"
        )
    else:
        lines.append(
            "Recommended        : None"
        )

    lines.extend(
        [
            "-----------------------------------------------------",
            f"Results JSON       : {results_json}",
            f"Results CSV        : {results_csv}",
            f"Summary TXT        : {summary_txt}",
            "=====================================================",
        ]
    )

    summary_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return PrecisionSweepResult(
        out_dir=sweep_dir,
        results_json=results_json,
        summary_txt=summary_txt,
        results_csv=results_csv,
        passed=True,
    )