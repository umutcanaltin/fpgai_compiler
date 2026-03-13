from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import copy
import csv
import json

from fpgai.analysis.quantization_report import run_quantization_report


def _cfg_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _set_cfg(d: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def _candidate_defaults(cand: Dict[str, Any]) -> Dict[str, Any]:
    d = cand.get("defaults", {}) or {}
    return {
        "activation": dict(d.get("activation", {"type": "ap_fixed", "total_bits": 16, "int_bits": 6})),
        "weight": dict(d.get("weight", {"type": "ap_fixed", "total_bits": 16, "int_bits": 6})),
        "bias": dict(d.get("bias", {"type": "ap_fixed", "total_bits": 24, "int_bits": 10})),
        "accum": dict(d.get("accum", {"type": "ap_fixed", "total_bits": 24, "int_bits": 10})),
    }


def _fmt_spec(spec: Dict[str, Any]) -> str:
    return f"ap_fixed<{int(spec['total_bits'])},{int(spec['int_bits'])}>"


def _recommend_candidate(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not rows:
        return None

    pred_ok = [r for r in rows if r.get("prediction_match", False)]
    cosine_ok = [r for r in pred_ok if float(r.get("output_cosine", 0.0)) >= 0.99]

    if cosine_ok:
        # Smallest activation bits first, then lowest MSE
        return sorted(
            cosine_ok,
            key=lambda r: (
                int(r["activation_bits"]),
                int(r["weight_bits"]),
                float(r["output_mse"]),
            ),
        )[0]

    if pred_ok:
        return sorted(
            pred_ok,
            key=lambda r: (
                int(r["activation_bits"]),
                int(r["weight_bits"]),
                float(r["output_mse"]),
            ),
        )[0]

    return sorted(rows, key=lambda r: float(r["output_mse"]))[0]


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
    out_dir = Path(out_dir).resolve()
    sdir = out_dir / "precision_sweep"
    if sdir.exists():
        for p in sdir.glob("**/*"):
            if p.is_file():
                p.unlink()
    sdir.mkdir(parents=True, exist_ok=True)

    sweep_cfg = _cfg_get(raw_cfg, "analysis.precision_sweep", {}) or {}
    candidates = sweep_cfg.get("candidates", []) or []

    if not candidates:
        raise RuntimeError("Precision sweep enabled but no analysis.precision_sweep.candidates provided")

    rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []

    for idx, cand in enumerate(candidates):
        name = str(cand.get("name", f"candidate_{idx}"))
        c_defaults = _candidate_defaults(cand)

        cand_cfg = copy.deepcopy(raw_cfg)
        _set_cfg(cand_cfg, "numerics.defaults.activation", c_defaults["activation"])
        _set_cfg(cand_cfg, "numerics.defaults.weight", c_defaults["weight"])
        _set_cfg(cand_cfg, "numerics.defaults.bias", c_defaults["bias"])
        _set_cfg(cand_cfg, "numerics.defaults.accum", c_defaults["accum"])

        # keep layerwise overrides if user wants them
        # quant report must stay enabled for the subrun
        _set_cfg(cand_cfg, "analysis.quantization_report.enabled", True)

        cand_out_dir = sdir / name
        cand_out_dir.mkdir(parents=True, exist_ok=True)

        qres = run_quantization_report(
            model_path=model_path,
            raw_cfg=cand_cfg,
            out_dir=cand_out_dir,
        )

        qmetrics = json.loads(qres.metrics_json.read_text(encoding="utf-8"))
        final = qmetrics["final"]
        worst = qmetrics.get("worst_layer") or {}

        row = {
            "name": name,
            "activation_bits": int(c_defaults["activation"]["total_bits"]),
            "activation_int_bits": int(c_defaults["activation"]["int_bits"]),
            "weight_bits": int(c_defaults["weight"]["total_bits"]),
            "weight_int_bits": int(c_defaults["weight"]["int_bits"]),
            "bias_bits": int(c_defaults["bias"]["total_bits"]),
            "bias_int_bits": int(c_defaults["bias"]["int_bits"]),
            "accum_bits": int(c_defaults["accum"]["total_bits"]),
            "accum_int_bits": int(c_defaults["accum"]["int_bits"]),
            "output_mse": float(final["output_mse"]),
            "output_mae": float(final["output_mae"]),
            "output_max_abs": float(final["output_max_abs"]),
            "output_cosine": float(final["output_cosine"]),
            "float_top1": int(final["float_top1"]),
            "quant_top1": int(final["quant_top1"]),
            "prediction_match": bool(final["prediction_match"]),
            "worst_layer_name": str(worst.get("layer_name", "")),
            "worst_layer_type": str(worst.get("op_type", "")),
            "worst_layer_mse": float(worst.get("mse", 0.0)),
            "quant_metrics_json": str(qres.metrics_json),
            "quant_summary_txt": str(qres.summary_txt),
            "quant_layerwise_csv": str(qres.layerwise_csv),
        }
        rows.append(row)

        detailed_results.append(
            {
                "name": name,
                "defaults": c_defaults,
                "quant_report": qmetrics,
            }
        )

    recommended = _recommend_candidate(rows)

    results_json = sdir / "results.json"
    results_csv = sdir / "results.csv"
    summary_txt = sdir / "summary.txt"

    payload = {
        "model_path": str(model_path),
        "recommended": recommended,
        "results": rows,
        "detailed_results": detailed_results,
    }
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
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
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines: List[str] = []
    lines.append("=============== FPGAI Precision Sweep ===============")
    lines.append(f"Model path         : {model_path}")
    lines.append(f"Candidates         : {len(rows)}")
    lines.append("-----------------------------------------------------")
    lines.append("Name       Act       Wgt       Bias      Acc       MSE         MAE         Cosine      PredMatch")
    for r in rows:
        lines.append(
            f"{r['name']:<10} "
            f"{_fmt_spec({'total_bits': r['activation_bits'], 'int_bits': r['activation_int_bits']}):<10} "
            f"{_fmt_spec({'total_bits': r['weight_bits'], 'int_bits': r['weight_int_bits']}):<10} "
            f"{_fmt_spec({'total_bits': r['bias_bits'], 'int_bits': r['bias_int_bits']}):<11} "
            f"{_fmt_spec({'total_bits': r['accum_bits'], 'int_bits': r['accum_int_bits']}):<11} "
            f"{r['output_mse']:<11.8f} "
            f"{r['output_mae']:<11.8f} "
            f"{r['output_cosine']:<11.8f} "
            f"{str(r['prediction_match'])}"
        )

    if recommended is not None:
        lines.append("-----------------------------------------------------")
        lines.append(f"Recommended        : {recommended['name']}")
        lines.append(
            "Reason             : smallest candidate with good numerical behavior "
            f"(prediction_match={recommended['prediction_match']}, cosine={recommended['output_cosine']:.6f})"
        )

    lines.append("-----------------------------------------------------")
    lines.append(f"Results JSON       : {results_json}")
    lines.append(f"Results CSV        : {results_csv}")
    lines.append(f"Summary TXT        : {summary_txt}")
    lines.append("=====================================================")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return PrecisionSweepResult(
        out_dir=sdir,
        results_json=results_json,
        summary_txt=summary_txt,
        results_csv=results_csv,
        passed=True,
    )