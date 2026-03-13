from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import copy
import csv
import json

from fpgai.analysis.precision_sweep import run_precision_sweep
from fpgai.analysis.resource_estimator import estimate_resources_from_descriptors
from fpgai.analysis.performance_estimator import estimate_performance
from fpgai.engine.analysis import analyze_graph


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


def _recommend_smallest_valid(rows: List[Dict[str, Any]], require_match: bool, min_cosine: float):
    valid = []
    for r in rows:
        if require_match and not r["prediction_match"]:
            continue
        if float(r["output_cosine"]) < min_cosine:
            continue
        valid.append(r)
    if not valid:
        return None
    return sorted(valid, key=lambda r: (r["activation_bits"], r["weight_bits"], r["predicted_lut"], r["output_mse"]))[0]


def _recommend_balanced(rows: List[Dict[str, Any]], require_match: bool):
    valid = [r for r in rows if (r["prediction_match"] or not require_match)]
    if not valid:
        valid = rows[:]
    if not valid:
        return None

    max_mse = max(float(r["output_mse"]) for r in valid) or 1.0
    max_lut = max(float(r["predicted_lut"]) for r in valid) or 1.0
    max_lat = max(float(r["predicted_latency_ms"]) for r in valid) or 1.0

    scored = []
    for r in valid:
        score = (
            0.5 * (float(r["output_mse"]) / max_mse) +
            0.25 * (float(r["predicted_lut"]) / max_lut) +
            0.25 * (float(r["predicted_latency_ms"]) / max_lat)
        )
        scored.append((score, r))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _recommend_best_accuracy(rows: List[Dict[str, Any]]):
    if not rows:
        return None
    return sorted(rows, key=lambda r: (float(r["output_mse"]), -float(r["output_cosine"])))[0]


def _terminal_summary(rows: List[Dict[str, Any]], rec_smallest, rec_balanced, rec_best) -> str:
    lines: List[str] = []
    lines.append("=============== FPGAI Design Space Summary ===============")
    lines.append("Candidate  Match  Cosine    MSE         LUT      DSP   BRAM  Lat(ms)  Speedup  Verdict")
    for r in rows:
        verdict = ""
        if not r["prediction_match"]:
            verdict = "too aggressive"
        elif rec_smallest is not None and r["name"] == rec_smallest["name"]:
            verdict = "smallest valid"
        elif rec_balanced is not None and r["name"] == rec_balanced["name"]:
            verdict = "balanced"
        elif rec_best is not None and r["name"] == rec_best["name"]:
            verdict = "best accuracy"

        lines.append(
            f"{r['name']:<10} "
            f"{str(r['prediction_match']):<5} "
            f"{r['output_cosine']:<9.6f} "
            f"{r['output_mse']:<11.6g} "
            f"{r['predicted_lut']:<8} "
            f"{r['predicted_dsp']:<5} "
            f"{r['predicted_bram18']:<5} "
            f"{r['predicted_latency_ms']:<8.4f} "
            f"{r['predicted_speedup_vs_cpu']:<8.2f} "
            f"{verdict}"
        )

    lines.append("----------------------------------------------------------")
    if rec_smallest is not None:
        lines.append(f"Recommended (smallest valid): {rec_smallest['name']}")
    if rec_balanced is not None:
        lines.append(f"Recommended (balanced)      : {rec_balanced['name']}")
    if rec_best is not None:
        lines.append(f"Recommended (best accuracy) : {rec_best['name']}")
    lines.append("==========================================================")
    return "\n".join(lines)


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
    graph,
    model_path: str | Path,
    raw_cfg: Dict[str, Any],
    out_dir: str | Path,
) -> DesignSpaceReportResult:
    out_dir = Path(out_dir).resolve()
    ddir = out_dir / "design_space"
    if ddir.exists():
        for p in ddir.glob("**/*"):
            if p.is_file():
                p.unlink()
    ddir.mkdir(parents=True, exist_ok=True)

    sweep_cfg = _cfg_get(raw_cfg, "analysis.precision_sweep", {}) or {}
    candidates = sweep_cfg.get("candidates", []) or []
    if not candidates:
        raise RuntimeError("Design space report requires analysis.precision_sweep.candidates")

    sweep_result = run_precision_sweep(
        model_path=model_path,
        raw_cfg=raw_cfg,
        out_dir=out_dir,
    )
    sweep_payload = json.loads(sweep_result.results_json.read_text(encoding="utf-8"))
    sweep_rows = sweep_payload["results"]

    cand_map = {str(c.get("name", f"candidate_{i}")): c for i, c in enumerate(candidates)}

    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []

    for row in sweep_rows:
        name = row["name"]
        cand = cand_map[name]
        defaults = _candidate_defaults(cand)

        cand_cfg = copy.deepcopy(raw_cfg)
        _set_cfg(cand_cfg, "numerics.defaults.activation", defaults["activation"])
        _set_cfg(cand_cfg, "numerics.defaults.weight", defaults["weight"])
        _set_cfg(cand_cfg, "numerics.defaults.bias", defaults["bias"])
        _set_cfg(cand_cfg, "numerics.defaults.accum", defaults["accum"])

        descs = analyze_graph(graph)
        resource_est = estimate_resources_from_descriptors(descs, cand_cfg)
        perf_est = estimate_performance(resource_estimate=resource_est, raw_cfg=cand_cfg)

        merged = dict(row)
        merged.update(resource_est["totals"])
        merged.update(perf_est)
        rows.append(merged)

        detailed.append(
            {
                "name": name,
                "defaults": defaults,
                "quant": row,
                "resource_estimate": resource_est,
                "performance_estimate": perf_est,
            }
        )

    require_match = bool(_cfg_get(raw_cfg, "analysis.design_space.recommendation.require_prediction_match", True))
    min_cosine = float(_cfg_get(raw_cfg, "analysis.design_space.recommendation.min_cosine", 0.999))

    rec_smallest = _recommend_smallest_valid(rows, require_match, min_cosine)
    rec_balanced = _recommend_balanced(rows, require_match)
    rec_best = _recommend_best_accuracy(rows)

    terminal_summary = _terminal_summary(rows, rec_smallest, rec_balanced, rec_best)

    results_json = ddir / "results.json"
    results_csv = ddir / "results.csv"
    summary_txt = ddir / "summary.txt"

    payload = {
        "model_path": str(model_path),
        "recommended_smallest_valid": rec_smallest,
        "recommended_balanced": rec_balanced,
        "recommended_best_accuracy": rec_best,
        "results": rows,
        "detailed_results": detailed,
        "terminal_summary": terminal_summary,
    }
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_fields = [
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
        "predicted_lut",
        "predicted_ff",
        "predicted_dsp",
        "predicted_bram18",
        "total_macs",
        "clock_mhz",
        "predicted_parallel_macs",
        "predicted_cycles",
        "predicted_latency_ms",
        "predicted_throughput_fps",
        "predicted_speedup_vs_cpu",
    ]

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    summary_txt.write_text(terminal_summary + "\n", encoding="utf-8")

    return DesignSpaceReportResult(
        out_dir=ddir,
        results_json=results_json,
        summary_txt=summary_txt,
        results_csv=results_csv,
        terminal_summary=terminal_summary,
        passed=True,
    )