#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"

DEFAULT_MODELS = [
    "models/mlp.onnx",
    "models/cnn_mnist.onnx",
]

DEFAULT_PRECISIONS = [
    "Uniform-12",
    "Uniform-16",
    "Mixed-Conservative",
]

DEFAULT_POLICIES = [
    "Resource-First",
    "Balanced",
    "Latency-First",
]


@dataclass
class SweepCase:
    model_path: str
    precision_policy: str
    parallel_policy: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_nested(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def sanitize_name(x: str) -> str:
    out = x.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out = out.replace(".onnx", "")
    out = out.replace(":", "_")
    out = out.replace("-", "_")
    return out


def resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def parse_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def tail_text_file(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def recursive_find_files(root: Path, name_candidates: Iterable[str]) -> List[Path]:
    names = set(name_candidates)
    hits: List[Path] = []
    if not root.exists():
        return hits
    for p in root.rglob("*"):
        if p.is_file() and p.name in names:
            hits.append(p)
    return hits


def maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def maybe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def parse_csynth_xml(xml_path: Path) -> Dict[str, Any]:
    if not xml_path.exists():
        return {}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return {}

    def find_text(paths: List[str]) -> Optional[str]:
        for p in paths:
            e = root.find(p)
            if e is not None and e.text is not None:
                return e.text.strip()
        return None

    clock_ns = maybe_float(find_text([
        ".//EstimatedClockPeriod",
        ".//SummaryOfTimingAnalysis/EstimatedClockPeriod",
    ]))

    unc_ns = maybe_float(find_text([
        ".//SummaryOfTimingAnalysis/EstimatedClockUncertainty",
        ".//EstimatedClockUncertainty",
    ]))

    lat_min = maybe_int(find_text([
        ".//SummaryOfOverallLatency/Best-caseLatency",
        ".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency",
    ]))
    lat_max = maybe_int(find_text([
        ".//SummaryOfOverallLatency/Worst-caseLatency",
        ".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency",
    ]))
    ii = maybe_int(find_text([
        ".//SummaryOfOverallLatency/Interval-min",
        ".//PerformanceEstimates/SummaryOfOverallLatency/Interval-min",
    ]))

    lut = maybe_int(find_text([
        ".//AreaEstimates/Resources/LUT",
        ".//Resources/LUT",
    ]))
    ff = maybe_int(find_text([
        ".//AreaEstimates/Resources/FF",
        ".//Resources/FF",
    ]))
    dsp = maybe_int(find_text([
        ".//AreaEstimates/Resources/DSP",
        ".//Resources/DSP",
    ]))
    bram = maybe_int(find_text([
        ".//AreaEstimates/Resources/BRAM_18K",
        ".//Resources/BRAM_18K",
    ]))
    uram = maybe_int(find_text([
        ".//AreaEstimates/Resources/URAM",
        ".//Resources/URAM",
    ]))

    avail_lut = maybe_int(find_text([
        ".//AreaEstimates/AvailableResources/LUT",
    ]))
    avail_ff = maybe_int(find_text([
        ".//AreaEstimates/AvailableResources/FF",
    ]))
    avail_dsp = maybe_int(find_text([
        ".//AreaEstimates/AvailableResources/DSP",
    ]))
    avail_bram = maybe_int(find_text([
        ".//AreaEstimates/AvailableResources/BRAM_18K",
    ]))
    avail_uram = maybe_int(find_text([
        ".//AreaEstimates/AvailableResources/URAM",
    ]))

    lat_s_min = lat_min * clock_ns * 1e-9 if lat_min is not None and clock_ns is not None else None
    lat_s_max = lat_max * clock_ns * 1e-9 if lat_max is not None and clock_ns is not None else None

    out = {
        "latency_cycles_min": lat_min,
        "latency_cycles_max": lat_max,
        "latency_seconds_min": lat_s_min,
        "latency_seconds_max": lat_s_max,
        "ii": ii,
        "estimated_clock_ns": clock_ns,
        "estimated_clock_uncertainty_ns": unc_ns,
        "lut": lut,
        "ff": ff,
        "dsp": dsp,
        "bram_18k": bram,
        "uram": uram,
        "avail_lut": avail_lut,
        "avail_ff": avail_ff,
        "avail_dsp": avail_dsp,
        "avail_bram_18k": avail_bram,
        "avail_uram": avail_uram,
    }

    if lat_s_min is not None:
        out["latency_ms_from_hls"] = lat_s_min * 1e3
    if ii not in (None, 0) and clock_ns is not None:
        out["throughput_fps_est"] = 1.0 / (ii * clock_ns * 1e-9)

    return out


def find_csynth_xml(output_dir: Path) -> Optional[Path]:
    candidates = [
        output_dir / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "deeplearn_csynth.xml",
        output_dir / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.xml",
        output_dir / "hls" / "solution1" / "syn" / "report" / "csynth.xml",
        output_dir / "hls" / "syn" / "report" / "csynth.xml",
    ]
    for c in candidates:
        if c.exists():
            return c
    all_xml = list(output_dir.glob("**/*csynth*.xml"))
    return all_xml[0] if all_xml else None


def find_hls_logs(output_dir: Path) -> Dict[str, str]:
    stdout_hits = list(output_dir.rglob("vitis_hls_stdout.log"))
    stderr_hits = list(output_dir.rglob("vitis_hls_stderr.log"))

    out: Dict[str, str] = {
        "hls_stdout_log": str(stdout_hits[0]) if stdout_hits else "",
        "hls_stderr_log": str(stderr_hits[0]) if stderr_hits else "",
        "hls_stdout_tail": tail_text_file(stdout_hits[0], 80) if stdout_hits else "",
        "hls_stderr_tail": tail_text_file(stderr_hits[0], 80) if stderr_hits else "",
    }
    return out


def apply_precision_policy(cfg: Dict[str, Any], precision_policy: str) -> Dict[str, Any]:
    if "numerics" not in cfg:
        cfg["numerics"] = {}
    if "defaults" not in cfg["numerics"]:
        cfg["numerics"]["defaults"] = {}

    if precision_policy == "Uniform-12":
        cfg["numerics"]["defaults"] = {
            "activation": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
            "weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
            "bias": {"type": "ap_fixed", "total_bits": 20, "int_bits": 8},
            "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
        }
        cfg["numerics"]["layers"] = []

    elif precision_policy == "Uniform-16":
        cfg["numerics"]["defaults"] = {
            "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
            "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
            "bias": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
            "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
        }
        cfg["numerics"]["layers"] = []

    elif precision_policy == "Mixed-Conservative":
        cfg["numerics"]["defaults"] = {
            "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
            "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
            "bias": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
            "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
        }
        cfg["numerics"]["layers"] = [
            {
                "match": {"op_type": "Conv"},
                "weight": {"type": "ap_fixed", "total_bits": 14, "int_bits": 5},
                "bias": {"type": "ap_fixed", "total_bits": 20, "int_bits": 8},
                "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
            },
            {
                "match": {"op_type": "Dense"},
                "weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                "bias": {"type": "ap_fixed", "total_bits": 20, "int_bits": 8},
                "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
            },
            {
                "match": {"op_type": "Sigmoid"},
                "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 4},
            },
            {
                "match": {"op_type": "Softmax"},
                "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 4},
            },
        ]
    else:
        raise ValueError(f"Unsupported precision policy: {precision_policy}")

    set_nested(cfg, ["analysis", "paper_precision_policy"], precision_policy)
    return cfg


def parse_compile_plan(output_dir: Path) -> Optional[Dict[str, Any]]:
    cands = [
        output_dir / "ir" / "compile_plan.json",
        output_dir / "compile_plan.json",
    ]
    for c in cands:
        if c.exists():
            return parse_json_if_exists(c)

    hits = recursive_find_files(output_dir, ["compile_plan.json"])
    return parse_json_if_exists(hits[0]) if hits else None


def parse_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    cands = [
        output_dir / "bench" / "metrics.json",
        output_dir / "metrics.json",
    ]
    for c in cands:
        if c.exists():
            return parse_json_if_exists(c)

    hits = recursive_find_files(output_dir, ["metrics.json"])
    for h in hits:
        if "bench" in str(h).lower():
            return parse_json_if_exists(h)
    return parse_json_if_exists(hits[0]) if hits else None


def parse_quant_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    q1 = output_dir / "quant_report" / "metrics.json"
    q2 = output_dir / "precision_sweep" / "results.json"
    if q1.exists():
        return parse_json_if_exists(q1)
    if q2.exists():
        return parse_json_if_exists(q2)

    hits = recursive_find_files(output_dir, ["metrics.json", "results.json"])
    for h in hits:
        s = str(h).lower()
        if "quant" in s or "precision" in s:
            return parse_json_if_exists(h)
    return None


def extract_compile_plan_fields(cp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cp:
        return {}

    notes = cp.get("notes", {}) or {}
    lps = cp.get("layer_plans", []) or []

    dense_lp = next((x for x in lps if x.get("op_type") == "Dense"), None)
    conv_lp = next((x for x in lps if x.get("op_type") == "Conv"), None)

    def get_tile(lp: Optional[Dict[str, Any]], key: str) -> Optional[Any]:
        return (lp.get("tile") or {}).get(key) if lp else None

    def get_unroll(lp: Optional[Dict[str, Any]], key: str) -> Optional[Any]:
        return (lp.get("unroll") or {}).get(key) if lp else None

    return {
        "compile_clock_mhz": cp.get("clock_mhz"),
        "parallel_policy_effective": notes.get("parallel_policy"),
        "parallel_pe": notes.get("parallel_pe"),
        "parallel_simd": notes.get("parallel_simd"),
        "parallel_partition_factor": notes.get("parallel_partition_factor"),
        "weight_region_preference": ",".join(notes.get("weight_region_preference", []) or []),
        "activation_region_preference": ",".join(notes.get("activation_region_preference", []) or []),
        "allow_double_buffer": notes.get("allow_double_buffer"),
        "pred_total_param_bytes": get_nested(cp, ["global_resource_budget", "total_param_bytes"]),
        "pred_total_activation_bytes_in": get_nested(cp, ["global_resource_budget", "total_activation_bytes_in"]),
        "pred_total_activation_bytes_out": get_nested(cp, ["global_resource_budget", "total_activation_bytes_out"]),
        "conv_tile_oh": get_tile(conv_lp, "oh"),
        "conv_tile_ow": get_tile(conv_lp, "ow"),
        "conv_tile_oc": get_tile(conv_lp, "oc"),
        "conv_unroll_ic": get_unroll(conv_lp, "ic"),
        "conv_unroll_oc": get_unroll(conv_lp, "oc"),
        "dense_tile_in": get_tile(dense_lp, "in"),
        "dense_tile_out": get_tile(dense_lp, "out"),
        "dense_unroll_in": get_unroll(dense_lp, "in"),
        "dense_unroll_out": get_unroll(dense_lp, "out"),
        "dense_weight_mode": dense_lp.get("weight_mode") if dense_lp else None,
        "conv_weight_mode": conv_lp.get("weight_mode") if conv_lp else None,
    }


def extract_quant_fields(qm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not qm:
        return {}
    return {
        "quant_match": qm.get("match") if "match" in qm else qm.get("passed"),
        "quant_cosine": qm.get("cosine_similarity", qm.get("cosine")),
        "quant_mse": qm.get("mse"),
        "quant_mae": qm.get("mae"),
        "quant_rmse": qm.get("rmse"),
        "quant_max_abs_error": qm.get("max_abs_error", qm.get("max_abs")),
        "quant_argmax_match": qm.get("argmax_match"),
        "quant_first_bad_layer": qm.get("first_bad_layer"),
    }


def extract_bench_fields(bm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not bm:
        return {}
    return {
        "benchmark_passed": bm.get("passed"),
        "bench_cosine": bm.get("cosine_similarity", bm.get("cosine")),
        "bench_mse": bm.get("mse"),
        "bench_mae": bm.get("mae"),
        "bench_rmse": bm.get("rmse"),
        "bench_max_abs_error": bm.get("max_abs_error", bm.get("max_abs")),
        "bench_argmax_match": bm.get("argmax_match"),
        "bench_latency_ms": bm.get("latency_ms"),
    }


def run_case(
    base_cfg_path: Path,
    case: SweepCase,
    out_root: Path,
    keep_temp_configs: bool,
    keep_success_logs: bool,
) -> Dict[str, Any]:
    resolved_model = resolve_model_path(case.model_path)
    model_tag = sanitize_name(Path(resolved_model).name)
    case_tag = f"{model_tag}__{sanitize_name(case.precision_policy)}__{sanitize_name(case.parallel_policy)}"

    temp_dir = Path(tempfile.mkdtemp(prefix="fpgai_policy_", dir=str(out_root)))
    cfg_path = temp_dir / f"{case_tag}.yml"

    cfg = load_yaml(base_cfg_path)
    cfg = apply_precision_policy(cfg, case.precision_policy)

    case_output_dir = out_root / case_tag
    case_project_name = case_tag

    set_nested(cfg, ["project", "name"], case_project_name)
    set_nested(cfg, ["project", "out_dir"], str(case_output_dir))
    set_nested(cfg, ["project", "clean"], True)

    set_nested(cfg, ["model", "path"], str(resolved_model))
    set_nested(cfg, ["pipeline", "mode"], "inference")

    # policy aliases so planner/codegen has more than one chance to see it
    set_nested(cfg, ["analysis", "design_space", "policy_name"], case.parallel_policy)
    set_nested(cfg, ["analysis", "design_space", "parallel_policy"], case.parallel_policy)
    set_nested(cfg, ["optimization", "parallel_policy"], case.parallel_policy)
    set_nested(cfg, ["planner", "parallel_policy"], case.parallel_policy)
    set_nested(cfg, ["planner", "policy"], case.parallel_policy)
    set_nested(cfg, ["compiler", "parallel_policy"], case.parallel_policy)

    dump_yaml(cfg, cfg_path)

    stdout_log = temp_dir / "stdout.log"
    stderr_log = temp_dir / "stderr.log"

    cmd = [sys.executable, str(MAIN_PY), "--config", str(cfg_path)]
    with stdout_log.open("w", encoding="utf-8") as so, stderr_log.open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=so, stderr=se, text=True)

    csynth_xml = find_csynth_xml(case_output_dir)
    hls_metrics = parse_csynth_xml(csynth_xml) if csynth_xml else {}

    cp = parse_compile_plan(case_output_dir)
    bm = parse_metrics(case_output_dir)
    qm = parse_quant_metrics(case_output_dir)

    defaults = get_nested(cfg, ["numerics", "defaults"], {}) or {}
    act = defaults.get("activation", {}) or {}
    wgt = defaults.get("weight", {}) or {}
    bias = defaults.get("bias", {}) or {}
    accum = defaults.get("accum", {}) or {}

    row: Dict[str, Any] = {
        "timestamp": utc_now(),
        "mode": "inference",
        "run_id": case_tag,
        "model_name": model_tag,
        "model_path": str(resolved_model),
        "project_name": case_project_name,
        "out_dir": str(case_output_dir),
        "precision_policy": case.precision_policy,
        "parallel_policy": case.parallel_policy,
        "activation_bits": act.get("total_bits"),
        "activation_int_bits": act.get("int_bits"),
        "weight_bits": wgt.get("total_bits"),
        "weight_int_bits": wgt.get("int_bits"),
        "bias_bits": bias.get("total_bits"),
        "bias_int_bits": bias.get("int_bits"),
        "accum_bits": accum.get("total_bits"),
        "accum_int_bits": accum.get("int_bits"),
        "compile_returncode": proc.returncode,
        "compile_ok": proc.returncode == 0,
        "benchmark_passed": bool((bm or {}).get("passed")) if bm else None,
        "hls_ok": bool(hls_metrics),
        "metrics_json": str(case_output_dir / "bench" / "metrics.json"),
        "compile_plan_json": str(case_output_dir / "ir" / "compile_plan.json"),
        "memory_plan_json": str(case_output_dir / "ir" / "memory_plan.json"),
        "comm_plan_json": str(case_output_dir / "ir" / "comm_plan.json"),
        "manifest_json": str(case_output_dir / "manifest.json"),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "compile_stdout_tail": tail_text_file(stdout_log, 30),
        "compile_stderr_tail": tail_text_file(stderr_log, 30),
        "csynth_report_path": str(csynth_xml) if csynth_xml else "",
    }

    row.update(find_hls_logs(case_output_dir))
    row.update(extract_compile_plan_fields(cp))
    row.update(extract_bench_fields(bm))
    row.update(extract_quant_fields(qm))
    row.update(hls_metrics)

    # util percentages
    for used_key, avail_key, pct_key in [
        ("lut", "avail_lut", "lut_util_pct"),
        ("ff", "avail_ff", "ff_util_pct"),
        ("dsp", "avail_dsp", "dsp_util_pct"),
        ("bram_18k", "avail_bram_18k", "bram_util_pct"),
        ("uram", "avail_uram", "uram_util_pct"),
    ]:
        used = maybe_float(row.get(used_key))
        avail = maybe_float(row.get(avail_key))
        row[pct_key] = 100.0 * used / avail if used is not None and avail not in (None, 0) else None

    if not keep_temp_configs:
        cfg_path.unlink(missing_ok=True)

    if proc.returncode == 0 and not keep_success_logs:
        stdout_log.unlink(missing_ok=True)
        stderr_log.unlink(missing_ok=True)

    return row


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_cases(models: List[str], precisions: List[str], policies: List[str]) -> List[SweepCase]:
    return [SweepCase(m, p, pol) for m in models for p in precisions for pol in policies]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--keep-temp-configs", action="store_true")
    ap.add_argument("--keep-success-logs", action="store_true")
    ap.add_argument("--stop-on-first-error", action="store_true")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--precisions", nargs="*", default=DEFAULT_PRECISIONS)
    ap.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    base_cfg_path = Path(args.config).resolve()
    csv_out = Path(args.csv_out).resolve()
    out_root = csv_out.parent
    out_root.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.models, args.precisions, args.policies)

    print("=" * 72)
    print(f"[INFO] Running inference paper sweep: {len(cases)} cases")
    print("=" * 72)

    rows: List[Dict[str, Any]] = []
    for case in cases:
        print(f"[RUN] model={case.model_path} precision={case.precision_policy} parallel={case.parallel_policy}")
        row = run_case(
            base_cfg_path=base_cfg_path,
            case=case,
            out_root=out_root,
            keep_temp_configs=args.keep_temp_configs,
            keep_success_logs=args.keep_success_logs,
        )
        rows.append(row)

        print(
            f"[DONE] returncode={row['compile_returncode']} "
            f"compile_ok={row['compile_ok']} "
            f"hls_ok={row['hls_ok']} "
            f"effective_policy={row.get('parallel_policy_effective')} "
            f"benchmark_passed={row.get('benchmark_passed')}"
        )

        if args.stop_on_first_error and (not row["compile_ok"] or not row["hls_ok"]):
            print("[STOP] First compile/HLS error encountered.")
            break

    write_csv(rows, csv_out)
    print(f"\n[OK] Sweep completed: {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())