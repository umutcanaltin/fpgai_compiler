#!/usr/bin/env python3
"""
Paper experiment runner for FPGAI.

This version is defensive and debug-friendly:

- compile first for every case
- collect inner Vitis HLS stdout/stderr tails into CSV
- only run inference benchmark if compile returncode == 0 AND HLS succeeded
- resolve model paths relative to repo root
- records policy, HLS resource/latency, benchmark metrics, quant metrics,
  training metrics, and predicted-vs-actual fields
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import textwrap
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml")
    raise

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
MAIN_PY = REPO_ROOT / "main.py"

DEFAULT_INFER_MODELS = [
    "models/mlp.onnx",
    "models/cnn_mnist.onnx",
]

DEFAULT_TRAIN_MODELS = [
    "models/mlp.onnx",
    "models/cnn_mnist.onnx",
]

DEFAULT_PRECISIONS = [
    "Uniform-8",
    "Uniform-12",
    "Uniform-16",
    "Mixed-Conservative",
]

DEFAULT_POLICIES_INFER = [
    "Resource-First",
    "Balanced",
    "Latency-First",
]

DEFAULT_POLICIES_TRAIN = [
    "Balanced",
    "Throughput-First",
    "Latency-First",
]


@dataclass(frozen=True)
class SweepCase:
    mode: str
    model_path: str
    precision_policy: str
    parallel_policy: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sanitize_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


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


def tail_text_file(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def apply_precision_policy(cfg: Dict[str, Any], precision_policy: str) -> Dict[str, Any]:
    if "numerics" not in cfg:
        cfg["numerics"] = {}
    if "defaults" not in cfg["numerics"]:
        cfg["numerics"]["defaults"] = {}

    if precision_policy == "Uniform-8":
        cfg["numerics"]["defaults"] = {
            "activation": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
            "weight": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
            "bias": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
            "accum": {"type": "ap_fixed", "total_bits": 20, "int_bits": 8},
        }
        cfg["numerics"]["layers"] = []

    elif precision_policy == "Uniform-12":
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


def set_mode_specific_config(cfg: Dict[str, Any], mode: str, benchmark_enabled: bool) -> Dict[str, Any]:
    if mode == "inference":
        set_nested(cfg, ["pipeline", "mode"], "inference")
        set_nested(cfg, ["benchmark", "enabled"], benchmark_enabled)
    elif mode == "training":
        set_nested(cfg, ["pipeline", "mode"], "training_on_device")
        set_nested(cfg, ["benchmark", "enabled"], False)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return cfg


def set_parallel_policy(cfg: Dict[str, Any], policy: str) -> Dict[str, Any]:
    # Keep several aliases because planner/config loaders may consume different ones
    set_nested(cfg, ["analysis", "design_space", "policy_name"], policy)
    set_nested(cfg, ["analysis", "design_space", "parallel_policy"], policy)
    set_nested(cfg, ["optimization", "parallel_policy"], policy)
    set_nested(cfg, ["planner", "parallel_policy"], policy)
    set_nested(cfg, ["planner", "policy"], policy)
    set_nested(cfg, ["compiler", "parallel_policy"], policy)
    return cfg


def set_project_and_model(
    cfg: Dict[str, Any],
    model_path: str,
    out_dir: Path,
    project_name: str,
) -> Dict[str, Any]:
    set_nested(cfg, ["project", "name"], project_name)
    set_nested(cfg, ["project", "out_dir"], str(out_dir))
    set_nested(cfg, ["project", "clean"], True)
    set_nested(cfg, ["model", "path"], model_path)
    return cfg


def enforce_hls_flags(cfg: Dict[str, Any], run_hls: bool = True) -> Dict[str, Any]:
    set_nested(cfg, ["tools", "vitis_hls"], run_hls)
    set_nested(cfg, ["tools", "vivado"], False)
    set_nested(cfg, ["hls", "run"], run_hls)
    set_nested(cfg, ["backends", "hls", "enabled"], True)
    set_nested(cfg, ["backends", "hls", "vitis", "enabled"], True)
    return cfg


def find_csynth_xml(output_dir: Path) -> Optional[Path]:
    direct = [
        output_dir / "hls" / "solution1" / "syn" / "report" / "csynth.xml",
        output_dir / "hls" / "syn" / "report" / "csynth.xml",
    ]
    hit = find_first_existing(direct)
    if hit:
        return hit
    hits = list(output_dir.rglob("csynth.xml"))
    return hits[0] if hits else None


def parse_csynth_xml(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path or not path.exists():
        return out

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return out

    def first_text(xpath: str) -> Optional[str]:
        node = root.find(xpath)
        return node.text.strip() if node is not None and node.text is not None else None

    out["latency_cycles_min"] = maybe_int(first_text(".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"))
    out["latency_cycles_max"] = maybe_int(first_text(".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"))
    out["ii_min"] = maybe_int(first_text(".//PerformanceEstimates/SummaryOfOverallLatency/Interval-min"))
    out["ii_max"] = maybe_int(first_text(".//PerformanceEstimates/SummaryOfOverallLatency/Interval-max"))
    out["estimated_clock_ns"] = maybe_float(first_text(".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"))
    out["target_clock_ns"] = maybe_float(first_text(".//UserAssignments/TargetClockPeriod"))
    out["lut"] = maybe_int(first_text(".//AreaEstimates/Resources/LUT"))
    out["ff"] = maybe_int(first_text(".//AreaEstimates/Resources/FF"))
    out["dsp"] = maybe_int(first_text(".//AreaEstimates/Resources/DSP"))
    out["bram_18k"] = maybe_int(first_text(".//AreaEstimates/Resources/BRAM_18K"))
    out["uram"] = maybe_int(first_text(".//AreaEstimates/Resources/URAM"))
    out["avail_lut"] = maybe_int(first_text(".//AreaEstimates/AvailableResources/LUT"))
    out["avail_ff"] = maybe_int(first_text(".//AreaEstimates/AvailableResources/FF"))
    out["avail_dsp"] = maybe_int(first_text(".//AreaEstimates/AvailableResources/DSP"))
    out["avail_bram_18k"] = maybe_int(first_text(".//AreaEstimates/AvailableResources/BRAM_18K"))
    out["avail_uram"] = maybe_int(first_text(".//AreaEstimates/AvailableResources/URAM"))
    return out


def parse_compile_plan(output_dir: Path) -> Optional[Dict[str, Any]]:
    hits = recursive_find_files(output_dir, ["compile_plan.json"])
    return parse_json_if_exists(hits[0]) if hits else None


def parse_bench_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = recursive_find_files(output_dir, ["metrics.json"])
    for p in candidates:
        if "bench" in str(p.parent).lower():
            return parse_json_if_exists(p)
    return parse_json_if_exists(candidates[0]) if candidates else None


def parse_quant_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    candidates.extend(recursive_find_files(output_dir, ["metrics.json"]))
    candidates.extend(recursive_find_files(output_dir, ["results.json"]))
    for p in candidates:
        s = str(p).lower()
        if "quant" in s or "precision" in s:
            return parse_json_if_exists(p)
    return None


def parse_training_compare(output_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = recursive_find_files(
        output_dir,
        ["training_compare.json", "compare_training.json", "training_metrics.json", "metrics_training.json"],
    )
    return parse_json_if_exists(candidates[0]) if candidates else None


def parse_training_resource_estimate(output_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = recursive_find_files(
        output_dir,
        ["training_resource_estimate.json", "resource_estimate.json", "train_resource_estimate.json"],
    )
    return parse_json_if_exists(candidates[0]) if candidates else None


def parse_memory_plan(output_dir: Path) -> Optional[Dict[str, Any]]:
    hits = recursive_find_files(output_dir, ["memory_plan.json"])
    return parse_json_if_exists(hits[0]) if hits else None


def parse_comm_plan(output_dir: Path) -> Optional[Dict[str, Any]]:
    hits = recursive_find_files(output_dir, ["comm_plan.json"])
    return parse_json_if_exists(hits[0]) if hits else None


def find_hls_log_paths(output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    stdout_candidates = list(output_dir.rglob("vitis_hls_stdout.log"))
    stderr_candidates = list(output_dir.rglob("vitis_hls_stderr.log"))
    return (
        stdout_candidates[0] if stdout_candidates else None,
        stderr_candidates[0] if stderr_candidates else None,
    )


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
        "bench_first_bad_layer": bm.get("first_bad_layer"),
    }


def extract_quant_fields(qm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not qm:
        return {}
    return {
        "quant_match": qm.get("match", qm.get("passed")),
        "quant_cosine": qm.get("cosine_similarity", qm.get("cosine")),
        "quant_mse": qm.get("mse"),
        "quant_mae": qm.get("mae"),
        "quant_rmse": qm.get("rmse"),
        "quant_max_abs_error": qm.get("max_abs_error", qm.get("max_abs")),
        "quant_argmax_match": qm.get("argmax_match"),
        "quant_first_bad_layer": qm.get("first_bad_layer"),
    }


def extract_training_compare_fields(tm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not tm:
        return {}
    g = tm.get("global", {}) or {}
    grads = g.get("grads", {}) or {}
    wb = g.get("weights_before", {}) or {}
    wa = g.get("weights_after", {}) or {}
    return {
        "train_grads_cosine": grads.get("cosine"),
        "train_grads_mae": grads.get("mae"),
        "train_grads_max_abs": grads.get("max_abs"),
        "train_grads_relative_l2": grads.get("relative_l2"),
        "train_weights_before_cosine": wb.get("cosine"),
        "train_weights_before_mae": wb.get("mae"),
        "train_weights_after_cosine": wa.get("cosine"),
        "train_weights_after_mae": wa.get("mae"),
    }


def extract_training_resource_estimate_fields(tr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not tr:
        return {}
    return {
        "pred_weight_storage": tr.get("weight_storage"),
        "pred_activation_storage": tr.get("activation_storage"),
        "pred_gradient_storage": tr.get("gradient_storage"),
        "pred_optimizer_state_storage": tr.get("optimizer_state_storage"),
        "pred_param_bytes": tr.get("param_bytes"),
        "pred_activation_cache_bytes": tr.get("activation_cache_bytes"),
        "pred_gradient_bytes": tr.get("gradient_bytes"),
        "pred_optimizer_state_bytes": tr.get("optimizer_state_bytes"),
        "pred_forward_ops_proxy": tr.get("forward_ops_proxy"),
        "pred_backward_ops_proxy": tr.get("backward_ops_proxy"),
        "pred_update_ops_proxy": tr.get("update_ops_proxy"),
    }


def extract_memory_plan_fields(mp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not mp:
        return {}
    return {
        "mem_weight_region": mp.get("weights_region"),
        "mem_activation_region": mp.get("activations_region"),
        "mem_grad_region": mp.get("gradients_region"),
        "mem_optimizer_region": mp.get("optimizer_state_region"),
    }


def extract_comm_plan_fields(cp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cp:
        return {}
    return {
        "comm_input_mode": cp.get("input_mode"),
        "comm_output_mode": cp.get("output_mode"),
        "comm_weight_mode": cp.get("weight_mode"),
        "comm_dma_required": cp.get("dma_required"),
    }


def run_case(
    base_cfg_path: Path,
    case: SweepCase,
    out_root: Path,
    keep_temp_configs: bool = False,
    keep_success_logs: bool = False,
) -> Dict[str, Any]:
    model_tag = sanitize_name(Path(case.model_path).stem)
    case_tag = f"{case.mode}__{model_tag}__{sanitize_name(case.precision_policy)}__{sanitize_name(case.parallel_policy)}"

    resolved_model = resolve_model_path(case.model_path)
    if not resolved_model.exists():
        return {
            "timestamp": utc_now(),
            "run_id": case_tag,
            "mode": case.mode,
            "model_name": model_tag,
            "model_path": case.model_path,
            "compile_returncode": -1,
            "compile_ok": False,
            "hls_ok": False,
            "error": f"Model file not found: {resolved_model}",
        }

    temp_dir = Path(tempfile.mkdtemp(prefix="fpgai_paper_", dir=str(out_root)))
    case_output_dir = out_root / case_tag
    project_name = case_tag

    def build_cfg(benchmark_enabled: bool, suffix: str) -> Tuple[Path, Path, Path]:
        cfg = load_yaml(base_cfg_path)
        cfg = apply_precision_policy(cfg, case.precision_policy)
        cfg = set_mode_specific_config(cfg, case.mode, benchmark_enabled=benchmark_enabled)
        cfg = set_parallel_policy(cfg, case.parallel_policy)
        cfg = set_project_and_model(cfg, str(resolved_model), case_output_dir, project_name)
        cfg = enforce_hls_flags(cfg, run_hls=True)

        cfg_path = temp_dir / f"{case_tag}{suffix}.yml"
        stdout_log = temp_dir / f"stdout{suffix}.log"
        stderr_log = temp_dir / f"stderr{suffix}.log"
        dump_yaml(cfg, cfg_path)
        return cfg_path, stdout_log, stderr_log

    cfg_compile, stdout_compile, stderr_compile = build_cfg(False, "__compile")

    cmd_compile = [sys.executable, str(MAIN_PY), "--config", str(cfg_compile)]
    with stdout_compile.open("w", encoding="utf-8") as so, stderr_compile.open("w", encoding="utf-8") as se:
        proc_compile = subprocess.run(
            cmd_compile,
            cwd=str(REPO_ROOT),
            stdout=so,
            stderr=se,
            text=True,
        )

    csynth_xml = find_csynth_xml(case_output_dir)
    hls_metrics = parse_csynth_xml(csynth_xml) if csynth_xml else {}
    compile_plan = parse_compile_plan(case_output_dir)
    train_res_est = parse_training_resource_estimate(case_output_dir)
    memory_plan = parse_memory_plan(case_output_dir)
    comm_plan = parse_comm_plan(case_output_dir)
    train_compare = parse_training_compare(case_output_dir)
    hls_stdout_log, hls_stderr_log = find_hls_log_paths(case_output_dir)

    compile_cfg_loaded = load_yaml(cfg_compile)
    defaults = get_nested(compile_cfg_loaded, ["numerics", "defaults"], {}) or {}
    act = defaults.get("activation", {}) or {}
    wgt = defaults.get("weight", {}) or {}
    bias = defaults.get("bias", {}) or {}
    accum = defaults.get("accum", {}) or {}

    row: Dict[str, Any] = {
        "timestamp": utc_now(),
        "run_id": case_tag,
        "mode": case.mode,
        "model_name": model_tag,
        "model_path": str(resolved_model),
        "project_name": project_name,
        "out_dir": str(case_output_dir),
        "precision_policy": case.precision_policy,
        "parallel_policy": case.parallel_policy,
        "compile_returncode": proc_compile.returncode,
        "compile_ok": proc_compile.returncode == 0,
        "stdout_log": str(stdout_compile),
        "stderr_log": str(stderr_compile),
        "csynth_report_path": str(csynth_xml) if csynth_xml else "",
        "compile_stdout_tail": tail_text_file(stdout_compile, 30),
        "compile_stderr_tail": tail_text_file(stderr_compile, 30),
        "hls_stdout_log": str(hls_stdout_log) if hls_stdout_log else "",
        "hls_stderr_log": str(hls_stderr_log) if hls_stderr_log else "",
        "hls_stdout_tail": tail_text_file(hls_stdout_log, 60) if hls_stdout_log else "",
        "hls_stderr_tail": tail_text_file(hls_stderr_log, 60) if hls_stderr_log else "",
        "activation_bits": act.get("total_bits"),
        "activation_int_bits": act.get("int_bits"),
        "weight_bits": wgt.get("total_bits"),
        "weight_int_bits": wgt.get("int_bits"),
        "bias_bits": bias.get("total_bits"),
        "bias_int_bits": bias.get("int_bits"),
        "accum_bits": accum.get("total_bits"),
        "accum_int_bits": accum.get("int_bits"),
    }

    row.update(extract_compile_plan_fields(compile_plan))
    row.update(extract_training_resource_estimate_fields(train_res_est))
    row.update(extract_memory_plan_fields(memory_plan))
    row.update(extract_comm_plan_fields(comm_plan))
    row.update(extract_training_compare_fields(train_compare))
    row.update(hls_metrics)

    # Prefer real HLS success, not just process return code
    row["hls_ok"] = bool(row.get("lut") is not None or row.get("latency_cycles_min") is not None)

    cyc = row.get("latency_cycles_min")
    clk_ns = row.get("estimated_clock_ns")
    clk_target_ns = row.get("target_clock_ns")
    compile_clock_mhz = maybe_float(row.get("compile_clock_mhz"))

    row["latency_ms_from_cycles_estclk"] = cyc * clk_ns * 1e-6 if cyc is not None and clk_ns is not None else None
    row["clock_mhz_from_estimated_clock"] = 1000.0 / clk_ns if clk_ns else None
    row["clock_mhz_from_target_clock"] = 1000.0 / clk_target_ns if clk_target_ns else None
    row["latency_ms_from_cycles_targetmhz"] = cyc / (compile_clock_mhz * 1e3) if cyc is not None and compile_clock_mhz else None
    row["throughput_fps_est"] = 1000.0 / row["latency_ms_from_cycles_estclk"] if row["latency_ms_from_cycles_estclk"] else None

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

    if row.get("pred_total_param_bytes") is not None and row.get("bram_18k") is not None:
        actual_bram_bytes = row["bram_18k"] * 2304
        row["delta_param_bytes_vs_bram18k_bytes"] = row["pred_total_param_bytes"] - actual_bram_bytes
    else:
        row["delta_param_bytes_vs_bram18k_bytes"] = None

    # Only run inference benchmark if compile returned success AND HLS really produced metrics
    if case.mode == "inference" and row["compile_ok"] and row["hls_ok"]:
        cfg_bench, stdout_bench, stderr_bench = build_cfg(True, "__bench")
        cmd_bench = [sys.executable, str(MAIN_PY), "--config", str(cfg_bench)]

        with stdout_bench.open("w", encoding="utf-8") as so, stderr_bench.open("w", encoding="utf-8") as se:
            proc_bench = subprocess.run(
                cmd_bench,
                cwd=str(REPO_ROOT),
                stdout=so,
                stderr=se,
                text=True,
            )

        bench_metrics = parse_bench_metrics(case_output_dir)
        quant_metrics = parse_quant_metrics(case_output_dir)

        row["bench_returncode"] = proc_bench.returncode
        row["bench_ok"] = proc_bench.returncode == 0
        row["bench_stdout_log"] = str(stdout_bench)
        row["bench_stderr_log"] = str(stderr_bench)
        row["bench_stdout_tail"] = tail_text_file(stdout_bench, 30)
        row["bench_stderr_tail"] = tail_text_file(stderr_bench, 30)

        row.update(extract_bench_fields(bench_metrics))
        row.update(extract_quant_fields(quant_metrics))

        if proc_bench.returncode == 0 and not keep_success_logs:
            stdout_bench.unlink(missing_ok=True)
            stderr_bench.unlink(missing_ok=True)

        if not keep_temp_configs:
            cfg_bench.unlink(missing_ok=True)
    else:
        row["bench_returncode"] = None
        row["bench_ok"] = None
        row["bench_stdout_log"] = ""
        row["bench_stderr_log"] = ""
        row["bench_stdout_tail"] = ""
        row["bench_stderr_tail"] = ""

    if proc_compile.returncode == 0 and not keep_success_logs:
        stdout_compile.unlink(missing_ok=True)
        stderr_compile.unlink(missing_ok=True)

    if not keep_temp_configs:
        cfg_compile.unlink(missing_ok=True)

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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def filter_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("mode") == mode]


def summarize_best_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r.get("mode")), str(r.get("model_name"))), []).append(r)

    summary: List[Dict[str, Any]] = []

    for (mode, model), grp in groups.items():
        passing = [x for x in grp if x.get("benchmark_passed") or x.get("quant_match")]

        def by_best_cos(x: Dict[str, Any]) -> float:
            vals = [
                v for v in [
                    maybe_float(x.get("bench_cosine")),
                    maybe_float(x.get("quant_cosine")),
                    maybe_float(x.get("train_grads_cosine")),
                ] if v is not None
            ]
            return max(vals) if vals else -1.0

        def by_latency(x: Dict[str, Any]) -> float:
            v = maybe_float(x.get("latency_ms_from_cycles_estclk"))
            return v if v is not None else float("inf")

        def by_lut(x: Dict[str, Any]) -> float:
            v = maybe_float(x.get("lut"))
            return v if v is not None else float("inf")

        if grp:
            best_acc = max(grp, key=by_best_cos)
            summary.append({
                "mode": mode,
                "model_name": model,
                "selection": "best_correctness",
                "run_id": best_acc.get("run_id"),
                "precision_policy": best_acc.get("precision_policy"),
                "parallel_policy": best_acc.get("parallel_policy"),
                "bench_cosine": best_acc.get("bench_cosine"),
                "quant_cosine": best_acc.get("quant_cosine"),
                "train_grads_cosine": best_acc.get("train_grads_cosine"),
                "latency_ms": best_acc.get("latency_ms_from_cycles_estclk"),
                "lut": best_acc.get("lut"),
                "ff": best_acc.get("ff"),
                "dsp": best_acc.get("dsp"),
                "bram_18k": best_acc.get("bram_18k"),
            })

        if passing:
            low_lat = min(passing, key=by_latency)
            summary.append({
                "mode": mode,
                "model_name": model,
                "selection": "lowest_latency_passing",
                "run_id": low_lat.get("run_id"),
                "precision_policy": low_lat.get("precision_policy"),
                "parallel_policy": low_lat.get("parallel_policy"),
                "bench_cosine": low_lat.get("bench_cosine"),
                "quant_cosine": low_lat.get("quant_cosine"),
                "latency_ms": low_lat.get("latency_ms_from_cycles_estclk"),
                "lut": low_lat.get("lut"),
                "ff": low_lat.get("ff"),
                "dsp": low_lat.get("dsp"),
                "bram_18k": low_lat.get("bram_18k"),
            })

            low_lut = min(passing, key=by_lut)
            summary.append({
                "mode": mode,
                "model_name": model,
                "selection": "lowest_lut_passing",
                "run_id": low_lut.get("run_id"),
                "precision_policy": low_lut.get("precision_policy"),
                "parallel_policy": low_lut.get("parallel_policy"),
                "bench_cosine": low_lut.get("bench_cosine"),
                "quant_cosine": low_lut.get("quant_cosine"),
                "latency_ms": low_lut.get("latency_ms_from_cycles_estclk"),
                "lut": low_lut.get("lut"),
                "ff": low_lut.get("ff"),
                "dsp": low_lut.get("dsp"),
                "bram_18k": low_lut.get("bram_18k"),
            })

    return summary


def write_predicted_vs_actual(rows: List[Dict[str, Any]], path: Path) -> None:
    compact: List[Dict[str, Any]] = []
    for r in rows:
        compact.append({
            "mode": r.get("mode"),
            "model_name": r.get("model_name"),
            "precision_policy": r.get("precision_policy"),
            "parallel_policy": r.get("parallel_policy"),
            "parallel_policy_effective": r.get("parallel_policy_effective"),
            "pred_total_param_bytes": r.get("pred_total_param_bytes"),
            "pred_activation_cache_bytes": r.get("pred_activation_cache_bytes"),
            "pred_gradient_bytes": r.get("pred_gradient_bytes"),
            "pred_optimizer_state_bytes": r.get("pred_optimizer_state_bytes"),
            "pred_forward_ops_proxy": r.get("pred_forward_ops_proxy"),
            "pred_backward_ops_proxy": r.get("pred_backward_ops_proxy"),
            "pred_update_ops_proxy": r.get("pred_update_ops_proxy"),
            "actual_lut": r.get("lut"),
            "actual_ff": r.get("ff"),
            "actual_dsp": r.get("dsp"),
            "actual_bram_18k": r.get("bram_18k"),
            "actual_uram": r.get("uram"),
            "actual_latency_cycles": r.get("latency_cycles_min"),
            "actual_latency_ms": r.get("latency_ms_from_cycles_estclk"),
            "clock_mhz_from_estimated_clock": r.get("clock_mhz_from_estimated_clock"),
        })
    write_csv(compact, path)


def render_compact_latex(rows: List[Dict[str, Any]], mode: str) -> str:
    rows = [r for r in rows if r.get("mode") == mode]
    lines = []
    lines.append(r"\begin{tabular}{ll l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Sweep & Policy & Pass & Latency (cycles) & LUT & FF & DSP & BRAM \\")
    lines.append(r"\midrule")

    for model in sorted(set(str(r.get("model_name")) for r in rows)):
        grp = [r for r in rows if r.get("model_name") == model]
        grp = sorted(grp, key=lambda x: (str(x.get("precision_policy")), str(x.get("parallel_policy"))))

        for idx, rr in enumerate(grp):
            passed = rr.get("benchmark_passed")
            if passed is None:
                passed = rr.get("quant_match")
            pass_txt = "Yes" if passed else "No"
            sweep_name = "Policy"
            policy_name = rr.get("parallel_policy") or rr.get("precision_policy")
            lines.append(
                f"{model if idx == 0 else ''} & {sweep_name} & {policy_name} & {pass_txt} & "
                f"{rr.get('latency_cycles_min','')} & {rr.get('lut','')} & {rr.get('ff','')} & "
                f"{rr.get('dsp','')} & {rr.get('bram_18k','')} \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def build_cases(
    infer_models: List[str],
    train_models: List[str],
    precisions: List[str],
    infer_policies: List[str],
    train_policies: List[str],
    modes: List[str],
) -> List[SweepCase]:
    cases: List[SweepCase] = []

    if "inference" in modes:
        for model in infer_models:
            for precision in precisions:
                for policy in infer_policies:
                    cases.append(SweepCase("inference", model, precision, policy))

    if "training" in modes:
        for model in train_models:
            for precision in precisions:
                for policy in train_policies:
                    cases.append(SweepCase("training", model, precision, policy))

    return cases


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-config", default="fpgai_infer.yml")
    ap.add_argument("--train-config", default="fpgai_train.yml")
    ap.add_argument("--out-dir", default="build/paper_experiments")
    ap.add_argument("--modes", nargs="*", default=["inference", "training"], choices=["inference", "training"])
    ap.add_argument("--infer-models", nargs="*", default=DEFAULT_INFER_MODELS)
    ap.add_argument("--train-models", nargs="*", default=DEFAULT_TRAIN_MODELS)
    ap.add_argument("--precisions", nargs="*", default=DEFAULT_PRECISIONS)
    ap.add_argument("--infer-policies", nargs="*", default=DEFAULT_POLICIES_INFER)
    ap.add_argument("--train-policies", nargs="*", default=DEFAULT_POLICIES_TRAIN)
    ap.add_argument("--keep-temp-configs", action="store_true")
    ap.add_argument("--keep-success-logs", action="store_true")
    ap.add_argument("--stop-on-first-error", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg = Path(args.infer_config).resolve()
    train_cfg = Path(args.train_config).resolve()

    if "inference" in args.modes and not infer_cfg.exists():
        raise FileNotFoundError(f"Inference config not found: {infer_cfg}")
    if "training" in args.modes and not train_cfg.exists():
        raise FileNotFoundError(f"Training config not found: {train_cfg}")
    if not MAIN_PY.exists():
        raise FileNotFoundError(f"main.py not found at: {MAIN_PY}")

    cases = build_cases(
        infer_models=args.infer_models,
        train_models=args.train_models,
        precisions=args.precisions,
        infer_policies=args.infer_policies,
        train_policies=args.train_policies,
        modes=args.modes,
    )

    print("=" * 80)
    print(f"[INFO] Running paper experiments: {len(cases)} cases")
    print(f"[INFO] Output root: {out_dir}")
    print("=" * 80)

    rows: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        print(
            f"[{i:03d}/{len(cases):03d}] "
            f"mode={case.mode} model={case.model_path} "
            f"precision={case.precision_policy} policy={case.parallel_policy}"
        )

        base_cfg = infer_cfg if case.mode == "inference" else train_cfg

        row = run_case(
            base_cfg_path=base_cfg,
            case=case,
            out_root=out_dir,
            keep_temp_configs=args.keep_temp_configs,
            keep_success_logs=args.keep_success_logs,
        )
        rows.append(row)

        print(
            "     "
            f"ret={row.get('compile_returncode')} "
            f"compile_ok={row.get('compile_ok')} "
            f"hls_ok={row.get('hls_ok')} "
            f"effective_policy={row.get('parallel_policy_effective')} "
            f"cycles={row.get('latency_cycles_min')} "
            f"LUT={row.get('lut')} DSP={row.get('dsp')} BRAM={row.get('bram_18k')}"
        )

        write_csv(rows, out_dir / "paper_experiments_master.csv")

        if args.stop_on_first_error and (not row.get("compile_ok", False) or not row.get("hls_ok", False)):
            print("[STOP] First compile/HLS error encountered.")
            break

    master_csv = out_dir / "paper_experiments_master.csv"
    write_csv(rows, master_csv)
    write_csv(filter_rows(rows, "inference"), out_dir / "paper_experiments_inference.csv")
    write_csv(filter_rows(rows, "training"), out_dir / "paper_experiments_training.csv")
    write_csv(summarize_best_rows(rows), out_dir / "paper_experiments_summary.csv")
    write_predicted_vs_actual(rows, out_dir / "predicted_vs_actual.csv")

    (out_dir / "latex_inference_table.tex").write_text(render_compact_latex(rows, "inference"), encoding="utf-8")
    (out_dir / "latex_training_table.tex").write_text(render_compact_latex(rows, "training"), encoding="utf-8")

    readme = textwrap.dedent(
        """
        Generated by scripts/run_paper_experiments.py

        Main files:
        - paper_experiments_master.csv
        - paper_experiments_inference.csv
        - paper_experiments_training.csv
        - paper_experiments_summary.csv
        - predicted_vs_actual.csv
        - latex_inference_table.tex
        - latex_training_table.tex

        Important:
        - benchmark runs only if compile_ok and hls_ok are both true
        - inspect hls_stdout_tail and hls_stderr_tail first when HLS fails
        - compare parallel_policy against parallel_policy_effective
        """
    ).strip()
    (out_dir / "README_RESULTS.txt").write_text(readme + "\n", encoding="utf-8")

    print("\n[OK] Done.")
    print(f"[OK] Master CSV: {master_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())