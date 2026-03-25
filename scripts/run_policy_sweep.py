from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


PRECISION_POLICIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Uniform-8": {
        "activation": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
        "weight": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
        "bias": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "accum": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
    },
    "Uniform-12": {
        "activation": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
        "weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
        "bias": {"type": "ap_fixed", "total_bits": 20, "int_bits": 8},
        "accum": {"type": "ap_fixed", "total_bits": 20, "int_bits": 8},
    },
    "Uniform-16": {
        "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "bias": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
        "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
    },
    "Mixed-Conservative": {
        "activation": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
        "weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
        "bias": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
        "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
    },
    "Mixed-Aggressive": {
        "activation": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
        "weight": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
        "bias": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "accum": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
    },
}

PARALLEL_POLICIES: Dict[str, Dict[str, Any]] = {
    "Resource-First": {
        "pe": 1,
        "simd": 1,
        "unroll_factor": 1,
        "partition_factor": 1,
        "pipeline_style": "conservative",
        "target_clock_mhz": None,
    },
    "Balanced": {
        "pe": 2,
        "simd": 2,
        "unroll_factor": 2,
        "partition_factor": 1,
        "pipeline_style": "balanced",
        "target_clock_mhz": None,
    },
    "Latency-First": {
        "pe": 4,
        "simd": 4,
        "unroll_factor": 4,
        "partition_factor": 1,
        "pipeline_style": "aggressive",
        "target_clock_mhz": None,
    },
}

DEFAULT_MODELS = [
    "models/suite/mlp_mnist.onnx",
    "models/cnn_mnist.onnx",
]

CSV_COLUMNS = [
    "timestamp",
    "run_id",
    "model_name",
    "model_path",
    "project_name",
    "out_dir",
    "precision_policy",
    "parallel_policy",
    "activation_bits",
    "activation_int_bits",
    "weight_bits",
    "weight_int_bits",
    "bias_bits",
    "bias_int_bits",
    "accum_bits",
    "accum_int_bits",
    "pe",
    "simd",
    "unroll_factor",
    "partition_factor",
    "pipeline_style",
    "target_clock_mhz",
    "compile_returncode",
    "compile_ok",
    "benchmark_passed",
    "hls_ok",
    "max_abs_error",
    "mean_abs_error",
    "rmse",
    "cosine_similarity",
    "argmax_match",
    "first_bad_layer",
    "latency_cycles_min",
    "latency_cycles_max",
    "latency_seconds_min",
    "latency_seconds_max",
    "ii",
    "estimated_clock_ns",
    "estimated_clock_uncertainty_ns",
    "lut",
    "ff",
    "dsp",
    "bram_18k",
    "uram",
    "csynth_report_path",
    "hls_metrics_json",
    "metrics_json",
    "benchmark_manifest_json",
    "compile_plan_json",
    "manifest_json",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_set(root: Dict[str, Any], dotted_path: str, value: Any) -> None:
    cur = root
    keys = dotted_path.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def deep_get(root: Dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cur: Any = root
    for k in dotted_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def model_name_from_path(model_path: str) -> str:
    return Path(model_path).stem


def ensure_list(x: Optional[Iterable[str]]) -> List[str]:
    if x is None:
        return []
    return list(x)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return None
        return float(s)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return None
        return int(float(s))
    except Exception:
        return None


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_bad_report_name(name: str) -> bool:
    n = name.lower()
    return "design_size" in n


def find_hls_report(out_dir: Path) -> Optional[Path]:
    report_dir = out_dir / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"

    preferred = [
        report_dir / "csynth.xml",
        report_dir / "deeplearn_csynth.xml",
        report_dir / "csynth.rpt",
        report_dir / "deeplearn_csynth.rpt",
    ]
    for p in preferred:
        if p.exists() and not _is_bad_report_name(p.name):
            return p

    hls_dir = out_dir / "hls"
    if not hls_dir.exists():
        return None

    candidates = []
    for p in hls_dir.rglob("*csynth*.xml"):
        if not _is_bad_report_name(p.name):
            candidates.append(p)
    for p in hls_dir.rglob("*csynth*.rpt"):
        if not _is_bad_report_name(p.name):
            candidates.append(p)

    if not candidates:
        return None

    def rank(p: Path) -> Tuple[int, int]:
        name_rank = {
            "csynth.xml": 0,
            "deeplearn_csynth.xml": 1,
            "csynth.rpt": 2,
            "deeplearn_csynth.rpt": 3,
        }.get(p.name, 10)
        return (name_rank, len(str(p)))

    candidates.sort(key=rank)
    return candidates[0]


def parse_hls_report(report_path: Optional[Path], clock_mhz: Optional[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "latency_cycles_min": None,
        "latency_cycles_max": None,
        "latency_seconds_min": None,
        "latency_seconds_max": None,
        "ii": None,
        "estimated_clock_ns": None,
        "estimated_clock_uncertainty_ns": None,
        "lut": None,
        "ff": None,
        "dsp": None,
        "bram_18k": None,
        "uram": None,
    }

    if report_path is None or not report_path.exists():
        return out

    if report_path.suffix.lower() == ".xml":
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()

            def find_text(tag_name: str) -> Optional[str]:
                elem = root.find(f".//{tag_name}")
                if elem is not None and elem.text is not None:
                    s = elem.text.strip()
                    return s if s != "" else None
                return None

            out["latency_cycles_min"] = safe_int(find_text("Best-caseLatency"))
            out["latency_cycles_max"] = safe_int(find_text("Worst-caseLatency"))
            out["ii"] = safe_int(find_text("Interval-min"))
            out["estimated_clock_ns"] = safe_float(find_text("EstimatedClockPeriod"))
            out["estimated_clock_uncertainty_ns"] = safe_float(find_text("ClockUncertainty"))
            out["lut"] = safe_int(find_text("LUT"))
            out["ff"] = safe_int(find_text("FF"))
            out["dsp"] = safe_int(find_text("DSP"))
            out["bram_18k"] = safe_int(find_text("BRAM_18K"))
            out["uram"] = safe_int(find_text("URAM"))
        except Exception as e:
            print(f"[WARN] Failed to parse XML report {report_path}: {e}")
            return out
    else:
        text = report_path.read_text(encoding="utf-8", errors="ignore")

        def rx(pattern: str) -> Optional[str]:
            m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else None

        out["latency_cycles_min"] = safe_int(rx(r"Latency.*?min.*?([0-9][0-9,]*)"))
        out["latency_cycles_max"] = safe_int(rx(r"Latency.*?max.*?([0-9][0-9,]*)"))
        out["ii"] = safe_int(rx(r"(?:Initiation\s*Interval|Interval).*?([0-9][0-9,]*)"))
        out["estimated_clock_ns"] = safe_float(rx(r"Estimated\s*Clock\s*Period.*?([0-9]+(?:\.[0-9]+)?)"))
        out["estimated_clock_uncertainty_ns"] = safe_float(rx(r"Clock\s*Uncertainty.*?([0-9]+(?:\.[0-9]+)?)"))
        out["lut"] = safe_int(rx(r"\bLUT\b.*?([0-9][0-9,]*)"))
        out["ff"] = safe_int(rx(r"\bFF\b.*?([0-9][0-9,]*)"))
        out["dsp"] = safe_int(rx(r"\bDSP\b.*?([0-9][0-9,]*)"))
        out["bram_18k"] = safe_int(rx(r"\bBRAM(?:_18K)?\b.*?([0-9][0-9,]*)"))
        out["uram"] = safe_int(rx(r"\bURAM\b.*?([0-9][0-9,]*)"))

    clk_ns = out["estimated_clock_ns"]
    if clk_ns is None and clock_mhz:
        clk_ns = 1000.0 / float(clock_mhz)

    if clk_ns is not None:
        if out["latency_cycles_min"] is not None:
            out["latency_seconds_min"] = out["latency_cycles_min"] * clk_ns * 1e-9
        if out["latency_cycles_max"] is not None:
            out["latency_seconds_max"] = out["latency_cycles_max"] * clk_ns * 1e-9

    return out


def apply_precision_policy(cfg: Dict[str, Any], policy_name: str) -> None:
    pol = deepcopy(PRECISION_POLICIES[policy_name])
    deep_set(cfg, "numerics.defaults.activation", pol["activation"])
    deep_set(cfg, "numerics.defaults.weight", pol["weight"])
    deep_set(cfg, "numerics.defaults.bias", pol["bias"])
    deep_set(cfg, "numerics.defaults.accum", pol["accum"])
    deep_set(cfg, "optimization.precision_policy", policy_name)


def apply_parallel_policy(cfg: Dict[str, Any], policy_name: str) -> None:
    pol = deepcopy(PARALLEL_POLICIES[policy_name])
    deep_set(cfg, "optimization.parallel_policy", policy_name)
    deep_set(cfg, "optimization.parallel.pe", pol["pe"])
    deep_set(cfg, "optimization.parallel.simd", pol["simd"])
    deep_set(cfg, "optimization.parallel.unroll_factor", pol["unroll_factor"])
    deep_set(cfg, "optimization.parallel.partition_factor", pol["partition_factor"])
    deep_set(cfg, "optimization.parallel.pipeline_style", pol["pipeline_style"])
    deep_set(cfg, "backends.hls.policy.parallel", pol)
    if pol.get("target_clock_mhz") is not None:
        deep_set(cfg, "targets.platform.clocks.0.target_mhz", pol["target_clock_mhz"])


def apply_model_override(cfg: Dict[str, Any], model_path: str) -> None:
    deep_set(cfg, "model.path", model_path)


def make_project_identity(model_path: str, precision_policy: str, parallel_policy: str) -> Tuple[str, str]:
    model_name = model_name_from_path(model_path)
    run_name = sanitize_name(f"{model_name}__{precision_policy}__{parallel_policy}")
    out_dir = str(Path("build") / "policy_sweeps" / run_name)
    return run_name, out_dir


def prepare_cfg(
    base_cfg: Dict[str, Any],
    model_path: str,
    precision_policy: str,
    parallel_policy: str,
) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    apply_model_override(cfg, model_path)
    apply_precision_policy(cfg, precision_policy)
    apply_parallel_policy(cfg, parallel_policy)

    project_name, out_dir = make_project_identity(model_path, precision_policy, parallel_policy)
    deep_set(cfg, "project.name", project_name)
    deep_set(cfg, "project.out_dir", out_dir)
    deep_set(cfg, "project.clean", True)

    return cfg


def run_one(repo_root: Path, python_exe: str, config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [python_exe, "main.py", "--config", str(config_path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )


def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def collect_metrics(out_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    metrics_json = out_dir / "bench" / "metrics.json"
    benchmark_manifest = out_dir / "bench" / "benchmark_manifest.json"
    compile_plan_json = out_dir / "ir" / "compile_plan.json"
    manifest_json = out_dir / "manifest.json"

    row["metrics_json"] = str(metrics_json.resolve()) if metrics_json.exists() else ""
    row["benchmark_manifest_json"] = str(benchmark_manifest.resolve()) if benchmark_manifest.exists() else ""
    row["compile_plan_json"] = str(compile_plan_json.resolve()) if compile_plan_json.exists() else ""
    row["manifest_json"] = str(manifest_json.resolve()) if manifest_json.exists() else ""

    metrics = read_json_if_exists(metrics_json) or {}
    benchmark_manifest_obj = read_json_if_exists(benchmark_manifest) or {}
    compile_plan = read_json_if_exists(compile_plan_json) or {}
    root_manifest = read_json_if_exists(manifest_json) or {}

    row["benchmark_passed"] = metrics.get("passed")
    row["max_abs_error"] = metrics.get("max_abs_error")
    row["mean_abs_error"] = metrics.get("mean_abs_error")
    row["rmse"] = metrics.get("rmse")
    row["cosine_similarity"] = metrics.get("cosine_similarity")
    row["argmax_match"] = metrics.get("argmax_match")

    bm = benchmark_manifest_obj.get("benchmark", {}) if isinstance(benchmark_manifest_obj, dict) else {}
    row["first_bad_layer"] = bm.get("intermediate_first_bad_layer")

    report_path = find_hls_report(out_dir)

    if report_path is None:
        report_path_str = bm.get("hls_csynth_report")
        if report_path_str:
            rp = Path(report_path_str)
            if rp.exists() and not _is_bad_report_name(rp.name):
                report_path = rp

    row["csynth_report_path"] = str(report_path.resolve()) if report_path and report_path.exists() else ""
    row["hls_ok"] = bool(report_path and report_path.exists())

    target_clock_mhz = deep_get(root_manifest, "targets.platform.clocks.0.target_mhz", None)
    if target_clock_mhz is None:
        target_clock_mhz = deep_get(compile_plan, "target_clock_mhz", None)
    target_clock_mhz = safe_float(target_clock_mhz)

    hls = parse_hls_report(report_path, target_clock_mhz)
    row.update(hls)

    hls_metrics_json = out_dir / "bench" / "hls_metrics.json"
    hls_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with hls_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "report_path": str(report_path.resolve()) if report_path else None,
                "target_clock_mhz": target_clock_mhz,
                "parsed_metrics": hls,
            },
            f,
            indent=2,
        )
    row["hls_metrics_json"] = str(hls_metrics_json.resolve())

    print(f"[DEBUG] report_path={report_path}")
    print(f"[DEBUG] hls_metrics={hls}")

    return row


def main() -> None:
    parser = argparse.ArgumentParser("FPGAI policy sweep runner")
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--precision-policies",
        nargs="*",
        default=list(PRECISION_POLICIES.keys()),
        choices=list(PRECISION_POLICIES.keys()),
    )
    parser.add_argument(
        "--parallel-policies",
        nargs="*",
        default=list(PARALLEL_POLICIES.keys()),
        choices=list(PARALLEL_POLICIES.keys()),
    )
    parser.add_argument("--csv-out", default="build/policy_sweeps/policy_sweep_results.csv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-temp-configs", action="store_true")
    parser.add_argument("--keep-builds", action="store_true")
    parser.add_argument("--keep-success-logs", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_cfg_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    csv_out = (repo_root / args.csv_out).resolve() if not Path(args.csv_out).is_absolute() else Path(args.csv_out)

    base_cfg = load_yaml(base_cfg_path)
    models = ensure_list(args.models) or DEFAULT_MODELS

    sweep_dir = csv_out.parent
    sweep_dir.mkdir(parents=True, exist_ok=True)

    for model_path in models:
        for precision_policy in args.precision_policies:
            for parallel_policy in args.parallel_policies:
                tstamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                run_id = sanitize_name(f"{model_name_from_path(model_path)}__{precision_policy}__{parallel_policy}")

                cfg = prepare_cfg(
                    base_cfg=base_cfg,
                    model_path=model_path,
                    precision_policy=precision_policy,
                    parallel_policy=parallel_policy,
                )

                out_dir = Path(deep_get(cfg, "project.out_dir")).resolve()
                project_name = str(deep_get(cfg, "project.name"))
                clock_mhz = deep_get(cfg, "targets.platform.clocks.0.target_mhz", None)

                tmp_dir = Path(tempfile.mkdtemp(prefix="fpgai_policy_", dir=str(sweep_dir)))
                tmp_cfg_path = tmp_dir / f"{run_id}.yml"
                dump_yaml(tmp_cfg_path, cfg)

                print("=" * 72)
                print(f"[RUN] model={model_path} precision={precision_policy} parallel={parallel_policy}")
                print(f"[CFG] {tmp_cfg_path}")
                print(f"[OUT] {out_dir}")

                proc = run_one(repo_root=repo_root, python_exe=args.python, config_path=tmp_cfg_path)

                stdout_path = tmp_dir / "stdout.log"
                stderr_path = tmp_dir / "stderr.log"
                if proc.returncode != 0 or args.keep_success_logs:
                    try:
                        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="ignore")
                        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="ignore")
                    except OSError as e:
                        print(f"[WARN] Could not write logs: {e}")
                        stdout_path = None
                        stderr_path = None
                else:
                    stdout_path = None
                    stderr_path = None

                row: Dict[str, Any] = {k: "" for k in CSV_COLUMNS}
                row["timestamp"] = tstamp
                row["run_id"] = run_id
                row["model_name"] = model_name_from_path(model_path)
                row["model_path"] = model_path
                row["project_name"] = project_name
                row["out_dir"] = str(out_dir)
                row["precision_policy"] = precision_policy
                row["parallel_policy"] = parallel_policy
                row["compile_returncode"] = proc.returncode
                row["compile_ok"] = (proc.returncode == 0)
                row["target_clock_mhz"] = clock_mhz

                pol = PRECISION_POLICIES[precision_policy]
                row["activation_bits"] = pol["activation"]["total_bits"]
                row["activation_int_bits"] = pol["activation"]["int_bits"]
                row["weight_bits"] = pol["weight"]["total_bits"]
                row["weight_int_bits"] = pol["weight"]["int_bits"]
                row["bias_bits"] = pol["bias"]["total_bits"]
                row["bias_int_bits"] = pol["bias"]["int_bits"]
                row["accum_bits"] = pol["accum"]["total_bits"]
                row["accum_int_bits"] = pol["accum"]["int_bits"]

                ppol = PARALLEL_POLICIES[parallel_policy]
                row["pe"] = ppol["pe"]
                row["simd"] = ppol["simd"]
                row["unroll_factor"] = ppol["unroll_factor"]
                row["partition_factor"] = ppol["partition_factor"]
                row["pipeline_style"] = ppol["pipeline_style"]

                if out_dir.exists():
                    row.update(collect_metrics(out_dir))
                else:
                    row["hls_ok"] = False

                append_csv_row(csv_out, row)

                print(f"[DONE] returncode={proc.returncode} csv={csv_out}")
                if proc.returncode != 0:
                    if stdout_path is not None:
                        print(f"[WARN] stdout: {stdout_path}")
                    if stderr_path is not None:
                        print(f"[WARN] stderr: {stderr_path}")

                if proc.returncode == 0 and not args.keep_builds:
                    try:
                        if out_dir.exists():
                            shutil.rmtree(out_dir, ignore_errors=True)
                    except OSError as e:
                        print(f"[WARN] Could not remove build dir {out_dir}: {e}")

                if not args.keep_temp_configs:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("")
    print(f"[OK] Sweep completed: {csv_out}")


if __name__ == "__main__":
    main()