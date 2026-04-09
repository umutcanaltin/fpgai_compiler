from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import xml.etree.ElementTree as ET

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml is required. Install with: pip install pyyaml") from exc


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        return int(float(x))
    except Exception:
        return None


def flatten_json(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}_{k}" if prefix else str(k)
            flatten_json(new_prefix, v, out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj


def ensure_nested(d: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    path = list(path)
    cur = d
    for p in path[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[path[-1]] = value


def ap_fixed_spec(total_bits: int, int_bits: int) -> Dict[str, Any]:
    return {
        "type": "ap_fixed",
        "total_bits": int(total_bits),
        "int_bits": int(int_bits),
    }


def precision_policy_to_numerics(policy_name: str) -> Dict[str, Dict[str, int]]:
    name = str(policy_name).strip()

    if name == "Uniform-12":
        return {
            "activation": ap_fixed_spec(12, 4),
            "weight": ap_fixed_spec(12, 4),
            "bias": ap_fixed_spec(20, 8),
            "accum": ap_fixed_spec(24, 10),
        }

    if name == "Uniform-16":
        return {
            "activation": ap_fixed_spec(16, 6),
            "weight": ap_fixed_spec(16, 6),
            "bias": ap_fixed_spec(24, 10),
            "accum": ap_fixed_spec(32, 12),
        }

    if name == "Mixed-Conservative":
        return {
            "activation": ap_fixed_spec(16, 6),
            "weight": ap_fixed_spec(12, 4),
            "bias": ap_fixed_spec(24, 10),
            "accum": ap_fixed_spec(32, 12),
        }

    raise ValueError(
        f"Unsupported precision policy '{policy_name}'. "
        f"Supported: Uniform-12, Uniform-16, Mixed-Conservative"
    )


def configure_training_yaml(
    template: Dict[str, Any],
    model_path: str,
    precision_policy: str,
    parallel_policy: str,
    out_dir: str,
    training_steps: Optional[int],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(template)

    numerics = precision_policy_to_numerics(precision_policy)

    # Required compiler keys
    ensure_nested(cfg, ["model", "path"], model_path)
    ensure_nested(cfg, ["pipeline", "mode"], "training_on_device")
    ensure_nested(cfg, ["project", "out_dir"], out_dir)
    ensure_nested(cfg, ["project", "clean"], True)

    # Helpful aliases
    ensure_nested(cfg, ["model_path"], model_path)
    ensure_nested(cfg, ["onnx_model"], model_path)
    ensure_nested(cfg, ["input_model"], model_path)
    ensure_nested(cfg, ["compiler", "model_path"], model_path)

    ensure_nested(cfg, ["mode"], "training_on_device")
    ensure_nested(cfg, ["compiler", "mode"], "training_on_device")

    ensure_nested(cfg, ["out_dir"], out_dir)
    ensure_nested(cfg, ["output_dir"], out_dir)
    ensure_nested(cfg, ["build_dir"], out_dir)
    ensure_nested(cfg, ["output", "dir"], out_dir)
    ensure_nested(cfg, ["compiler", "out_dir"], out_dir)
    ensure_nested(cfg, ["compiler", "output_dir"], out_dir)

    # Training enable
    ensure_nested(cfg, ["training", "enabled"], True)

    # Real planner policy key
    ensure_nested(cfg, ["optimization", "parallel_policy"], parallel_policy)
    ensure_nested(cfg, ["analysis", "design_space", "policy_name"], parallel_policy)

    # Paper bookkeeping
    ensure_nested(cfg, ["metadata", "paper_precision_policy"], precision_policy)
    ensure_nested(cfg, ["metadata", "paper_parallel_policy"], parallel_policy)

    # Numerics actually consumed by planner / codegen
    ensure_nested(cfg, ["numerics", "defaults", "activation"], numerics["activation"])
    ensure_nested(cfg, ["numerics", "defaults", "weight"], numerics["weight"])
    ensure_nested(cfg, ["numerics", "defaults", "bias"], numerics["bias"])
    ensure_nested(cfg, ["numerics", "defaults", "accum"], numerics["accum"])

    # Only schema-valid training roles
    ensure_nested(cfg, ["numerics", "training", "grad"], numerics["accum"])
    ensure_nested(cfg, ["numerics", "training", "grad_accum"], numerics["accum"])
    ensure_nested(cfg, ["numerics", "training", "master_weight"], numerics["weight"])
    ensure_nested(cfg, ["numerics", "training", "optimizer_state"], numerics["accum"])

    # Optional training steps
    if training_steps is not None:
        ensure_nested(cfg, ["training_steps"], training_steps)
        ensure_nested(cfg, ["training", "steps"], training_steps)
        ensure_nested(cfg, ["training", "num_steps"], training_steps)
        ensure_nested(cfg, ["training", "max_steps"], training_steps)

    if "benchmark" in cfg and isinstance(cfg["benchmark"], dict):
        cfg["benchmark"]["enabled"] = True

    return cfg


def newest_file(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def find_files(root: Path, contains_any: List[str], suffixes: Optional[List[str]] = None) -> List[Path]:
    if not root.exists():
        return []
    suffixes = suffixes or []
    results: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not any(token in name for token in contains_any):
            continue
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        results.append(p)
    return sorted(results)


def find_csynth_xml(out_dir: Path) -> Optional[Path]:
    return newest_file(list(out_dir.rglob("csynth.xml")))


def find_compile_plan(out_dir: Path) -> Optional[Path]:
    return newest_file(find_files(out_dir, ["compile_plan"], [".json"]))


def find_training_compare_json(out_dir: Path) -> Optional[Path]:
    files = find_files(out_dir, ["training_compare", "train_compare", "compare"], [".json"])
    preferred = [p for p in files if "train" in p.name.lower()]
    return newest_file(preferred or files)


def find_training_compare_txt(out_dir: Path) -> Optional[Path]:
    files = find_files(out_dir, ["training_compare", "train_compare", "compare"], [".txt"])
    preferred = [p for p in files if "train" in p.name.lower()]
    return newest_file(preferred or files)


def find_training_resource_json(out_dir: Path) -> Optional[Path]:
    files = find_files(
        out_dir,
        ["training_resource_estimate", "resource_estimate", "training_resources"],
        [".json"],
    )
    return newest_file(files)


def find_training_summary_txt(out_dir: Path) -> Optional[Path]:
    files = find_files(out_dir, ["training_summary", "summary"], [".txt"])
    preferred = [p for p in files if "train" in p.name.lower()]
    return newest_file(preferred or files)


def parse_json_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, Any] = {}
        flatten_json("", data, out)
        out["path"] = str(path)
        return out
    except Exception:
        return {}


def parse_compile_plan(path: Optional[Path]) -> Dict[str, Any]:
    raw = parse_json_file(path)
    return {f"plan_{k}": v for k, v in raw.items()}


def parse_training_compare_txt(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, Any] = {"training_compare_txt": str(path)}

    current_prefix: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if re.match(r"^\s*[A-Za-z0-9_./ -]+\s*:\s*$", line):
            indent = len(line) - len(line.lstrip())
            level = indent // 2
            section = line.strip()[:-1].strip().replace(" ", "_").replace("/", "_")
            current_prefix = current_prefix[:level]
            current_prefix.append(section)
            continue

        m = re.match(r"^\s*([A-Za-z0-9_./ -]+)\s*:\s*([-+eE0-9.]+|True|False)\s*$", line)
        if m:
            key = m.group(1).strip().replace(" ", "_").replace("/", "_")
            val = m.group(2).strip()
            full_key = "_".join(current_prefix + [key])
            if val in ("True", "False"):
                out[full_key] = (val == "True")
            else:
                out[full_key] = safe_float(val)

    return out


def parse_csynth_xml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}

    out: Dict[str, Any] = {"csynth_xml": str(path)}

    try:
        root = ET.parse(path).getroot()
    except Exception:
        return out

    tag_text: Dict[str, str] = {}
    for elem in root.iter():
        if elem.text and elem.text.strip():
            tag_text[elem.tag] = elem.text.strip()

    out["latency_cycles_min"] = safe_int(tag_text.get("Best-caseLatency"))
    out["latency_cycles_max"] = safe_int(tag_text.get("Worst-caseLatency"))
    out["ii"] = safe_int(tag_text.get("Interval-min")) or safe_int(tag_text.get("PipelineII"))

    out["dsp"] = safe_int(tag_text.get("DSP"))
    out["lut"] = safe_int(tag_text.get("LUT"))
    out["ff"] = safe_int(tag_text.get("FF"))
    out["bram_18k"] = safe_int(tag_text.get("BRAM_18K"))
    out["uram"] = safe_int(tag_text.get("URAM"))

    clock_ns = safe_float(tag_text.get("EstimatedClockPeriod")) or safe_float(tag_text.get("TargetClockPeriod"))
    out["estimated_clock_ns"] = clock_ns

    if clock_ns is not None:
        if out["latency_cycles_min"] is not None:
            out["latency_seconds_min"] = out["latency_cycles_min"] * clock_ns * 1e-9
        if out["latency_cycles_max"] is not None:
            out["latency_seconds_max"] = out["latency_cycles_max"] * clock_ns * 1e-9
        if out["ii"] not in (None, 0):
            out["throughput_fps_est"] = 1.0 / (out["ii"] * clock_ns * 1e-9)

    return out


def run_command(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def detect_out_dir_from_output(text: str) -> Optional[str]:
    patterns = [
        r"Out dir\s*:\s*(.+)",
        r"OUT_DIR\s*:\s*(.+)",
        r"out_dir\s*[:=]\s*(.+)",
    ]
    for line in text.splitlines():
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                return m.group(1).strip()
    return None


def make_case_name(model_path: str, precision: str, policy: str) -> str:
    return f"{Path(model_path).stem}__{precision.replace(' ', '_')}__{policy.replace(' ', '_')}"


def collect_case_row(
    out_dir: Path,
    model_path: str,
    precision: str,
    policy: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_path": model_path,
        "model_name": Path(model_path).stem,
        "precision_policy": precision,
        "parallel_policy": policy,
        "out_dir": str(out_dir),
        "compile_ok": (returncode == 0),
        "returncode": returncode,
    }

    csynth = parse_csynth_xml(find_csynth_xml(out_dir))
    row.update(csynth)
    row["hls_ok"] = bool(csynth)

    plan = parse_compile_plan(find_compile_plan(out_dir))
    row.update(plan)

    traincmp_json = parse_json_file(find_training_compare_json(out_dir))
    for k, v in traincmp_json.items():
        row[f"traincmp_{k}"] = v

    traincmp_txt = parse_training_compare_txt(find_training_compare_txt(out_dir))
    for k, v in traincmp_txt.items():
        row[f"traincmp_{k}"] = v

    trainres_json = parse_json_file(find_training_resource_json(out_dir))
    for k, v in trainres_json.items():
        row[f"trainres_{k}"] = v

    training_summary_txt = find_training_summary_txt(out_dir)
    if training_summary_txt:
        row["training_summary_txt"] = str(training_summary_txt)

    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"
    stdout_log.write_text(stdout, encoding="utf-8", errors="ignore")
    stderr_log.write_text(stderr, encoding="utf-8", errors="ignore")
    row["stdout_log"] = str(stdout_log)
    row["stderr_log"] = str(stderr_log)

    lat_s = safe_float(row.get("latency_seconds_max")) or safe_float(row.get("latency_seconds_min"))
    dsp = safe_float(row.get("dsp"))
    lut = safe_float(row.get("lut"))

    if lat_s is not None:
        row["latency_ms"] = lat_s * 1e3
    if dsp not in (None, 0) and lat_s is not None:
        row["latency_per_dsp_ms"] = (lat_s * 1e3) / dsp
    if lut not in (None, 0) and lat_s is not None:
        row["latency_per_lut_ms"] = (lat_s * 1e3) / lut

    return row


def append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: List[Dict[str, Any]] = []
    existing_fields: List[str] = []

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows.extend(reader)

    fieldnames = existing_fields[:]
    for k in row.keys():
        if k not in fieldnames:
            fieldnames.append(k)

    existing_rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Training-oriented FPGAI sweep runner")
    ap.add_argument("--config", required=True, help="Base training YAML, e.g. fpgai_train.yml")
    ap.add_argument("--models", nargs="+", required=True, help="Model paths")
    ap.add_argument("--precisions", nargs="+", required=True, help="Precision policies")
    ap.add_argument("--policies", nargs="+", required=True, help="Parallel policies")
    ap.add_argument("--csv-out", required=True, help="Output CSV path")
    ap.add_argument("--output-root", default="build/training_policy_sweeps", help="Per-case output root")
    ap.add_argument("--training-steps", type=int, default=None, help="Optional training step override")
    ap.add_argument("--python-bin", default=sys.executable, help="Python executable")
    ap.add_argument("--main-script", default="main.py", help="Entry script")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = Path(args.config).resolve()
    template = read_yaml(config_path)

    csv_out = Path(args.csv_out).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases: List[Tuple[str, str, str]] = []
    for model in args.models:
        for precision in args.precisions:
            for policy in args.policies:
                cases.append((model, precision, policy))

    print("=" * 72)
    print(f"[INFO] Running training sweep: {len(cases)} cases")
    print("=" * 72)

    for model_path, precision, policy in cases:
        case_name = make_case_name(model_path, precision, policy)
        case_out_dir = output_root / case_name

        if case_out_dir.exists():
            shutil.rmtree(case_out_dir)
        case_out_dir.mkdir(parents=True, exist_ok=True)

        cfg = configure_training_yaml(
            template=template,
            model_path=model_path,
            precision_policy=precision,
            parallel_policy=policy,
            out_dir=str(case_out_dir),
            training_steps=args.training_steps,
        )

        used_cfg_path = case_out_dir / "used_config.yml"
        write_yaml(used_cfg_path, cfg)

        cmd = [
            args.python_bin,
            args.main_script,
            "--config",
            str(used_cfg_path),
        ]

        print(f"[RUN] model={model_path} precision={precision} parallel={policy}")
        returncode, stdout, stderr = run_command(cmd, cwd=repo_root)

        detected = detect_out_dir_from_output(stdout) or detect_out_dir_from_output(stderr)
        if detected:
            detected_path = Path(detected)
            if detected_path.exists():
                case_out_dir = detected_path

        row = collect_case_row(
            out_dir=case_out_dir,
            model_path=model_path,
            precision=precision,
            policy=policy,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
        row["used_config"] = str(used_cfg_path)

        append_csv(csv_out, row)

        print(
            f"[DONE] returncode={returncode} "
            f"compile_ok={row.get('compile_ok')} "
            f"hls_ok={row.get('hls_ok')} "
            f"latency_ms={row.get('latency_ms')}"
        )

    print("=" * 72)
    print(f"[OK] CSV written to: {csv_out}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())