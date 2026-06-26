
"""End-to-end FPGAI audit runner.

This module is intentionally under fpgai.devtools instead of a loose script.
It creates a small truth-audit matrix that checks whether YAML hardware
settings materialize into compile plans, predictions, HLS/Vivado artifacts,
and validation artifacts where the local toolchain is available.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _repo_root() -> Path:
    return Path.cwd()


def _load_base(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_config_path(root: Path, requested: str, candidates: List[str]) -> Path:
    path = root / requested
    if path.exists():
        return path

    for candidate in candidates:
        candidate_path = root / candidate
        if candidate_path.exists():
            return candidate_path

    searched = [requested] + candidates
    raise FileNotFoundError(
        "Could not find config file. Searched:\n"
        + "\n".join(f"  - {item}" for item in searched)
    )


def _set_common_disabled_runtime(cfg: Dict[str, Any]) -> None:
    cfg.setdefault("project", {})
    cfg["project"]["clean"] = True

    cfg.setdefault("backends", {})
    cfg["backends"].setdefault("host_cpp", {})
    cfg["backends"]["host_cpp"]["enabled"] = True

    cfg["backends"].setdefault("hls", {})
    cfg["backends"]["hls"]["enabled"] = True

    cfg.setdefault("toolchain", {})
    cfg["toolchain"].setdefault("vivado", {})
    cfg["toolchain"]["vivado"]["enabled"] = False


def _configure_target(
    cfg: Dict[str, Any],
    *,
    out_dir: str,
    board: str,
    part: str,
    clock: Dict[str, Any],
    policy: str,
    knobs: Dict[str, Any],
    fit_policy: str = "report_only",
) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    cfg.setdefault("project", {})
    cfg["project"]["out_dir"] = out_dir
    cfg["project"]["clean"] = True

    cfg.setdefault("targets", {})
    cfg["targets"].setdefault("platform", {})
    cfg["targets"]["platform"]["board"] = board
    cfg["targets"]["platform"]["part"] = part
    cfg["targets"]["platform"]["clocks"] = [clock]
    cfg["targets"]["platform"]["fit_policy"] = fit_policy

    cfg.setdefault("optimization", {})
    cfg["optimization"]["parallel_policy"] = policy

    par = cfg["optimization"].setdefault("parallel", {})
    pipe = cfg["optimization"].setdefault("pipeline", {})
    tiling = cfg["optimization"].setdefault("tiling", {})

    for key in ("pe", "simd", "unroll_factor", "partition_factor", "array_partition_mode"):
        if key in knobs:
            par[key] = knobs[key]
    if "pipeline_style" in knobs:
        pipe["style"] = knobs["pipeline_style"]
    if "pipeline_ii" in knobs:
        pipe["ii"] = knobs["pipeline_ii"]
    if "dense_tile" in knobs:
        tiling["dense"] = knobs["dense_tile"]
    if "conv_tile" in knobs:
        tiling["conv"] = knobs["conv_tile"]

    cfg.setdefault("memory", {})
    if "weight_storage" in knobs:
        cfg["memory"]["weight_storage"] = knobs["weight_storage"]
    if "allow_double_buffer" in knobs:
        cfg["memory"]["allow_double_buffer"] = knobs["allow_double_buffer"]

    return cfg


def _cases(base_infer: Dict[str, Any], base_train: Dict[str, Any], root: Path) -> Dict[str, Dict[str, Any]]:
    out_root = root / "paper_experiments" / "full_pipeline_gate" / "sprint25_e2e_audit"

    safe_knobs = {
        "pe": 2,
        "simd": 2,
        "unroll_factor": 2,
        "partition_factor": 2,
        "array_partition_mode": "cyclic",
        "pipeline_style": "balanced",
        "pipeline_ii": 2,
        "dense_tile": {"tm": 8, "tn": 8, "tk": 8},
        "conv_tile": {"tm": 4, "tn": 4, "tr": 8, "tc": 8, "tk": 3},
        "weight_storage": "bram",
        "allow_double_buffer": False,
    }

    aggressive_knobs = {
        "pe": 8,
        "simd": 8,
        "unroll_factor": 8,
        "partition_factor": 8,
        "array_partition_mode": "complete",
        "pipeline_style": "aggressive",
        "pipeline_ii": 1,
        "dense_tile": {"tm": 32, "tn": 32, "tk": 32},
        "conv_tile": {"tm": 16, "tn": 16, "tr": 16, "tc": 16, "tk": 3},
        "weight_storage": "uram",
        "allow_double_buffer": True,
    }

    return {
        "inference_pynq_safe": _configure_target(
            base_infer,
            out_dir=str(out_root / "inference_pynq_safe"),
            board="pynq_z2",
            part="xc7z020clg400-1",
            clock={"name": "ap_clk"},
            policy="Balanced",
            knobs=safe_knobs,
            fit_policy="report_only",
        ),
        "inference_kv260_aggressive": _configure_target(
            base_infer,
            out_dir=str(out_root / "inference_kv260_aggressive"),
            board="kv260",
            part="xck26-sfvc784-2LV-c",
            clock={"name": "ap_clk"},
            policy="Latency-First",
            knobs=aggressive_knobs,
            fit_policy="report_only",
        ),
        "training_kv260_safe": _configure_target(
            base_train,
            out_dir=str(out_root / "training_kv260_safe"),
            board="kv260",
            part="xck26-sfvc784-2LV-c",
            clock={"name": "ap_clk"},
            policy="Balanced",
            knobs=safe_knobs,
            fit_policy="report_only",
        ),
        "training_kv260_aggressive": _configure_target(
            base_train,
            out_dir=str(out_root / "training_kv260_aggressive"),
            board="kv260",
            part="xck26-sfvc784-2LV-c",
            clock={"name": "ap_clk"},
            policy="Latency-First",
            knobs=aggressive_knobs,
            fit_policy="report_only",
        ),
    }


def _run(cmd: List[str], *, cwd: Path, log: Path) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(proc.stdout, encoding="utf-8")
    return {"cmd": cmd, "returncode": proc.returncode, "log": str(log)}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"failed to parse json: {exc}"}


def _artifact_presence(out_dir: Path) -> Dict[str, bool]:
    return {
        "manifest": (out_dir / "manifest.json").exists(),
        "compile_plan": (out_dir / "ir" / "compile_plan.json").exists(),
        "prediction_summary": (out_dir / "reports" / "prediction_summary.md").exists(),
        "resource_prediction": (out_dir / "reports" / "resource_prediction.json").exists(),
        "timing_prediction": (out_dir / "reports" / "timing_prediction.json").exists(),
        "board_fit": (out_dir / "reports" / "board_fit.json").exists(),
        "hardware_knob_contract": (out_dir / "reports" / "hardware_knob_contract.json").exists(),
        "hls_dir": (out_dir / "hls").exists(),
        "vivado_bridge_manifest": (out_dir / "vivado_bridge" / "manifest.json").exists(),
    }


def _contract_summary(out_dir: Path) -> Dict[str, Any]:
    contract = _read_json(out_dir / "reports" / "hardware_knob_contract.json")
    if not isinstance(contract, dict):
        return {"available": False}

    knobs = contract.get("knobs", [])
    important = [
        "optimization.parallel.pe",
        "optimization.parallel.simd",
        "optimization.parallel.unroll_factor",
        "optimization.parallel.partition_factor",
        "optimization.pipeline.style",
        "optimization.pipeline.ii",
        "optimization.tiling.dense",
        "optimization.tiling.conv",
        "memory.weight_storage",
        "memory.allow_double_buffer",
        "targets.platform.clocks.0.target_mhz",
        "targets.platform.fit_policy",
    ]

    rows = {}
    for path in important:
        row = next((x for x in knobs if x.get("path") == path), None)
        rows[path] = {
            "present": row is not None,
            "requested": None if row is None else row.get("requested"),
            "effective": None if row is None else row.get("effective"),
            "source": None if row is None else row.get("source"),
            "status": None if row is None else row.get("status"),
        }

    return {"available": True, "rows": rows}


def _case_summary(name: str, cfg_path: Path, out_dir: Path, compile_result: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _read_json(out_dir / "manifest.json")
    compile_plan = _read_json(out_dir / "ir" / "compile_plan.json")
    board_fit = _read_json(out_dir / "reports" / "board_fit.json")

    return {
        "name": name,
        "config": str(cfg_path),
        "out_dir": str(out_dir),
        "compile": compile_result,
        "artifacts": _artifact_presence(out_dir),
        "manifest_clock_requested": None
        if not isinstance(manifest, dict)
        else manifest.get("configuration", {}).get("requested", {}).get("clock_mhz"),
        "manifest_clock_effective": None
        if not isinstance(manifest, dict)
        else manifest.get("configuration", {}).get("effective", {}).get("clock_mhz"),
        "compile_plan_clock": None if not isinstance(compile_plan, dict) else compile_plan.get("clock_mhz"),
        "compile_plan_policy": None if not isinstance(compile_plan, dict) else compile_plan.get("policy"),
        "board_fit_format": None if not isinstance(board_fit, dict) else board_fit.get("format"),
        "board_fit_truth_boundary": None if not isinstance(board_fit, dict) else board_fit.get("truth_boundary"),
        "contract": _contract_summary(out_dir),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-base", default="configs/examples/inference_compile.yml")
    ap.add_argument("--training-base", default="configs/examples/training_compile.yml")
    ap.add_argument("--out-root", default="paper_experiments/full_pipeline_gate/sprint25_e2e_audit")
    args = ap.parse_args(argv)

    root = _repo_root()
    out_root = root / args.out_root
    configs_dir = out_root / "configs"
    logs_dir = out_root / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    inference_base_path = _resolve_config_path(
        root,
        args.inference_base,
        [
            "configs/examples/inference_compile.yml",
            "configs/examples/inference.yml",
            "fpgai_infer.yml",
        ],
    )
    training_base_path = _resolve_config_path(
        root,
        args.training_base,
        [
            "configs/examples/training_compile_smoke.yml",
            "configs/examples/training.yml",
            "configs/examples/train.yml",
            "configs/examples/training_accelerator.yml",
            "configs/examples/training_inference.yml",
            "fpgai_train.yml",
            "configs/sweeps/training_accelerator.yml",
            "configs/sweeps/sprint13a_training_accelerator.yml",
        ],
    )

    base_infer = _load_base(inference_base_path)
    base_train = _load_base(training_base_path)
    cases = _cases(base_infer, base_train, root)

    results = {
        "format": "fpgai.sprint25_e2e_audit.v1",
        "inference_base": str(inference_base_path),
        "training_base": str(training_base_path),
        "tool_availability": {
            "vitis_hls": shutil.which("vitis_hls"),
            "vivado": shutil.which("vivado"),
            "vitis": shutil.which("vitis"),
        },
        "cases": [],
    }

    for name, cfg in cases.items():
        cfg_path = configs_dir / f"{name}.yml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        out_dir = Path(cfg["project"]["out_dir"])
        log_path = logs_dir / f"{name}.compile.log"
        compile_result = _run(
            [sys.executable, "-m", "fpgai.cli", "compile", "--config", str(cfg_path)],
            cwd=root,
            log=log_path,
        )
        results["cases"].append(_case_summary(name, cfg_path, out_dir, compile_result))

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = out_root / "summary.md"
    lines = ["# Sprint 25 end-to-end audit", ""]
    lines.append("## Base configs")
    lines.append(f"- inference_base: `{results['inference_base']}`")
    lines.append(f"- training_base: `{results['training_base']}`")
    lines.append("")
    lines.append("## Tool availability")
    for k, v in results["tool_availability"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Cases")
    for c in results["cases"]:
        lines.append(f"### {c['name']}")
        lines.append(f"- compile returncode: `{c['compile']['returncode']}`")
        lines.append(f"- out_dir: `{c['out_dir']}`")
        lines.append(f"- requested clock: `{c['manifest_clock_requested']}`")
        lines.append(f"- effective clock: `{c['manifest_clock_effective']}`")
        lines.append(f"- compile plan clock: `{c['compile_plan_clock']}`")
        lines.append("- artifacts:")
        for k, v in c["artifacts"].items():
            lines.append(f"  - {k}: `{v}`")
        lines.append("- important knobs:")
        rows = c["contract"].get("rows", {}) if isinstance(c.get("contract"), dict) else {}
        for path, row in rows.items():
            lines.append(
                f"  - `{path}`: present={row.get('present')} requested=`{row.get('requested')}` "
                f"effective=`{row.get('effective')}` source=`{row.get('source')}` status=`{row.get('status')}`"
            )
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Wrote {summary_path}")
    print(f"[OK] Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
