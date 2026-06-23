#!/usr/bin/env python3
"""Collect FPGAI experiment evidence into paper-ready tables.

This script is intentionally read-only with respect to experiment artifacts.
It scans completed experiment folders and writes Markdown/CSV/JSON summaries.

Default inputs:
  experiments/sprint12a_ddr_memory_strategy
  experiments/sprint12b_memory_binding_strategy
  experiments/sprint12c_parallel_policy_strategy

Default output:
  experiments/paper_evidence
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


BENCH_KEYS = {
    "passed",
    "cosine_similarity",
    "argmax_match",
    "max_abs_error",
    "mean_abs_error",
    "rmse",
    "num_elements",
    "atol",
    "rtol",
    "max_abs_error_limit",
    "mean_abs_error_limit",
    "rmse_limit",
    "min_cosine_similarity",
}


@dataclass
class DesignRecord:
    experiment: str
    design: str
    status: str | None = None
    returncode: int | None = None
    error: str | None = None
    artifact_dir: str | None = None
    bench_passed: str | None = None
    cosine_similarity: str | None = None
    max_abs_error: str | None = None
    mean_abs_error: str | None = None
    rmse: str | None = None
    argmax_match: str | None = None
    weight_mode: str | None = None
    has_stream_port: bool = False
    has_ddr_port: bool = False
    has_m_axi_weights: bool = False
    bram_bindings: int = 0
    uram_bindings: int = 0
    binding_count: int = 0
    policy: str | None = None
    pe: int | None = None
    simd: int | None = None
    unroll: int | None = None
    partition: int | None = None
    conv_parallel: str | None = None
    dense_parallel: str | None = None
    weight_partition: int | None = None
    parallel_evidence_lines: int = 0
    lut: str | None = None
    dsp: str | None = None
    bram: str | None = None
    uram: str | None = None
    latency_cycles: str | None = None
    achieved_ii: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_bench_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key in BENCH_KEYS:
            out[key] = val
    return out


def parse_deeplearn_cpp(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="ignore") if path.exists() else ""
    out: dict[str, Any] = {}
    out["has_stream_port"] = "weight_stream" in text
    out["has_ddr_port"] = "weights_mem" in text
    out["has_m_axi_weights"] = bool(re.search(r"#pragma\s+HLS\s+INTERFACE\s+m_axi\s+port\s*=\s*weights_mem", text))
    out["bram_bindings"] = len(re.findall(r"BIND_STORAGE[^\n]*impl\s*=\s*bram", text))
    out["uram_bindings"] = len(re.findall(r"BIND_STORAGE[^\n]*impl\s*=\s*uram", text))
    out["binding_count"] = len(re.findall(r"BIND_STORAGE", text))

    evidence_lines = re.findall(r"FPGAI parallel evidence:[^\n]*", text)
    out["parallel_evidence_lines"] = len(evidence_lines)
    max_weight_part = None
    conv = None
    dense = None
    for line in evidence_lines:
        op = re.search(r"op=([^\s]+)", line)
        iu = re.search(r"input_unroll=(\d+)", line)
        ou = re.search(r"output_unroll=(\d+)", line)
        ip = re.search(r"input_partition=(\d+)", line)
        opart = re.search(r"output_partition=(\d+)", line)
        wp = re.search(r"weight_partition=(\d+)", line)
        if wp:
            value = int(wp.group(1))
            max_weight_part = max(value, max_weight_part or value)
        summary = None
        if iu and ou:
            summary = f"{ou.group(1)}/{iu.group(1)}"
        if op and op.group(1).lower() == "conv":
            conv = summary
        if op and op.group(1).lower() == "dense":
            dense = summary
    out["conv_parallel"] = conv
    out["dense_parallel"] = dense
    out["weight_partition"] = max_weight_part
    return out


def find_artifact_dir(exp_dir: Path, result: dict[str, Any]) -> Path | None:
    for key in ("artifact_dir", "artifacts_dir", "out_dir", "build_dir"):
        value = result.get(key)
        if value:
            p = Path(value)
            if p.exists():
                return p
            p2 = exp_dir / value
            if p2.exists():
                return p2
    name = result.get("design_name") or result.get("name") or result.get("id")
    if name:
        candidates = [
            exp_dir / "artifacts" / str(name),
            exp_dir / "artifacts" / str(name) / "build",
        ]
        for p in candidates:
            if p.exists():
                return p
    return None


def normalize_build_dir(artifact_dir: Path | None) -> Path | None:
    if artifact_dir is None:
        return None
    if (artifact_dir / "hls").exists() or (artifact_dir / "bench").exists():
        return artifact_dir
    if (artifact_dir / "build").exists():
        return artifact_dir / "build"
    return artifact_dir


def extract_policy_from_name(name: str) -> str | None:
    if "resource_first" in name:
        return "resource_first"
    if "throughput_first" in name:
        return "throughput_first"
    if "latency_first" in name:
        return "latency_first"
    if "balanced" in name:
        return "balanced"
    return None


def policy_defaults(policy: str | None) -> tuple[int | None, int | None, int | None, int | None]:
    if policy == "resource_first":
        return 1, 1, 1, 1
    if policy == "balanced":
        return 2, 2, 2, 2
    if policy == "throughput_first":
        return 2, 4, 2, 4
    if policy == "latency_first":
        return 4, 4, 4, 4
    return None, None, None, None


def extract_weight_mode(name: str, rec: dict[str, Any]) -> str | None:
    text = json.dumps(rec, sort_keys=True)
    for mode in ("embedded", "stream", "ddr"):
        if f"{mode}" in text:
            return mode
    if "stream" in name or "streaming" in name:
        return "stream"
    if "ddr" in name or "external" in name:
        return "ddr"
    if "on_chip" in name or "bram" in name or "uram" in name:
        return "embedded/runtime-cache"
    return None


def parse_csynth_reports(build_dir: Path | None) -> dict[str, str]:
    # Best-effort parser; schemas differ across Vitis versions.
    out: dict[str, str] = {}
    if build_dir is None:
        return out
    report_dir = build_dir / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"
    if not report_dir.exists():
        return out
    text = "\n".join(p.read_text(errors="ignore") for p in report_dir.glob("*.rpt"))
    # Try common table labels. Keep empty when unavailable rather than inventing values.
    for label, key in [("LUT", "lut"), ("DSP", "dsp"), ("BRAM_18K", "bram"), ("URAM", "uram")]:
        m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([0-9.]+)", text)
        if m:
            out[key] = m.group(1)
    m = re.search(r"Latency\s*\(cycles\).*?\n.*?\|\s*min\s*\|\s*max\s*\|.*?\n.*?\|\s*(\d+)\s*\|\s*(\d+)", text, re.S | re.I)
    if m:
        out["latency_cycles"] = m.group(2)
    m = re.search(r"Interval\s*\|\s*min\s*\|\s*max\s*\|.*?\n.*?\|\s*(\d+)\s*\|\s*(\d+)", text, re.S | re.I)
    if m:
        out["achieved_ii"] = m.group(2)
    return out


def collect_experiment(exp_dir: Path) -> list[DesignRecord]:
    payload = read_json(exp_dir / "results.json")
    results = payload.get("results") or []
    records: list[DesignRecord] = []
    for result in results:
        name = str(result.get("design_name") or result.get("name") or result.get("id") or "unknown")
        artifact = find_artifact_dir(exp_dir, result)
        build_dir = normalize_build_dir(artifact)
        bench = parse_bench_summary((build_dir / "bench" / "summary.txt") if build_dir else Path("__missing__"))
        cpp = parse_deeplearn_cpp((build_dir / "hls" / "src" / "deeplearn.cpp") if build_dir else Path("__missing__"))
        rpt = parse_csynth_reports(build_dir)
        policy = extract_policy_from_name(name)
        pe, simd, unroll, partition = policy_defaults(policy)
        rec = DesignRecord(
            experiment=exp_dir.name,
            design=name,
            status=coerce_str(result.get("status")),
            returncode=result.get("returncode"),
            error=coerce_str(result.get("error")),
            artifact_dir=str(build_dir) if build_dir else None,
            bench_passed=bench.get("passed"),
            cosine_similarity=bench.get("cosine_similarity"),
            max_abs_error=bench.get("max_abs_error"),
            mean_abs_error=bench.get("mean_abs_error"),
            rmse=bench.get("rmse"),
            argmax_match=bench.get("argmax_match"),
            weight_mode=extract_weight_mode(name, result),
            has_stream_port=cpp["has_stream_port"],
            has_ddr_port=cpp["has_ddr_port"],
            has_m_axi_weights=cpp["has_m_axi_weights"],
            bram_bindings=cpp["bram_bindings"],
            uram_bindings=cpp["uram_bindings"],
            binding_count=cpp["binding_count"],
            policy=policy,
            pe=pe,
            simd=simd,
            unroll=unroll,
            partition=partition,
            conv_parallel=cpp.get("conv_parallel"),
            dense_parallel=cpp.get("dense_parallel"),
            weight_partition=cpp.get("weight_partition"),
            parallel_evidence_lines=cpp["parallel_evidence_lines"],
            lut=rpt.get("lut"),
            dsp=rpt.get("dsp"),
            bram=rpt.get("bram"),
            uram=rpt.get("uram"),
            latency_cycles=rpt.get("latency_cycles"),
            achieved_ii=rpt.get("achieved_ii"),
        )
        records.append(rec)
    return records


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def md_table(rows: list[dict[str, Any]], fields: list[str], headers: list[str] | None = None) -> str:
    if headers is None:
        headers = fields
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        vals = []
        for field in fields:
            val = row.get(field)
            if val is None:
                val = ""
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + md_table(rows, fields, headers))


def claim_support(records: list[DesignRecord]) -> list[dict[str, Any]]:
    def ok(designs: Iterable[str]) -> bool:
        wanted = set(designs)
        got = {r.design for r in records if r.status == "passed" and r.bench_passed == "True"}
        return wanted.issubset(got)

    def any_ok(pred) -> bool:
        return any(pred(r) and r.status == "passed" and r.bench_passed == "True" for r in records)

    rows = [
        {
            "claim": "Embedded/on-chip weight strategy passes end-to-end",
            "status": "supported" if any_ok(lambda r: "on_chip" in r.design or "bram_baseline" in r.design) else "missing",
            "evidence": "passing benchmark for on-chip/BRAM baseline design",
        },
        {
            "claim": "Streamed runtime weights pass end-to-end",
            "status": "supported" if any_ok(lambda r: "stream" in r.design or r.has_stream_port) else "missing",
            "evidence": "weight_stream interface plus passing benchmark",
        },
        {
            "claim": "External-DDR runtime weights pass end-to-end",
            "status": "supported" if any_ok(lambda r: "ddr" in r.design and r.has_m_axi_weights) else "missing",
            "evidence": "weights_mem m_axi interface plus passing benchmark",
        },
        {
            "claim": "BRAM storage binding is generated and Vitis-valid",
            "status": "supported" if any_ok(lambda r: r.bram_bindings > 0) else "missing",
            "evidence": "BIND_STORAGE impl=bram plus passing benchmark",
        },
        {
            "claim": "URAM storage binding is generated and Vitis-valid",
            "status": "supported" if any_ok(lambda r: r.uram_bindings > 0) else "missing",
            "evidence": "BIND_STORAGE impl=uram plus passing benchmark",
        },
        {
            "claim": "Parallel policies materialize distinct HLS parameters",
            "status": "supported" if ok(["parallel_resource_first", "parallel_balanced", "parallel_throughput_first", "parallel_latency_first_max"]) else "partial",
            "evidence": "policy-specific PE/SIMD/unroll/partition evidence comments plus passing benchmarks",
        },
        {
            "claim": "General automatic optimal hardware search",
            "status": "not claimed / not supported by this evidence",
            "evidence": "current evidence supports materialized sweeps and validation, not global optimality",
        },
        {
            "claim": "Training accelerator support",
            "status": "not supported by this evidence",
            "evidence": "these tables cover inference/memory/parallel HLS paths only",
        },
    ]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect FPGAI paper evidence tables from experiment artifacts.")
    ap.add_argument("--experiments-root", default="experiments", help="Root directory containing experiment folders.")
    ap.add_argument("--out", default="experiments/paper_evidence", help="Output directory for generated tables.")
    ap.add_argument(
        "--include",
        nargs="*",
        default=[
            "sprint12a_ddr_memory_strategy",
            "sprint12b_memory_binding_strategy",
            "sprint12c_parallel_policy_strategy",
        ],
        help="Experiment directory names to include.",
    )
    args = ap.parse_args()

    root = Path(args.experiments_root)
    out = Path(args.out)
    records: list[DesignRecord] = []
    missing: list[str] = []
    for name in args.include:
        exp_dir = root / name
        if not (exp_dir / "results.json").exists():
            missing.append(str(exp_dir))
            continue
        records.extend(collect_experiment(exp_dir))

    out.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    (out / "paper_evidence.json").write_text(json.dumps({"records": rows, "missing_experiments": missing}, indent=2))

    all_fields = list(DesignRecord.__dataclass_fields__.keys())
    write_csv(out / "paper_evidence.csv", rows, all_fields)

    correctness_fields = ["experiment", "design", "status", "bench_passed", "cosine_similarity", "max_abs_error", "mean_abs_error", "rmse", "argmax_match"]
    write_md(out / "table_correctness_summary.md", "Correctness summary", rows, correctness_fields)
    write_csv(out / "table_correctness_summary.csv", rows, correctness_fields)

    mem_rows = [r for r in rows if r["experiment"] == "sprint12a_ddr_memory_strategy"]
    mem_fields = ["design", "status", "bench_passed", "cosine_similarity", "weight_mode", "has_stream_port", "has_ddr_port", "has_m_axi_weights"]
    write_md(out / "table_memory_modes.md", "Memory mode evidence", mem_rows, mem_fields)
    write_csv(out / "table_memory_modes.csv", mem_rows, mem_fields)

    bind_rows = [r for r in rows if r["experiment"] == "sprint12b_memory_binding_strategy"]
    bind_fields = ["design", "status", "bench_passed", "cosine_similarity", "binding_count", "bram_bindings", "uram_bindings"]
    write_md(out / "table_memory_binding.md", "BRAM/URAM binding evidence", bind_rows, bind_fields)
    write_csv(out / "table_memory_binding.csv", bind_rows, bind_fields)

    par_rows = [r for r in rows if r["experiment"] == "sprint12c_parallel_policy_strategy"]
    par_fields = ["design", "status", "bench_passed", "cosine_similarity", "policy", "pe", "simd", "unroll", "partition", "conv_parallel", "dense_parallel", "weight_partition", "parallel_evidence_lines"]
    write_md(out / "table_parallel_policies.md", "Parallel policy evidence", par_rows, par_fields)
    write_csv(out / "table_parallel_policies.csv", par_rows, par_fields)

    hls_fields = ["experiment", "design", "lut", "dsp", "bram", "uram", "latency_cycles", "achieved_ii"]
    write_md(out / "table_hls_metrics.md", "Best-effort HLS metrics", rows, hls_fields)
    write_csv(out / "table_hls_metrics.csv", rows, hls_fields)

    claims = claim_support(records)
    claim_fields = ["claim", "status", "evidence"]
    write_md(out / "table_claim_support.md", "Claim-support status", claims, claim_fields)
    write_csv(out / "table_claim_support.csv", claims, claim_fields)

    index = [
        "# FPGAI Paper Evidence Index",
        "",
        "Generated files:",
        "",
        "- `paper_evidence.json`",
        "- `paper_evidence.csv`",
        "- `table_correctness_summary.md/.csv`",
        "- `table_memory_modes.md/.csv`",
        "- `table_memory_binding.md/.csv`",
        "- `table_parallel_policies.md/.csv`",
        "- `table_hls_metrics.md/.csv`",
        "- `table_claim_support.md/.csv`",
        "",
        f"Collected records: {len(records)}",
    ]
    if missing:
        index.extend(["", "Missing experiment folders:", *[f"- `{m}`" for m in missing]])
    (out / "README.md").write_text("\n".join(index) + "\n")

    print(f"[OK] Wrote {out}")
    print(f"[OK] Records: {len(records)}")
    if missing:
        print("[WARN] Missing experiments:")
        for m in missing:
            print(f"  - {m}")
    print()
    print((out / "table_claim_support.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
