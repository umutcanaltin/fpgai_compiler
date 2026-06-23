#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_summary(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value in {"True", "False"}:
            out[key] = value == "True"
            continue
        try:
            if any(ch in value for ch in [".", "e", "E"]):
                out[key] = float(value)
            else:
                out[key] = int(value)
            continue
        except Exception:
            out[key] = value
    return out


def _parse_materialized_metadata(config_path: Path) -> Dict[str, Any]:
    candidates = [
        config_path.with_suffix(".metadata.json"),
        config_path.parent / (config_path.stem + ".metadata.json"),
    ]
    for candidate in candidates:
        data = _load_json(candidate)
        if isinstance(data, dict):
            params = data.get("parameters")
            return params if isinstance(params, dict) else {}
    return {}


def _find_config_for_design(root: Path, design: str) -> Optional[Path]:
    candidates = [
        root / "configs" / f"{design}.yml",
        root / "configs" / f"{design}.yaml",
        root / "artifacts" / design / "config.yml",
        root / "artifacts" / design / "build" / "config.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list((root / "configs").glob(f"*{design}*.yml")) if (root / "configs").exists() else []
    if matches:
        return matches[0]
    return None


def _extract_evidence_lines(source_path: Path) -> List[Dict[str, Any]]:
    if not source_path.exists():
        return []
    pattern = re.compile(
        r"FPGAI parallel evidence:\s+"
        r"op=(?P<op>\S+)\s+"
        r"name=(?P<name>\S+)\s+"
        r"pipeline_ii=(?P<pipeline_ii>\d+)\s+"
        r"input_unroll=(?P<input_unroll>\d+)\s+"
        r"output_unroll=(?P<output_unroll>\d+)\s+"
        r"input_partition=(?P<input_partition>\d+)\s+"
        r"output_partition=(?P<output_partition>\d+)\s+"
        r"weight_partition=(?P<weight_partition>\d+)"
    )
    out: List[Dict[str, Any]] = []
    for lineno, line in enumerate(source_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        m = pattern.search(line)
        if not m:
            continue
        row: Dict[str, Any] = {"line": lineno, **m.groupdict()}
        for key in ["pipeline_ii", "input_unroll", "output_unroll", "input_partition", "output_partition", "weight_partition"]:
            row[key] = int(row[key])
        out.append(row)
    return out


def _first_metric(evidence: List[Dict[str, Any]], op: str, key: str) -> Optional[int]:
    for row in evidence:
        if row.get("op") == op and key in row:
            return int(row[key])
    return None


def _parse_csynth_report(report_dir: Path) -> Dict[str, Any]:
    # Keep this intentionally tolerant. Vitis report formatting differs by flow.
    out: Dict[str, Any] = {}
    if not report_dir.exists():
        return out
    candidates = list(report_dir.glob("*csynth*.rpt")) + list(report_dir.glob("*.rpt"))
    for rpt in candidates:
        text = rpt.read_text(encoding="utf-8", errors="ignore")
        out.setdefault("csynth_report", str(rpt))
        # Latency often appears in a table row with min/max latency.
        m = re.search(r"Latency\s*\(cycles\).*?\n(?:[-+|\s]+\n)?(?P<body>.{0,800})", text, flags=re.S)
        if m and "latency_section_seen" not in out:
            out["latency_section_seen"] = True
        # Resource rows often include BRAM_18K, DSP, FF, LUT.
        for key, aliases in {
            "bram_18k": ["BRAM_18K", "BRAM"],
            "dsp": ["DSP48E", "DSP"],
            "ff": ["FF"],
            "lut": ["LUT"],
            "uram": ["URAM"],
        }.items():
            if key in out:
                continue
            for alias in aliases:
                mm = re.search(rf"\b{re.escape(alias)}\b[^\n]*", text)
                if mm:
                    out[key + "_row"] = mm.group(0).strip()
                    break
    return out


def _results_by_name(root: Path) -> Dict[str, Dict[str, Any]]:
    data = _load_json(root / "results.json")
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for row in data.get("results", []) or []:
            if isinstance(row, dict) and row.get("design_name"):
                out[str(row["design_name"])] = row
    return out


def _design_dirs(root: Path, results: Dict[str, Dict[str, Any]]) -> Iterable[str]:
    names = list(results.keys())
    artifacts = root / "artifacts"
    if artifacts.exists():
        for p in sorted(artifacts.iterdir()):
            if p.is_dir() and p.name not in names:
                names.append(p.name)
    return names


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: extract_parallel_policy_evidence.py <experiment_dir>", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    if not root.exists():
        print(f"[ERROR] experiment dir does not exist: {root}", file=sys.stderr)
        return 1

    results = _results_by_name(root)
    rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}

    for design in _design_dirs(root, results):
        build = root / "artifacts" / design / "build"
        hls = build / "hls"
        source = hls / "src" / "deeplearn.cpp"
        evidence = _extract_evidence_lines(source)
        summary = _parse_summary(build / "bench" / "summary.txt")
        config_path = _find_config_for_design(root, design)
        params = _parse_materialized_metadata(config_path) if config_path else {}
        report_data = _parse_csynth_report(hls / "fpgai_hls_proj" / "sol1" / "syn" / "report")
        result = results.get(design, {})

        row = {
            "design": design,
            "status": result.get("status"),
            "returncode": result.get("returncode"),
            "passed": summary.get("passed"),
            "cosine": summary.get("cosine_similarity"),
            "policy": params.get("policy"),
            "pe": params.get("pe"),
            "simd": params.get("simd"),
            "unroll_factor": params.get("unroll_factor"),
            "partition_factor": params.get("partition_factor"),
            "pipeline_style": params.get("pipeline_style"),
            "evidence_lines": len(evidence),
            "conv_pipeline_ii": _first_metric(evidence, "Conv", "pipeline_ii"),
            "conv_oc_unroll": _first_metric(evidence, "Conv", "output_unroll"),
            "conv_ic_unroll": _first_metric(evidence, "Conv", "input_unroll"),
            "conv_weight_partition": _first_metric(evidence, "Conv", "weight_partition"),
            "dense_pipeline_ii": _first_metric(evidence, "Dense", "pipeline_ii"),
            "dense_in_unroll": _first_metric(evidence, "Dense", "input_unroll"),
            "dense_out_unroll": _first_metric(evidence, "Dense", "output_unroll"),
            "dense_weight_partition": _first_metric(evidence, "Dense", "weight_partition"),
            "has_array_partition_pragmas": bool(source.exists() and "ARRAY_PARTITION" in source.read_text(encoding="utf-8", errors="ignore")),
            "has_unroll_pragmas": bool(source.exists() and "UNROLL" in source.read_text(encoding="utf-8", errors="ignore")),
            "has_pipeline_pragmas": bool(source.exists() and "PIPELINE" in source.read_text(encoding="utf-8", errors="ignore")),
        }
        row.update(report_data)
        rows.append(row)
        detailed[design] = {
            "result": result,
            "summary": summary,
            "parameters": params,
            "evidence": evidence,
            "reports": report_data,
        }

    out_dir = root / "parallel_policy_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "parallel_policy_evidence.json"
    md_path = out_dir / "parallel_policy_evidence.md"
    csv_path = out_dir / "parallel_policy_evidence.csv"

    json_path.write_text(json.dumps({"rows": rows, "detailed": detailed}, indent=2), encoding="utf-8")

    headers = [
        "design", "status", "passed", "cosine", "policy", "pe", "simd",
        "unroll_factor", "partition_factor", "conv_oc_unroll", "conv_ic_unroll",
        "dense_out_unroll", "dense_in_unroll", "dense_weight_partition", "evidence_lines",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("| design | status | passed | cosine | policy | PE | SIMD | unroll | partition | Conv OC/IC | Dense OUT/IN | W part | evidence |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---|---|---:|---:|")
    for row in rows:
        lines.append(
            "| {design} | {status} | {passed} | {cosine} | {policy} | {pe} | {simd} | {unroll_factor} | {partition_factor} | {conv_oc_unroll}/{conv_ic_unroll} | {dense_out_unroll}/{dense_in_unroll} | {dense_weight_partition} | {evidence_lines} |".format(**{k: row.get(k, "") for k in headers + ["conv_oc_unroll", "conv_ic_unroll", "dense_out_unroll", "dense_in_unroll", "dense_weight_partition"]})
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[OK] Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
