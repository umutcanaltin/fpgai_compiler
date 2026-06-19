#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _find_first(root: Path, name: str) -> Optional[Path]:
    if not root.exists():
        return None
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


def _nested(d: Any, keys: List[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _training_compare_summary(compare: Dict[str, Any]) -> Dict[str, Any]:
    grads = compare.get("grads", {}) if isinstance(compare, dict) else {}
    weights_after = compare.get("weights_after", {}) if isinstance(compare, dict) else {}
    weight_delta = compare.get("weight_delta", {}) if isinstance(compare, dict) else {}
    layerwise = compare.get("layerwise_summary", {}) if isinstance(compare, dict) else {}

    return {
        "grad_cosine": grads.get("cosine"),
        "grad_mae": grads.get("mae"),
        "grad_max_abs": grads.get("max_abs"),
        "weight_after_cosine": weights_after.get("cosine"),
        "weight_after_mae": weights_after.get("mae"),
        "weight_after_max_abs": weights_after.get("max_abs"),
        "weight_delta_cosine": weight_delta.get("cosine"),
        "weight_delta_mae": weight_delta.get("mae"),
        "weight_delta_max_abs": weight_delta.get("max_abs"),
        "forward_layers": _nested(layerwise, ["forward_layerwise", "count"]),
        "backward_layers": _nested(layerwise, ["backward_input_layerwise", "count"]),
        "param_grad_weight_layers": _nested(layerwise, ["param_grad_weight_layerwise", "count"]),
        "param_grad_bias_layers": _nested(layerwise, ["param_grad_bias_layerwise", "count"]),
    }


def _source_markers(hls_src: Path) -> Dict[str, Any]:
    src = _read_text(hls_src)
    return {
        "has_training_top": 'extern "C" void' in src and "int mode" in src,
        "has_preload_mode": "mode == 0" in src,
        "has_emit_weights_mode": "mode == 1" in src,
        "has_train_mode": "mode == 2" in src or "target_buf" in src,
        "has_sgd_update": "sgd_update" in src,
        "has_backward": "backward" in src or "dW_" in src,
        "has_weight_stream": "aux" in src and "read_f32(aux)" in src,
    }


def _bench_files(hls_dir: Path) -> Dict[str, bool]:
    return {
        "hls_grads_bin": _find_first(hls_dir, "grads.bin") is not None,
        "hls_weights_before_bin": _find_first(hls_dir, "weights_before.bin") is not None,
        "hls_weights_after_bin": _find_first(hls_dir, "weights_after.bin") is not None,
    }


def _model_from_config(config_path: Optional[Path]) -> str:
    if config_path is None or not config_path.exists():
        return ""
    text = _read_text(config_path)
    m = re.search(r"^\s*path:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def collect(root: Path) -> List[Dict[str, Any]]:
    results = _read_json(root / "results.json") or {}
    rows: List[Dict[str, Any]] = []
    for item in results.get("results", []):
        design = item.get("design_name") or item.get("name") or "unknown"
        artifact_root = root / "artifacts" / design / "build"
        hls_dir = artifact_root / "hls"
        config_path = root / "configs" / f"{design}.yml"
        manifest = _read_json(artifact_root / "manifest.json") or {}
        compare_path = artifact_root / "training_compare" / "results.json"
        compare = _read_json(compare_path) or {}
        top_src = hls_dir / "src" / "deeplearn.cpp"
        row: Dict[str, Any] = {
            "design": design,
            "status": item.get("status"),
            "returncode": item.get("returncode"),
            "model": _model_from_config(config_path),
            "hls_ok": _nested(manifest, ["hls", "ok"], item.get("returncode") == 0),
            "training_plan": bool(_nested(manifest, ["training_plan"], None)),
            "training_reference": bool(_nested(manifest, ["training_reference"], None)),
            "training_compare": bool(compare),
        }
        row.update(_source_markers(top_src))
        row.update(_bench_files(hls_dir))
        row.update(_training_compare_summary(compare))
        rows.append(row)
    return rows


def write_table(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "design", "status", "model", "hls_ok", "training_plan", "training_reference", "training_compare",
        "has_training_top", "has_preload_mode", "has_emit_weights_mode", "has_train_mode", "has_sgd_update", "has_backward",
        "hls_grads_bin", "hls_weights_before_bin", "hls_weights_after_bin",
        "grad_cosine", "grad_mae", "grad_max_abs", "weight_after_cosine", "weight_after_mae", "weight_after_max_abs",
        "weight_delta_cosine", "weight_delta_mae", "weight_delta_max_abs",
        "forward_layers", "backward_layers", "param_grad_weight_layers", "param_grad_bias_layers",
    ]
    (out_dir / "training_accelerator_evidence.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "training_accelerator_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    lines = ["# Training accelerator evidence", "", "| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in fields) + " |")
    (out_dir / "training_accelerator_evidence.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    claim_rows = []
    any_pass = any(r.get("status") == "passed" and r.get("training_compare") for r in rows)
    all_pass = rows and all(r.get("status") == "passed" for r in rows)
    has_all_bins = rows and all(r.get("hls_grads_bin") and r.get("hls_weights_before_bin") and r.get("hls_weights_after_bin") for r in rows)
    has_modes = rows and all(r.get("has_preload_mode") and r.get("has_emit_weights_mode") and r.get("has_train_mode") for r in rows)
    claim_rows.append({"claim":"On-device training HLS path emits forward/backward/update kernel", "status":"supported" if all_pass and has_modes else "partial", "evidence":"mode 0 preload, mode 1 weights snapshot, mode 2 training step"})
    claim_rows.append({"claim":"HLS training artifacts include gradients and before/after weights", "status":"supported" if all_pass and has_all_bins else "partial", "evidence":"grads.bin, weights_before.bin, weights_after.bin generated by C simulation"})
    claim_rows.append({"claim":"Training HLS step is compared against Python reference", "status":"supported" if any_pass else "partial", "evidence":"training_compare/results.json with gradient and weight-update metrics"})
    claim_rows.append({"claim":"Full multi-epoch training convergence on hardware", "status":"not claimed by this sprint", "evidence":"this sprint validates one training step; convergence remains a future experiment"})
    (out_dir / "training_claim_support.json").write_text(json.dumps(claim_rows, indent=2), encoding="utf-8")
    claim_fields = ["claim", "status", "evidence"]
    claim_lines = ["# Training claim-support status", "", "| claim | status | evidence |", "|---|---|---|"]
    for r in claim_rows:
        claim_lines.append(f"| {r['claim']} | {r['status']} | {r['evidence']} |")
    (out_dir / "training_claim_support.md").write_text("\n".join(claim_lines)+"\n", encoding="utf-8")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/sprint13a_training_accelerator")
    out = root / "training_accelerator_evidence"
    rows = collect(root)
    write_table(rows, out)
    print((out / "training_accelerator_evidence.md").read_text(encoding="utf-8"))
    print(f"[OK] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
