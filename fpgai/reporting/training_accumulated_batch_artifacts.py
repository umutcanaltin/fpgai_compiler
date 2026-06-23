#!/usr/bin/env python3
"""Extract accumulated mini-batch training artifacts.

This extractor is intentionally conservative:
- It only marks accumulated_batch=True when the design name/config indicates
  an accumulated-batch design, the HLS artifacts exist, and the manifest has
  a training_compare section.
- optimizer_location is reported as testbench_accumulated_update because the
  current training top ABI still exposes per-sample train/update behavior.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _truth(v: Any) -> bool:
    return bool(v)


def _find_design_dirs(exp_dir: Path) -> List[Path]:
    artifacts = exp_dir / "artifacts"
    if not artifacts.exists():
        return []
    return sorted([p for p in artifacts.iterdir() if p.is_dir()])


def _find_file(build: Path, name: str) -> Optional[Path]:
    direct = build / name
    if direct.exists():
        return direct
    hits = sorted(build.rglob(name))
    return hits[0] if hits else None


def _parse_design_params(design: str, manifest: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    # Prefer explicit config/manifest values if future versions store them.
    for key in ("training", "training_cfg", "config"):
        cfg = manifest.get(key)
        if isinstance(cfg, dict):
            steps = cfg.get("train_steps") or cfg.get("steps") or cfg.get("training_steps")
            batch = cfg.get("batch_size") or cfg.get("microbatch") or cfg.get("batch")
            if steps is not None or batch is not None:
                try:
                    return (int(steps) if steps is not None else None, int(batch) if batch is not None else None)
                except Exception:
                    pass

    m = re.search(r"_(\d+)step_b(\d+)_", design)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"_(\d+)step.*?_b(\d+)", design)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _parse_summary(build: Path) -> Dict[str, Any]:
    # Future-proof: read an explicit JSON summary if the testbench writes one.
    for name in ("accumulated_batch_summary.json", "training_batch_summary.json", "multistep_summary.json"):
        p = _find_file(build, name)
        if p:
            data = _load_json(p)
            if data:
                return data

    # Current fallback: parse Vitis/HLS logs for the printed summary line.
    text = ""
    for p in list((build / "hls").rglob("*.log"))[:50]:
        try:
            t = p.read_text(errors="ignore")
        except Exception:
            continue
        if "[TB-TRAIN]" in t:
            text += "\n" + t
    m = re.search(r"Multi-step summary:\s*train_steps=(\d+)\s+batch_size=(\d+)\s+total_train_calls=(\d+)", text)
    if m:
        return {
            "train_steps": int(m.group(1)),
            "batch_size": int(m.group(2)),
            "total_forward_backward_calls": int(m.group(3)),
        }
    return {}


def _manifest_training_compare(manifest: Dict[str, Any]) -> Dict[str, Any]:
    tc = manifest.get("training_compare")
    return tc if isinstance(tc, dict) else {}


def _weight_words_from_file(path: Optional[Path]) -> Optional[int]:
    if not path or not path.exists():
        return None
    try:
        return path.stat().st_size // 4
    except Exception:
        return None


def _row_for(design_dir: Path, results_by_name: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    design = design_dir.name
    build = design_dir / "build"
    manifest = _load_json(build / "manifest.json")
    result = results_by_name.get(design, {})

    weights_before = _find_file(build, "weights_before.bin")
    grads = _find_file(build, "grads.bin")
    weights_after = _find_file(build, "weights_after.bin")
    hls_ok = bool(weights_before and grads and weights_after)

    tc = _manifest_training_compare(manifest)
    training_compare = bool(tc)

    summary = _parse_summary(build)
    steps, batch = _parse_design_params(design, manifest)
    train_steps = int(summary.get("train_steps") or steps or 0) or None
    batch_size = int(summary.get("batch_size") or batch or 0) or None
    total_calls = summary.get("total_forward_backward_calls") or summary.get("total_train_calls")
    if total_calls is None and train_steps is not None and batch_size is not None:
        total_calls = train_steps * batch_size
    if total_calls is not None:
        total_calls = int(total_calls)

    is_accum_design = "_accum_" in design or "accum" in design.lower()
    accumulated_batch = bool(is_accum_design and hls_ok and training_compare and batch_size and batch_size > 1)
    averaged_gradients = bool(accumulated_batch)
    optimizer_update_calls = train_steps if accumulated_batch else None
    optimizer_location = "testbench_accumulated_update" if accumulated_batch else ""

    return {
        "design": design,
        "status": result.get("status") or manifest.get("status") or "",
        "weights_mode": "stream" if "stream" in design else ("embedded" if "embedded" in design else ""),
        "hls_ok": hls_ok,
        "training_compare": training_compare,
        "accumulated_batch": accumulated_batch,
        "averaged_gradients": averaged_gradients,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "total_forward_backward_calls": total_calls,
        "optimizer_update_calls": optimizer_update_calls,
        "optimizer_location": optimizer_location,
        "weight_words": _weight_words_from_file(weights_before),
        "grad_cosine": tc.get("grad_cosine", ""),
        "weight_after_cosine": tc.get("weight_after_cosine", ""),
        "weight_delta_cosine": tc.get("weight_delta_cosine", ""),
    }


def _markdown(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "design", "status", "weights_mode", "hls_ok", "training_compare",
        "accumulated_batch", "averaged_gradients", "train_steps", "batch_size",
        "total_forward_backward_calls", "optimizer_update_calls", "optimizer_location",
        "weight_words", "grad_cosine", "weight_after_cosine", "weight_delta_cosine",
    ]
    out = ["# Accumulated mini-batch training artifacts", ""]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) if r.get(c, "") is not None else "" for c in cols) + " |")
    return "\n".join(out) + "\n"


def _csv(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "design", "status", "weights_mode", "hls_ok", "training_compare",
        "accumulated_batch", "averaged_gradients", "train_steps", "batch_size",
        "total_forward_backward_calls", "optimizer_update_calls", "optimizer_location",
        "weight_words", "grad_cosine", "weight_after_cosine", "weight_delta_cosine",
    ]
    def esc(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        if any(ch in s for ch in ',"\n'):
            s = '"' + s.replace('"', '""') + '"'
        return s
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(esc(r.get(c, "")) for c in cols))
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m fpgai.reporting.training_accumulated_batch_artifacts <experiment_dir>", file=sys.stderr)
        return 2
    exp_dir = Path(sys.argv[1])
    results = _load_json(exp_dir / "results.json")
    results_by_name = {r.get("design_name") or r.get("design") or "": r for r in results.get("results", []) if isinstance(r, dict)}
    rows = [_row_for(d, results_by_name) for d in _find_design_dirs(exp_dir)]
    rows.sort(key=lambda r: r["design"])

    out_dir = exp_dir / "training_accumulated_batch_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_accumulated_batch_artifacts.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "training_accumulated_batch_artifacts.md").write_text(_markdown(rows), encoding="utf-8")
    (out_dir / "training_accumulated_batch_artifacts.csv").write_text(_csv(rows), encoding="utf-8")

    print(_markdown(rows))
    print(f"[OK] Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
