#!/usr/bin/env python3
"""
Collect Sprint 16B training convergence evidence from existing FPGAI experiment outputs.

Outputs:
  evidence/sprint16b_training_convergence/training_convergence.csv
  evidence/sprint16b_training_convergence/training_convergence.md
  evidence/sprint16b_training_convergence/training_convergence.json

The collector is intentionally schema-tolerant because earlier sprint outputs may store
metrics in results.json, summary.json, manifest files, per-design metadata, or logs.
It never invents pass evidence: unknown fields are written as UNKNOWN/blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

BOOL_TRUE = {"true", "1", "yes", "y", "pass", "passed", "ok"}
BOOL_FALSE = {"false", "0", "no", "n", "fail", "failed"}

KEY_ALIASES = {
    "design": ["design", "design_name", "name", "case", "experiment", "config_name"],
    "epochs": ["epochs", "num_epochs", "training_epochs"],
    "batch_size": ["batch_size", "batch", "mini_batch_size"],
    "train_steps": ["train_steps", "steps", "num_steps", "training_steps"],
    "initial_loss": ["initial_loss", "loss_initial", "first_loss", "start_loss", "eval_initial_loss", "loss_before"],
    "final_loss": ["final_loss", "loss_final", "last_loss", "end_loss", "eval_final_loss", "loss_after"],
    "loss_delta": ["loss_delta", "delta_loss", "loss_decrease", "loss_reduction"],
    "loss_decreased": ["loss_decreased", "decreased", "loss_reduced"],
    "training_compare": ["training_compare", "software_compare", "compare_training", "training_reference_compare"],
    "native_accumulated_optimizer": [
        "native_accumulated_optimizer", "native_accumulated", "accumulated_optimizer",
        "native_accumulated_batch", "gradient_accumulation", "accumulate_gradients"
    ],
    "eval_only_loss_mode": ["eval_only_loss_mode", "eval_loss_only", "loss_eval_only"],
    "software_reference_match": [
        "software_reference_match", "reference_match", "software_match",
        "matches_software", "training_compare_passed"
    ],
}

LOSS_KEYS = [
    "loss", "train_loss", "training_loss", "eval_loss", "validation_loss",
    "mse", "loss_value", "initial_loss", "final_loss"
]

@dataclass
class Row:
    design: str
    source: str
    epochs: str = ""
    batch_size: str = ""
    train_steps: str = ""
    initial_loss: str = ""
    final_loss: str = ""
    loss_delta: str = ""
    loss_decreased: str = "UNKNOWN"
    training_compare: str = "UNKNOWN"
    native_accumulated_optimizer: str = "UNKNOWN"
    eval_only_loss_mode: str = "UNKNOWN"
    software_reference_match: str = "UNKNOWN"
    status: str = "UNKNOWN"
    notes: str = ""


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def iter_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_dicts(item)


def norm_bool(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "UNKNOWN"
    s = str(v).strip().lower()
    if s in BOOL_TRUE:
        return "True"
    if s in BOOL_FALSE:
        return "False"
    return "UNKNOWN"


def as_float(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def fmt_num(x: Any) -> str:
    f = as_float(x)
    if f is None:
        return "" if x is None else str(x)
    return f"{f:.8g}"


def pick(d: dict[str, Any], key: str) -> Any:
    aliases = KEY_ALIASES.get(key, [key])
    lower = {str(k).lower(): k for k in d.keys()}
    for a in aliases:
        if a in d:
            return d[a]
        lk = a.lower()
        if lk in lower:
            return d[lower[lk]]
    return None


def find_loss_sequence(obj: Any) -> list[float]:
    losses: list[float] = []

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in LOSS_KEYS:
                    if isinstance(v, list):
                        for item in v:
                            f = as_float(item)
                            if f is not None:
                                losses.append(f)
                    else:
                        f = as_float(v)
                        if f is not None:
                            losses.append(f)
                visit(v)
        elif isinstance(x, list):
            for item in x:
                visit(item)

    visit(obj)
    return losses


def parse_losses_from_text(path: Path) -> list[float]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    vals: list[float] = []
    patterns = [
        r"(?:^|\b)(?:loss|train_loss|eval_loss|mse)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"(?:initial_loss|final_loss)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            f = as_float(m.group(1))
            if f is not None:
                vals.append(f)
    return vals


def infer_design(path: Path, exp_root: Path, d: dict[str, Any] | None = None) -> str:
    if d:
        val = pick(d, "design")
        if val:
            return str(val)
    parts = path.relative_to(exp_root).parts
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if path.parent.name not in {".", exp_root.name}:
        return path.parent.name
    return exp_root.name


def row_from_dict(d: dict[str, Any], source: Path, exp_root: Path, obj_for_losses: Any) -> Row:
    design = infer_design(source, exp_root, d)
    row = Row(design=design, source=str(source.relative_to(exp_root)))

    for field in ["epochs", "batch_size", "train_steps"]:
        val = pick(d, field)
        if val is not None:
            setattr(row, field, str(val))

    initial = pick(d, "initial_loss")
    final = pick(d, "final_loss")
    delta = pick(d, "loss_delta")

    losses = find_loss_sequence(obj_for_losses)
    if initial is None and losses:
        initial = losses[0]
    if final is None and len(losses) >= 2:
        final = losses[-1]

    row.initial_loss = fmt_num(initial)
    row.final_loss = fmt_num(final)

    fi = as_float(initial)
    ff = as_float(final)
    fd = as_float(delta)
    if fd is None and fi is not None and ff is not None:
        fd = fi - ff
    row.loss_delta = fmt_num(fd)

    for field in [
        "loss_decreased", "training_compare", "native_accumulated_optimizer",
        "eval_only_loss_mode", "software_reference_match"
    ]:
        val = pick(d, field)
        nb = norm_bool(val)
        if nb != "UNKNOWN":
            setattr(row, field, nb)

    if row.loss_decreased == "UNKNOWN" and fi is not None and ff is not None:
        row.loss_decreased = "True" if ff < fi else "False"

    status = d.get("status") or d.get("result") or d.get("passed") or d.get("ok")
    row.status = norm_bool(status)
    if row.status == "True":
        row.status = "PASS"
    elif row.status == "False":
        row.status = "FAIL"

    if losses:
        row.notes = f"loss_points={len(losses)}"

    return row


def merge_rows(rows: list[Row]) -> list[Row]:
    # Prefer richer rows per design. Keep separate rows if source differs and values are distinct.
    by_design: dict[str, Row] = {}
    for r in rows:
        score = sum(bool(getattr(r, f)) and getattr(r, f) != "UNKNOWN" for f in asdict(r))
        old = by_design.get(r.design)
        old_score = -1 if old is None else sum(bool(getattr(old, f)) and getattr(old, f) != "UNKNOWN" for f in asdict(old))
        if old is None or score > old_score:
            by_design[r.design] = r
    return sorted(by_design.values(), key=lambda x: x.design)


def write_outputs(rows: list[Row], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(Row("", "").__dict__.keys())

    csv_path = out / "training_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    json_path = out / "training_convergence.json"
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    md_path = out / "training_convergence.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Sprint 16B Training Convergence Evidence\n\n")
        f.write("This table is generated from existing experiment artifacts. Unknown values are not inferred as passing evidence.\n\n")
        f.write("| design | initial_loss | final_loss | loss_delta | loss_decreased | training_compare | native_accumulated_optimizer | eval_only_loss_mode | software_reference_match | source | notes |\n")
        f.write("|---|---:|---:|---:|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r.design} | {r.initial_loss} | {r.final_loss} | {r.loss_delta} | "
                f"{r.loss_decreased} | {r.training_compare} | {r.native_accumulated_optimizer} | "
                f"{r.eval_only_loss_mode} | {r.software_reference_match} | {r.source} | {r.notes} |\n"
            )
        f.write("\n## Safe claim\n\n")
        if any(r.loss_decreased == "True" for r in rows):
            f.write(
                "FPGAI-generated training accelerators demonstrate loss reduction in small "
                "multi-epoch HLS CSim convergence tests for the evaluated designs.\n"
            )
        else:
            f.write(
                "No loss-decrease claim should be made yet because this collector did not find "
                "a row with loss_decreased=True.\n"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="Experiment directory, e.g. experiments/sprint13f_training_multi_epoch_convergence_v3")
    ap.add_argument("--out", default="evidence/sprint16b_training_convergence")
    args = ap.parse_args()

    exp = Path(args.experiment).resolve()
    out = Path(args.out).resolve()
    if not exp.exists():
        raise SystemExit(f"Experiment directory does not exist: {exp}")

    rows: list[Row] = []
    json_files = sorted(exp.rglob("*.json"))
    for jf in json_files:
        obj = load_json(jf)
        if obj is None:
            continue
        # Rows from dicts containing meaningful keys.
        candidates = []
        for d in iter_dicts(obj):
            has_signal = any(pick(d, k) is not None for k in KEY_ALIASES) or bool(find_loss_sequence(d))
            if has_signal:
                candidates.append(d)
        if candidates:
            # Prefer top-level dict when meaningful; otherwise all rich candidates.
            for d in candidates[:25]:
                rows.append(row_from_dict(d, jf, exp, d))
        else:
            losses = find_loss_sequence(obj)
            if losses:
                rows.append(row_from_dict({}, jf, exp, obj))

    # Fallback: parse logs/text for loss values.
    if not rows or not any(r.initial_loss and r.final_loss for r in rows):
        for tf in list(exp.rglob("*.log")) + list(exp.rglob("*.txt")) + list(exp.rglob("*.md")):
            losses = parse_losses_from_text(tf)
            if len(losses) >= 2:
                row = Row(design=infer_design(tf, exp), source=str(tf.relative_to(exp)))
                row.initial_loss = fmt_num(losses[0])
                row.final_loss = fmt_num(losses[-1])
                row.loss_delta = fmt_num(losses[0] - losses[-1])
                row.loss_decreased = "True" if losses[-1] < losses[0] else "False"
                row.notes = f"parsed_text_loss_points={len(losses)}"
                rows.append(row)

    rows = merge_rows(rows)
    write_outputs(rows, out)

    print(f"Wrote {out / 'training_convergence.csv'}")
    print(f"Wrote {out / 'training_convergence.md'}")
    print(f"Wrote {out / 'training_convergence.json'}")
    print(f"rows={len(rows)}")
    print(f"loss_decreased_true={sum(r.loss_decreased == 'True' for r in rows)}")
    print(f"training_compare_true={sum(r.training_compare == 'True' for r in rows)}")
    print(f"native_accumulated_optimizer_true={sum(r.native_accumulated_optimizer == 'True' for r in rows)}")
    print(f"eval_only_loss_mode_true={sum(r.eval_only_loss_mode == 'True' for r in rows)}")
    print(f"software_reference_match_true={sum(r.software_reference_match == 'True' for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
