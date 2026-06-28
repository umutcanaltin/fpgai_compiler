"""Audit whether paper-matrix tiling and memory knobs materialize in generated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TILING_DESIGNS = [
    "kv260_tiling_small",
    "kv260_tiling_medium",
    "kv260_tiling_large",
]

MEMORY_DESIGNS = [
    "kv260_memory_bram",
    "kv260_memory_uram",
]

INTERESTING_PATTERNS = [
    re.compile(r"\btile\b", re.I),
    re.compile(r"\btiling\b", re.I),
    re.compile(r"\btile_size\b", re.I),
    re.compile(r"\bblock\b", re.I),
    re.compile(r"\bchunk\b", re.I),
    re.compile(r"ARRAY_PARTITION", re.I),
    re.compile(r"ARRAY_RESHAPE", re.I),
    re.compile(r"BIND_STORAGE", re.I),
    re.compile(r"RESOURCE", re.I),
    re.compile(r"RAM_1P|RAM_2P|RAM_S2P", re.I),
    re.compile(r"\bURAM\b", re.I),
    re.compile(r"\bBRAM\b", re.I),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm_text(path: Path) -> str:
    text = path.read_text(errors="ignore")
    # Remove absolute paths and comments that can make identical generated logic look different.
    text = re.sub(r"/home/[^\s\"']+", "<ABS_PATH>", text)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _norm_sha256(path: Path) -> str:
    return hashlib.sha256(_norm_text(path).encode()).hexdigest()


def _find_design_dir(base: Path, design: str) -> Path:
    matches = sorted(p for p in base.rglob(design) if p.is_dir() and p.name == design)
    if not matches:
        raise FileNotFoundError(f"could not find design directory for {design} under {base}")
    return matches[0]


def _collect_files(design_dir: Path) -> list[Path]:
    suffixes = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".tcl", ".yml", ".yaml", ".json"}
    out = []
    for p in design_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue
        rel = str(p.relative_to(design_dir))
        if "__pycache__" in rel:
            continue
        out.append(p)
    return sorted(out)


def _collect_hls_source_files(design_dir: Path) -> list[Path]:
    suffixes = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".tcl"}
    out = []
    for p in design_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in suffixes:
            continue
        rel = str(p.relative_to(design_dir))
        if "hls" not in rel.lower() and "src" not in rel.lower():
            continue
        out.append(p)
    return sorted(out)


def _interesting_lines(path: Path, max_lines: int = 80) -> list[str]:
    lines = []
    try:
        raw = path.read_text(errors="ignore").splitlines()
    except Exception:
        return lines
    for i, line in enumerate(raw, start=1):
        if any(p.search(line) for p in INTERESTING_PATTERNS):
            lines.append(f"{path.name}:{i}: {line.rstrip()}")
            if len(lines) >= max_lines:
                break
    return lines


def _design_record(base: Path, design: str) -> dict:
    d = _find_design_dir(base, design)
    files = _collect_files(d)
    hls_sources = _collect_hls_source_files(d)

    interesting = []
    for p in files:
        rel = p.relative_to(d)
        hits = _interesting_lines(p)
        if hits:
            interesting.append({
                "file": str(rel),
                "hits": hits[:40],
            })

    source_hashes = []
    for p in hls_sources:
        source_hashes.append({
            "file": str(p.relative_to(d)),
            "sha256": _sha256(p),
            "normalized_sha256": _norm_sha256(p),
            "bytes": p.stat().st_size,
        })

    return {
        "design": design,
        "design_dir": str(d),
        "file_count": len(files),
        "hls_source_count": len(hls_sources),
        "hls_source_hashes": source_hashes,
        "interesting_lines": interesting,
    }


def _compare(records: list[dict]) -> dict:
    by_rel: dict[str, dict[str, str]] = {}
    for r in records:
        design = r["design"]
        for h in r["hls_source_hashes"]:
            by_rel.setdefault(h["file"], {})[design] = h["normalized_sha256"]

    identical_files = []
    differing_files = []
    designs = [r["design"] for r in records]
    for rel, hashes in sorted(by_rel.items()):
        present = [d for d in designs if d in hashes]
        unique = sorted(set(hashes.values()))
        row = {
            "file": rel,
            "present_designs": present,
            "unique_normalized_hashes": len(unique),
        }
        if len(unique) == 1 and len(present) == len(designs):
            identical_files.append(row)
        else:
            differing_files.append(row)

    return {
        "designs": designs,
        "identical_hls_source_files": identical_files,
        "differing_hls_source_files": differing_files,
    }


def _write_md(path: Path, title: str, records: list[dict], comparison: dict) -> None:
    lines = [f"# {title}", ""]
    lines.append("## Design directories")
    for r in records:
        lines.append(f"- `{r['design']}`: `{r['design_dir']}`")
    lines.append("")

    lines.append("## HLS source comparison")
    lines.append(f"- identical HLS/source files: {len(comparison['identical_hls_source_files'])}")
    lines.append(f"- differing HLS/source files: {len(comparison['differing_hls_source_files'])}")
    lines.append("")

    lines.append("### Differing files")
    if comparison["differing_hls_source_files"]:
        for row in comparison["differing_hls_source_files"]:
            lines.append(f"- `{row['file']}`: unique normalized hashes = {row['unique_normalized_hashes']}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("### Interesting knob/materialization lines")
    for r in records:
        lines.append(f"## `{r['design']}`")
        if not r["interesting_lines"]:
            lines.append("- no tiling/memory/pragma lines found")
            continue
        for item in r["interesting_lines"]:
            lines.append(f"### `{item['file']}`")
            for hit in item["hits"]:
                lines.append(f"```text\n{hit}\n```")
    lines.append("")

    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
    ap.add_argument("--out", default="paper_results/audits")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = {
        "tiling": TILING_DESIGNS,
        "memory": MEMORY_DESIGNS,
    }

    summary = {}
    for group, designs in groups.items():
        records = [_design_record(base, d) for d in designs]
        comparison = _compare(records)
        payload = {
            "group": group,
            "records": records,
            "comparison": comparison,
        }
        (out / f"{group}_knob_materialization_audit.json").write_text(json.dumps(payload, indent=2))
        _write_md(
            out / f"{group}_knob_materialization_audit.md",
            f"{group.capitalize()} knob materialization audit",
            records,
            comparison,
        )

        summary[group] = {
            "designs": designs,
            "identical_hls_source_files": len(comparison["identical_hls_source_files"]),
            "differing_hls_source_files": len(comparison["differing_hls_source_files"]),
        }

    (out / "knob_materialization_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print("[SUMMARY]")
    for group, row in summary.items():
        print(
            f"{group}: identical_hls_source_files={row['identical_hls_source_files']} "
            f"differing_hls_source_files={row['differing_hls_source_files']}"
        )
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
