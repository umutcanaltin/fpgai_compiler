"""R4h: read-only, SHA-pinned original HLS memory-module SOURCE comparison.

Reuses the original R4g source/RTL/ROM provenance audits, then compares
the *content* and a bounded lexical memory signature of each original
reachable named memory module. This is NOT Verilog elaboration,
parameter evaluation, RAMB binding, mapped storage, or physical E4.
No vendor tool or original-evidence file mutation occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments import cst_r4g_rtl_memory_bridge as r4g
from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments.cst_r2w_rtl_memory_forensics import _limited_text
from fpgai_research.experiments.cst_r2x_rtl_hierarchy_analysis import strip_noncode

SCHEMA = "fpgai-research/R4h-attested-HLS-memory-source-delta/v1"
_LIT_REG = re.compile(
    r"\breg\s+(?:signed\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?"
    r"([A-Za-z_$][A-Za-z0-9_$]*)\s*"
    r"\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*;", re.I)
_MAX_SNIPPETS = 24
_MAX_SOURCE_CHARS = 5000


def literal_reg_arrays(source: str) -> list[dict]:
    """Recognize only fully literal reg arrays; symbolic widths are UNKNOWN.

    Never interpret this list as the entire inferred or physical memory.
    """
    if not isinstance(source, str):
        raise ValueError("R4h requires Verilog source text")
    cleaned = strip_noncode(source)
    result = []
    for match in _LIT_REG.finditer(cleaned):
        width = (abs(int(match[1]) - int(match[2])) + 1
                 if match[1] is not None else 1)
        elements = abs(int(match[4]) - int(match[5])) + 1
        if width > 1_000_000 or elements > 10_000_000:
            raise ValueError("R4h unbounded literal Verilog reg array")
        result.append({
            "name": match[3],
            "declared_width_bits": width,
            "declared_elements": elements,
            "lexical_bits_product": width * elements,
            "scope": "SOURCE_DECLARATION_ONLY_NOT_MAPPED_CAPACITY",
        })
    return result


def memory_source_signatures(source: str) -> dict:
    """Show bounded original lines to guide the NEXT causal RTL experiment."""
    if not isinstance(source, str) or len(source) > 12 * 1024 * 1024:
        raise ValueError("R4h source text invalid or oversized")
    lines = []
    truncated = False
    for index, line in enumerate(source.splitlines(), 1):
        if re.search(
            r"\b(?:module|parameter|localparam|reg|wire|logic)\b|"
            r"\breadmem[hb]\b|DataWidth|AddressWidth|DataDepth|ram_style|"
            r"\b(?:WIDTH|DEPTH|ADDR_WIDTH)\b", line, re.I
        ):
            if len(lines) >= _MAX_SNIPPETS:
                truncated = True
                break
            lines.append({
                "line": index,
                "text": line.strip()[:200],
            })
    return {
        "literal_reg_array_candidates": literal_reg_arrays(source),
        "literal_reg_array_recognizer_complete": False,
        "candidate_source_lines": lines,
        "source_line_snippets_truncated": truncated,
        "interpretation": "LEXICAL_SOURCE_ONLY_SYMBOLIC_DIMENSIONS_UNRESOLVED",
    }


def inspect_arm(study: Path, observation: dict) -> dict:
    fmt = observation["format"]
    arm = observation["arm"]
    result = {
        "format": fmt, "arm": arm,
        "source_status": "UNKNOWN_UNATTESTED_OR_UNSUPPORTED_ORIGINAL_RTL",
        "memory_modules": None,
        "rom_content_sha256_multiset": None,
        "memory_module_total": None,
        "native_calls": 0,
        "physical_BRAM": None,
    }
    if observation["rtl_observation_status"] != "ATTESTED_STATIC_TYPE_GRAPH_ONLY":
        return result
    bits, integer = fmt
    pkg = (study / f"format_{bits:02d}_{integer:02d}"
           / arm / "vitis_source")
    rtl = pkg / "fpgai_hls_project" / "solution1" / "syn" / "verilog"
    if (pkg.is_symlink() or any(folder.is_symlink() for folder in (
            rtl, rtl.parent, rtl.parent.parent, rtl.parent.parent.parent))
            or not rtl.is_dir()):
        raise ValueError("R4h source RTL path missing or symlinked")
    known = observation["rtl_file_sha256"]
    rom = observation["rom_file_sha256"]
    names = observation["provisionally_reachable_memory_module_types"]
    if (not isinstance(known, dict) or not isinstance(rom, dict)
            or not isinstance(names, list)
            or not all(isinstance(n, str) for n in names)):
        raise ValueError("R4h missing R4g original source inventory")
    entries = []
    for name in names:
        if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name):
            raise ValueError("R4h unsupported module type name")
        # Existing R2x parser identifies a source MODULE TYPE, not a
        # guaranteed file of that name. Never invent a missing file.
        filename = name + ".v"
        if filename not in known:
            entries.append({
                "module_type": name,
                "source_status": "UNKNOWN_NO_SAME_NAMED_SOURCE_FILE",
                "source_sha256": None,
                "signature": None,
            })
            continue
        file = rtl / filename
        if file.is_symlink() or not file.is_file():
            raise ValueError("R4h original HDL changed or symlinked")
        raw = _limited_text(file)
        actual = sha256(file.read_bytes()).hexdigest()
        if actual != known[filename]:
            raise ValueError("R4h original RTL SHA mismatch")
        if len(raw) > _MAX_SOURCE_CHARS * 1000:
            raise ValueError("R4h source file above lexical scan budget")
        entries.append({
            "module_type": name,
            "source_status": "SOURCE_SHA_ATTESTED_TEXT_ONLY",
            "source_sha256": actual,
            "signature": memory_source_signatures(raw),
        })
    result.update({
        "source_status": "ATTESTED_SOURCE_MODULES_WITH_EXPLICIT_UNKNOWNS",
        "memory_modules": entries,
        "memory_module_total": len(entries),
        "rom_content_sha256_multiset": dict(sorted(Counter(rom.values()).items())),
    })
    return result


def compare_pair(a: dict, b: dict, *, fmt: list[int], bram_delta: int | None) -> dict:
    if (a.get("format") != fmt or b.get("format") != fmt
            or a.get("arm") != "full_H" or b.get("arm") != "row_H"):
        raise ValueError("R4h incompatible matched format/arm")
    known = (a.get("memory_modules") is not None
             and b.get("memory_modules") is not None)
    complete = known and all(
        m["source_status"] == "SOURCE_SHA_ATTESTED_TEXT_ONLY"
        for o in (a, b) for m in o["memory_modules"])
    result = {
        "format": fmt,
        "original_memory_source_status": (
            "COMPLETE_NAMED_SOURCE_CROSSWALK_ONLY" if complete
            else "UNKNOWN_OR_PARTIAL_NAMED_SOURCE_CROSSWALK"),
        "original_HLS_estimated_BRAM18K_delta": bram_delta,
        "candidate_memory_module_count_full": (
            a["memory_module_total"] if known else None),
        "candidate_memory_module_count_row": (
            b["memory_module_total"] if known else None),
        "same_name_module_sources_unchanged": None,
        "same_name_module_sources_changed": None,
        "memory_source_names_full_only": None,
        "memory_source_names_row_only": None,
        "ROM_initialization_contents_identical_multiset": None,
        "physical_BRAM_delta": None,
    }
    if not complete:
        return result
    x = {m["module_type"]: m["source_sha256"] for m in a["memory_modules"]}
    y = {m["module_type"]: m["source_sha256"] for m in b["memory_modules"]}
    shared = sorted(set(x) & set(y))
    result.update({
        "same_name_module_sources_unchanged": [
            n for n in shared if x[n] == y[n]],
        "same_name_module_sources_changed": [
            n for n in shared if x[n] != y[n]],
        "memory_source_names_full_only": sorted(set(x) - set(y)),
        "memory_source_names_row_only": sorted(set(y) - set(x)),
        "ROM_initialization_contents_identical_multiset": (
            a["rom_content_sha256_multiset"] == b["rom_content_sha256_multiset"]),
    })
    return result


def inspect(*, study: str | Path, original_study: str | Path) -> dict:
    base = r4g.inspect(study=study, original_study=original_study)
    if base["prespecified_pairs"] != len(r4d.FORMATS) or len(base["arms"]) != 6:
        raise ValueError("R4h incomplete R4g original cohort")
    records, pairs = [], []
    for i, fmt in enumerate(r4d.FORMATS):
        full = inspect_arm(Path(study), base["arms"][i * 2])
        row = inspect_arm(Path(study), base["arms"][i * 2 + 1])
        records.extend((full, row))
        pairs.append(compare_pair(
            full, row, fmt=list(fmt),
            bram_delta=base["pairs"][i][
                "HLS_estimated_BRAM18K_delta_row_minus_full"]))
    return {
        "schema": SCHEMA,
        "original_R4a_source_revision": base["original_R4a_source_revision"],
        "original_R4d_source_revision": base["original_R4d_source_revision"],
        "prespecified_format_pairs": len(pairs),
        "source_crosswalk_complete_pairs": sum(
            p["original_memory_source_status"] ==
            "COMPLETE_NAMED_SOURCE_CROSSWALK_ONLY" for p in pairs),
        "arms": records,
        "pairs": pairs,
        "limits": [
            "Only exact source file names of statically reachable module types are paired.",
            "Different source SHA does not prove a different RAM depth or mapped hardware.",
            "Identical ROM payloads do not prove identical instance connectivity.",
            "A literal reg declaration is not a physical RAM allocation.",
            "Source parameter expressions and generated configurations remain unresolved.",
            "No synthesis netlist, place-and-route, Vivado, Vitis or timing measurement.",
            "Missing old or new cases remain UNKNOWN, never zero.",
        ],
        "new_vendor_calls": 0,
        "original_evidence_modified": False,
        "physical_E4": None,
    }


def run(*, study: str | Path, original_study: str | Path,
        output: str | Path | None = None) -> dict:
    report = inspect(study=study, original_study=original_study)
    if output is not None:
        target = Path(output)
        dest = target.resolve()
        if any(dest == r or r in dest.parents for r in (
                Path(study).resolve(), Path(original_study).resolve())):
            raise ValueError("R4h output must be outside original study evidence")
        if target.exists() or target.is_symlink():
            raise FileExistsError("R4h output is create-only")
        target.mkdir(parents=True, exist_ok=False)
        content = json.dumps(report, sort_keys=True, indent=2,
                             allow_nan=False) + "\n"
        (target / "report.json").write_text(content)
        (target / "artifact_sha256.json").write_text(json.dumps({
            "report.json": sha256(content.encode()).hexdigest()
        }, sort_keys=True, indent=2) + "\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--original-study", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(json.dumps(run(
        study=args.study, original_study=args.original_study,
        output=args.output), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
