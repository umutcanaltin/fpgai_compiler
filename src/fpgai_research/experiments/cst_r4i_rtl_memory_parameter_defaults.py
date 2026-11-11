"""R4i: read-only parameter-default memory-dimension crosswalk for R4h.

Every value is derived from SHA-attested ORIGINAL HLS-generated Verilog
declaration excerpts supplied by R4h. Only literal DataWidth, AddressWidth
and AddressRange defaults plus an exact symbolic array declaration are
recognized. Defaults and products are NOT elaborated instance parameters,
synthesized RAM capacity, BRAM packing, physical RAMB counts, or timing.
No vendor calls and no changes to original experiment evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments import cst_r4h_hls_memory_source_crosswalk as r4h

SCHEMA = "fpgai-research/R4i-attested-RTL-memory-parameter-defaults/v1"
_PARAMETERS = ("DataWidth", "AddressWidth", "AddressRange")
_PARAMETER_LINE = re.compile(
    r"parameter\s+(DataWidth|AddressWidth|AddressRange)\s*=\s*(\d+)\s*;",
    re.ASCII,
)
_ARRAY_TEMPLATE = re.compile(
    r"\breg\s*\[\s*DataWidth\s*-\s*1\s*:\s*0\s*\]\s*"
    r"(?:ram|rom0)\s*\[\s*0\s*:\s*AddressRange\s*-\s*1\s*\]\s*;",
    re.ASCII,
)
_SCRATCH_SUFFIX = "_scratch_RAM_AUTO_1R1W"
_ACC_SUFFIX = "_acc_RAM_AUTO_1R1W"


def parameter_defaults(module: dict) -> dict:
    """Parse a finite SOURCE declaration excerpt, never an RTL instance."""
    result = {
        "module_type": module.get("module_type"),
        "original_source_sha256": module.get("source_sha256"),
        "status": "UNKNOWN_SOURCE_PARAMETERS",
        "default_DataWidth_bits": None,
        "default_AddressWidth_bits": None,
        "default_AddressRange_entries": None,
        "default_declaration_data_bits_product": None,
        "array_template_recognized": False,
        "source_parameter_line_numbers": None,
        "assumes_instance_defaults_unoverridden": None,
        "elaborated_instance_depth": None,
        "mapped_BRAM18K": None,
    }
    if module.get("source_status") != "SOURCE_SHA_ATTESTED_TEXT_ONLY":
        return result
    sig = module.get("signature")
    if not isinstance(sig, dict) or not isinstance(
            sig.get("candidate_source_lines"), list):
        return result
    if sig.get("source_line_snippets_truncated") is not False:
        return result
    found = {}
    lines = []
    for entry in sig["candidate_source_lines"]:
        if (not isinstance(entry, dict)
                or not isinstance(entry.get("line"), int)
                or type(entry.get("text")) is not str):
            return result
        line = entry["text"]
        lines.append(line)
        match = _PARAMETER_LINE.fullmatch(line.strip())
        if match:
            name = match[1]
            if name in found:
                return result  # ambiguous same-name parameter declaration
            value = int(match[2])
            if value <= 0 or value > 1_000_000:
                return result
            found[name] = (value, entry["line"])
        elif re.search(r"\bparameter\s+(?:DataWidth|AddressWidth|AddressRange)\b",
                       line):
            return result  # symbolic expression, alternate syntax, or cap
    if set(found) != set(_PARAMETERS):
        return result
    width, depth, addr = (
        found["DataWidth"][0], found["AddressRange"][0],
        found["AddressWidth"][0])
    if width > 4096 or addr > 64 or depth > (1 << addr):
        # An impossible address range cannot be called an understood
        # source dimension; leave UNKNOWN rather than guess truncation.
        return result
    joined = "\n".join(lines)
    template = _ARRAY_TEMPLATE.search(joined) is not None
    if not template:
        return result
    result.update({
        "status": "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY",
        "default_DataWidth_bits": width,
        "default_AddressWidth_bits": addr,
        "default_AddressRange_entries": depth,
        "default_declaration_data_bits_product": width * depth,
        "array_template_recognized": True,
        "source_parameter_line_numbers": {
            name: found[name][1] for name in _PARAMETERS},
        "assumes_instance_defaults_unoverridden": None,
    })
    return result


def summarize_arm(arm: dict) -> dict:
    if not isinstance(arm, dict) or not isinstance(arm.get("format"), list):
        raise ValueError("R4i missing original R4h arm identity")
    entries = arm.get("memory_modules")
    result = {
        "format": arm["format"], "arm": arm.get("arm"),
        "status": "UNKNOWN_NO_ATTESTED_MEMORY_SOURCE",
        "parameter_source_modules": None,
        "memory_default_profiles": None,
        "source_module_count": None,
        "new_native_calls": 0,
    }
    if entries is None:
        return result
    if not isinstance(entries, list):
        raise ValueError("R4i invalid original R4h memory source list")
    profiles = [parameter_defaults(e) for e in entries]
    result.update({
        "status": "SOURCE_PARAMETER_DEFAULTS_WITH_EXPLICIT_UNKNOWNS",
        "parameter_source_modules": profiles,
        "memory_default_profiles": {
            p["module_type"]: p for p in profiles
            if p["status"] == "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY"
        },
        "source_module_count": len(profiles),
    })
    return result


def _one(profiles: list[dict], suffix: str) -> dict | None:
    candidates = [p for p in profiles
                  if isinstance(p["module_type"], str)
                  and p["module_type"].endswith(suffix)]
    if len(candidates) != 1:
        return None
    item = candidates[0]
    if item["status"] != "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY":
        return None
    return item


def _rom_profiles(profiles: list[dict]) -> list[list[int]] | None:
    candidates = [p for p in profiles
                  if isinstance(p["module_type"], str)
                  and "_ROM_" in p["module_type"]]
    # The original R4g/R4h case has exactly three weight-ROM modules.
    # Fewer/extra candidates are UNKNOWN, never promoted to a match.
    if len(candidates) != 3 or any(
            p["status"] != "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY"
            for p in candidates):
        return None
    return sorted([
        [p["default_DataWidth_bits"], p["default_AddressWidth_bits"],
         p["default_AddressRange_entries"]]
        for p in candidates
    ])


def compare_pair(full: dict, row: dict, *, format: list[int],
                 source_bram_delta: int | None,
                 rom_payloads_identical: bool | None) -> dict:
    if (full.get("format") != format or row.get("format") != format
            or full.get("arm") != "full_H"
            or row.get("arm") != "row_H"):
        raise ValueError("R4i mismatched original format/arms")
    result = {
        "format": format,
        "status": "UNKNOWN_PARAMETER_DEFAULTS",
        "HLS_estimated_BRAM18K_delta": source_bram_delta,
        "original_ROM_payloads_identical_multiset": rom_payloads_identical,
        "scratch_default_full": None,
        "scratch_default_row": None,
        "scratch_default_depth_entries_delta": None,
        "scratch_default_declaration_data_bits_delta": None,
        "scratch_default_depth_fraction_saved": None,
        "scratch_default_address_width_same": None,
        "accumulator_default_full": None,
        "accumulator_default_row": None,
        "accumulator_default_unchanged": None,
        "ROM_default_parameter_profiles_identical_multiset": None,
        "source_parameter_overrides_checked": False,
        "elaborated_instance_capacity_delta": None,
        "physical_BRAM_delta": None,
        "physical_achieved_Fmax": None,
    }
    x, y = full.get("parameter_source_modules"), row.get("parameter_source_modules")
    if x is None or y is None:
        return result
    sx, sy = _one(x, _SCRATCH_SUFFIX), _one(y, _SCRATCH_SUFFIX)
    ax, ay = _one(x, _ACC_SUFFIX), _one(y, _ACC_SUFFIX)
    rom_x, rom_y = _rom_profiles(x), _rom_profiles(y)
    if sx is None or sy is None or ax is None or ay is None:
        return result
    # Do not report a numerically exact default change without fixed width
    # and exact array-template evidence. All values are module defaults.
    same_width = sx["default_DataWidth_bits"] == sy["default_DataWidth_bits"]
    if not same_width:
        return result
    result.update({
        "status": "SOURCE_MODULE_DEFAULT_DIMENSIONS_ONLY",
        "scratch_default_full": {
            "DataWidth": sx["default_DataWidth_bits"],
            "AddressWidth": sx["default_AddressWidth_bits"],
            "AddressRange": sx["default_AddressRange_entries"]},
        "scratch_default_row": {
            "DataWidth": sy["default_DataWidth_bits"],
            "AddressWidth": sy["default_AddressWidth_bits"],
            "AddressRange": sy["default_AddressRange_entries"]},
        "scratch_default_depth_entries_delta": (
            sy["default_AddressRange_entries"] -
            sx["default_AddressRange_entries"]),
        "scratch_default_declaration_data_bits_delta": (
            sy["default_declaration_data_bits_product"] -
            sx["default_declaration_data_bits_product"]),
        "scratch_default_depth_fraction_saved": (
            (sx["default_AddressRange_entries"] -
             sy["default_AddressRange_entries"]) /
            sx["default_AddressRange_entries"]),
        "scratch_default_address_width_same": (
            sx["default_AddressWidth_bits"] ==
            sy["default_AddressWidth_bits"]),
        "accumulator_default_full": {
            "DataWidth": ax["default_DataWidth_bits"],
            "AddressWidth": ax["default_AddressWidth_bits"],
            "AddressRange": ax["default_AddressRange_entries"]},
        "accumulator_default_row": {
            "DataWidth": ay["default_DataWidth_bits"],
            "AddressWidth": ay["default_AddressWidth_bits"],
            "AddressRange": ay["default_AddressRange_entries"]},
        "accumulator_default_unchanged": (
            (ax["default_DataWidth_bits"],
             ax["default_AddressWidth_bits"],
             ax["default_AddressRange_entries"]) ==
            (ay["default_DataWidth_bits"],
             ay["default_AddressWidth_bits"],
             ay["default_AddressRange_entries"])),
        "ROM_default_parameter_profiles_identical_multiset": (
            rom_x == rom_y if rom_x is not None and rom_y is not None
            else None),
    })
    return result


def inspect(*, study: str | Path, original_study: str | Path) -> dict:
    base = r4h.inspect(study=study, original_study=original_study)
    if (base.get("prespecified_format_pairs") != len(r4d.FORMATS)
            or len(base.get("arms") or []) != 2 * len(r4d.FORMATS)
            or len(base.get("pairs") or []) != len(r4d.FORMATS)):
        raise ValueError("R4i incomplete original R4h source population")
    arms, pairs = [], []
    for i, pair in enumerate(base["pairs"]):
        width, integer = r4d.FORMATS[i]
        fmt = [width, integer]
        full = summarize_arm(base["arms"][2*i])
        row = summarize_arm(base["arms"][2*i+1])
        arms.extend((full, row))
        pairs.append(compare_pair(
            full, row, format=fmt,
            source_bram_delta=pair["original_HLS_estimated_BRAM18K_delta"],
            rom_payloads_identical=pair[
                "ROM_initialization_contents_identical_multiset"],
        ))
    return {
        "schema": SCHEMA,
        "original_R4a_source_revision": base["original_R4a_source_revision"],
        "original_R4d_source_revision": base["original_R4d_source_revision"],
        "prespecified_format_pairs": len(pairs),
        "source_default_pairs_interpretable": sum(
            p["status"] == "SOURCE_MODULE_DEFAULT_DIMENSIONS_ONLY"
            for p in pairs),
        "arms": arms,
        "pairs": pairs,
        "scientific_limits": [
            "Original SHA-attested Verilog module parameter defaults, NOT instance overrides.",
            "No Verilog parameter elaboration, netlist synthesis, memory packing or RAMB binding.",
            "AddressRange * DataWidth is ONLY a source declaration data-bit product.",
            "Same AddressWidth does NOT prove identical physical BRAM allocation.",
            "The 80-entry original HLS scratch default change must not be equated to 240 earlier source cells.",
            "Changed HLS schedule could independently affect DSP/LUT/FF and worst cycles.",
            "Unexecuted 16/6 and 32/8 new-format native arms remain UNKNOWN.",
            "No vendor tool calls, no physical E4, no original source modifications.",
        ],
        "native_vendor_calls_during_analysis": 0,
        "original_sources_modified": False,
        "physical_E4": None,
    }


def run(*, study: str | Path, original_study: str | Path,
        output: str | Path | None = None) -> dict:
    result = inspect(study=study, original_study=original_study)
    if output is None:
        return result
    target = Path(output)
    resolved = target.resolve()
    if any(resolved == root or root in resolved.parents for root in (
            Path(study).resolve(), Path(original_study).resolve())):
        raise ValueError("R4i output must be outside original study evidence")
    if target.exists() or target.is_symlink():
        raise FileExistsError("R4i output must be create-only")
    target.mkdir(parents=True, exist_ok=False)
    body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (target / "report.json").write_text(body)
    (target / "artifact_sha256.json").write_text(json.dumps(
        {"report.json": sha256(body.encode()).hexdigest()},
        sort_keys=True, indent=2) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--original-study", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(json.dumps(run(
        study=args.study, original_study=args.original_study,
        output=args.output), sort_keys=True, indent=2,
        allow_nan=False))


if __name__ == "__main__":
    main()
