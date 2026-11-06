"""R4j: source-attributed HLS memory-instance parameter-site audit.

Read only. Reuses original R4g Vitis RTL/ROM SHA attestation and the R4i
SHA-attested module-default audit. Recognizes a bounded subset of Verilog
memory instance declarations, optionally including literal named parameter
overrides. It does NOT elaborate generate blocks, defparams, symbolic
expressions or hierarchical connectivity, or determine mapped RAMB counts.
Unsupported forms are UNKNOWN, never interpreted as default parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments import cst_r2x_rtl_hierarchy_analysis as r2x
from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments import cst_r4g_rtl_memory_bridge as r4g
from fpgai_research.experiments import cst_r4i_rtl_memory_parameter_defaults as r4i
from fpgai_research.experiments.cst_r2w_rtl_memory_forensics import _limited_text

SCHEMA = "fpgai-research/R4j-original-memory-instance-source-sites/v1"
_ID = r"[A-Za-z_$][A-Za-z0-9_$]*"
_NAME = re.compile(_ID, re.ASCII)
_NAMED = re.compile(
    r"\.\s*(DataWidth|AddressWidth|AddressRange)\s*\(\s*(\d+)\s*\)\s*",
    re.ASCII,
)
_LABELS = ("DataWidth", "AddressWidth", "AddressRange")
_SCRATCH = "fpgai_model_scratch_RAM_AUTO_1R1W"
_ACC = "fpgai_model_acc_RAM_AUTO_1R1W"
_MAX_SOURCE_CHARS = 12 * 1024 * 1024


def _paren(text: str, start: int) -> tuple[str, int] | None:
    """Return contents and character immediately after one balanced (...) pair."""
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:idx], idx + 1
            if depth < 0:
                return None
    return None


def _arguments(source: str) -> tuple[dict[str, int] | None, str]:
    """Only comma-separated, nonduplicate named decimal literal overrides."""
    overrides = {}
    pieces = source.split(",")
    if not pieces or len(pieces) > 16:
        return None, "UNKNOWN_UNSUPPORTED_PARAMETER_ARGUMENTS"
    for piece in pieces:
        match = _NAMED.fullmatch(piece.strip())
        if match is None or match[1] in overrides:
            return None, "UNKNOWN_SYMBOLIC_POSITIONAL_OR_DUPLICATE_OVERRIDE"
        value = int(match[2])
        if value <= 0 or value > 1_000_000:
            return None, "UNKNOWN_OUT_OF_BOUNDS_PARAMETER_OVERRIDE"
        overrides[match[1]] = value
    return overrides, "SOURCE_LITERAL_NAMED_OVERRIDES_ONLY"


def scan_source_sites(source_files: dict[str, str],
                      memory_types: list[str]) -> dict:
    """Scan only reachable module *types*, cross-checking the R2x edge list."""
    if (not isinstance(source_files, dict) or not source_files
            or len(source_files) > 4096
            or not isinstance(memory_types, list)
            or not memory_types or len(memory_types) > 128
            or len(set(memory_types)) != len(memory_types)
            or any(not isinstance(name, str) or
                   re.fullmatch(_ID, name) is None for name in memory_types)):
        raise ValueError("R4j requires a finite list of original memory types")
    if any(not isinstance(text, str) or len(text) > _MAX_SOURCE_CHARS
           for text in source_files.values()):
        raise ValueError("R4j original RTL source exceeds bounded scan")
    static = r2x.parse_static_modules(source_files, top="fpgai_model")
    if static["parse_status"] != "STATIC_TYPE_GRAPH_ONLY":
        return {
            "status": "UNKNOWN_OR_UNSUPPORTED_STATIC_RTL",
            "instances": None,
            "unknown_sites": None,
            "r2x_expected_memory_sites": None,
        }
    reachable = set(static["provisionally_reachable_module_types"])
    expected = Counter(
        (e["parent_type"], e["child_type"], e["instance"])
        for e in static["static_instance_edges"]
        if e["parent_type"] in reachable and e["child_type"] in memory_types
    )
    if not expected:
        return {
            "status": "UNKNOWN_NO_R2X_MEMORY_INSTANCE_SITES",
            "instances": None,
            "unknown_sites": None,
            "r2x_expected_memory_sites": 0,
        }
    candidates = []
    unknown = []
    # A defparam may change an otherwise unoverridden declaration elsewhere.
    # This cannot be resolved by a source-site lexical parser.
    contains_defparam = any(
        re.search(r"\bdefparam\b", r2x.strip_noncode(text)) is not None
        for text in source_files.values())
    for filename, source in sorted(source_files.items()):
        cleaned = r2x.strip_noncode(source)
        for match in r2x._MODULE.finditer(cleaned):
            parent = match.group("name")
            if parent not in reachable:
                continue
            body = match.group("body")
            header = r2x._header_end(body)
            if header is None:
                unknown.append({"parent_module_type": parent,
                                "reason": "UNKNOWN_MODULE_HEADER"})
                continue
            code = body[header:]
            for typename in memory_types:
                pattern = re.compile(
                    rf"(?<![A-Za-z0-9_$]){re.escape(typename)}"
                    rf"(?![A-Za-z0-9_$])")
                for hit in pattern.finditer(code):
                    offset = hit.end()
                    offset += len(code[offset:]) - len(code[offset:].lstrip())
                    mode = "NO_EXPLICIT_OVERRIDE_AT_SOURCE_SITE"
                    overrides = {}
                    if offset < len(code) and code[offset] == "#":
                        offset += 1
                        offset += len(code[offset:]) - len(code[offset:].lstrip())
                        param = _paren(code, offset)
                        if param is None:
                            unknown.append({
                                "parent_module_type": parent,
                                "module_type": typename,
                                "reason": "UNKNOWN_UNBALANCED_PARAMETER_GROUP"})
                            continue
                        text, offset = param
                        overrides, mode = _arguments(text)
                        offset += len(code[offset:]) - len(code[offset:].lstrip())
                    instance_match = _NAME.match(code, offset)
                    if instance_match is None:
                        unknown.append({
                            "parent_module_type": parent,
                            "module_type": typename,
                            "reason": "UNKNOWN_INSTANCE_IDENTIFIER"})
                        continue
                    instance = instance_match.group()
                    offset = instance_match.end()
                    offset += len(code[offset:]) - len(code[offset:].lstrip())
                    connections = _paren(code, offset)
                    if connections is None:
                        unknown.append({
                            "parent_module_type": parent,
                            "module_type": typename,
                            "instance": instance,
                            "reason": "UNKNOWN_INSTANCE_PORT_SYNTAX"})
                        continue
                    _, offset = connections
                    offset += len(code[offset:]) - len(code[offset:].lstrip())
                    if offset >= len(code) or code[offset] != ";":
                        unknown.append({
                            "parent_module_type": parent,
                            "module_type": typename,
                            "instance": instance,
                            "reason": "UNKNOWN_MULTI_INSTANCE_OR_STATEMENT_END"})
                        continue
                    # This is a textual module declaration site. It may
                    # occur in generate/conditional scopes; do not claim
                    # physical instance multiplicity or elaboration.
                    line = source[:match.start("body") + header + hit.start()].count("\n") + 1
                    candidates.append({
                        "source_file": filename,
                        "source_line": line,
                        "parent_module_type": parent,
                        "module_type": typename,
                        "instance": instance,
                        "parameter_site_status": mode,
                        "literal_named_overrides": overrides,
                        "conditional_generate_and_hierarchy_unresolved": True,
                        "source_defparam_present_anywhere": contains_defparam,
                        "elaborated_instances": None,
                        "physical_BRAM": None,
                    })
    matched = Counter(
        (item["parent_module_type"], item["module_type"], item["instance"])
        for item in candidates)
    if matched != expected:
        unknown.append({"reason": "UNKNOWN_R2X_INSTANCE_EDGE_MISMATCH",
                        "expected_candidate_sites": sum(expected.values()),
                        "matched_candidate_sites": sum(matched.values())})
    status = ("SOURCE_INSTANCE_SITES_PARSED_WITHOUT_ELABORATION"
              if not unknown and not contains_defparam and all(
                  c["parameter_site_status"] in (
                      "NO_EXPLICIT_OVERRIDE_AT_SOURCE_SITE",
                      "SOURCE_LITERAL_NAMED_OVERRIDES_ONLY") for c in candidates)
              else "UNKNOWN_INCOMPLETE_OR_UNRESOLVED_INSTANCE_SITES")
    return {
        "status": status,
        "instances": sorted(candidates, key=lambda r: (
            r["module_type"], r["parent_module_type"], r["instance"],
            r["source_file"], r["source_line"])),
        "unknown_sites": unknown,
        "r2x_expected_memory_sites": sum(expected.values()),
        "source_defparam_present_anywhere": contains_defparam,
    }


def inspect_arm(study: Path, rtl: dict, defaults: dict) -> dict:
    fmt, arm = rtl["format"], rtl["arm"]
    if defaults.get("format") != fmt or defaults.get("arm") != arm:
        raise ValueError("R4j mismatched original R4g/R4i arm identity")
    report = {
        "format": fmt, "arm": arm,
        "status": "UNKNOWN_NO_QUALIFIED_OR_ATTESTED_SOURCE",
        "original_RTL_file_count": None,
        "source_memory_sites": None,
        "scratch_source_sites": None,
        "acc_source_sites": None,
        "source_override_diagnostics": None,
        "source_site_default_tuple": None,
        "source_site_literal_parameter_tuples": None,
        "elaborated_memory_capacity": None,
        "mapped_BRAM": None,
    }
    if (rtl["rtl_observation_status"] != "ATTESTED_STATIC_TYPE_GRAPH_ONLY"
            or defaults["status"] != "SOURCE_PARAMETER_DEFAULTS_WITH_EXPLICIT_UNKNOWNS"):
        return report
    source_sha = rtl["rtl_file_sha256"]
    memory_types = rtl["provisionally_reachable_memory_module_types"]
    if not isinstance(source_sha, dict):
        raise ValueError("R4j original RTL SHA inventory unavailable")
    bits, integer = fmt
    root = (study / f"format_{bits:02d}_{integer:02d}" / arm
            / "vitis_source" / "fpgai_hls_project" / "solution1"
            / "syn" / "verilog")
    if any(path.is_symlink() for path in (
            root, root.parent, root.parent.parent, root.parent.parent.parent,
            root.parent.parent.parent.parent)):
        raise ValueError("R4j symlinked original HLS RTL ancestor")
    if not root.is_dir():
        raise ValueError("R4j originally attested RTL source directory missing")
    source_files = {}
    for name, digest in sorted(source_sha.items()):
        if (not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*\.(?:v|sv)",name)
                   is None):
            raise ValueError("R4j unsafe original RTL filename")
        file = root / name
        if file.is_symlink() or not file.is_file():
            raise ValueError("R4j signed RTL file absent or symlinked")
        raw = file.read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError("R4j original RTL SHA mismatch")
        source_files[name] = _limited_text(file)
    sites = scan_source_sites(source_files, memory_types)
    report["original_RTL_file_count"] = len(source_files)
    report["status"] = sites["status"]
    report["source_override_diagnostics"] = sites.get("unknown_sites")
    if sites["instances"] is None:
        return report
    originals = defaults["memory_default_profiles"]
    listed = []
    for site in sites["instances"]:
        profile = originals.get(site["module_type"]) if isinstance(originals, dict) else None
        defaults_tuple = None
        if profile and profile["status"] == "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY":
            defaults_tuple = {
                "DataWidth": profile["default_DataWidth_bits"],
                "AddressWidth": profile["default_AddressWidth_bits"],
                "AddressRange": profile["default_AddressRange_entries"],
            }
        literal = None
        if (defaults_tuple is not None
                and site["parameter_site_status"] in (
                    "NO_EXPLICIT_OVERRIDE_AT_SOURCE_SITE",
                    "SOURCE_LITERAL_NAMED_OVERRIDES_ONLY")
                and not sites["source_defparam_present_anywhere"]):
            literal = {**defaults_tuple, **(site["literal_named_overrides"] or {})}
            if (literal["AddressRange"] > (1 << literal["AddressWidth"])
                    or any(v <= 0 for v in literal.values())):
                literal = None
        listed.append({**site,
                       "source_module_default_tuple": defaults_tuple,
                       "source_site_literal_parameter_tuple_if_supported": literal,
                       "effective_instance_parameter_tuple": None})
    scratch = [s for s in listed if s["module_type"] == _SCRATCH]
    acc = [s for s in listed if s["module_type"] == _ACC]
    report.update({
        "source_memory_sites": listed,
        "scratch_source_sites": scratch,
        "acc_source_sites": acc,
        "source_site_default_tuple": (
            scratch[0]["source_module_default_tuple"]
            if scratch and scratch[0]["source_module_default_tuple"] is not None
            and all(site["source_module_default_tuple"] ==
                    scratch[0]["source_module_default_tuple"]
                    for site in scratch) else None),
        "source_site_literal_parameter_tuples": (
            [s["source_site_literal_parameter_tuple_if_supported"] for s in scratch]
            if scratch else None),
    })
    return report


def compare_scratch_source_sites(full: dict, row: dict) -> dict:
    """Pair original scratch source DECLARATION sites by exact parent/name.

    A source site is not an elaborated hardware instance. The scalar
    depth delta is reported only if every paired site has a literal,
    identical-within-arm tuple and both arms use the same word width
    and address width. No physical memory quantity is inferred.
    """
    left = full.get("scratch_source_sites")
    right = row.get("scratch_source_sites")
    result = {
        "status": "UNKNOWN_SCRATCH_INSTANCE_PARAMETER_SITES",
        "scratch_source_site_count_full": len(left) if isinstance(left, list) else None,
        "scratch_source_site_count_row": len(right) if isinstance(right, list) else None,
        "scratch_source_site_pairs": None,
        "scratch_literal_source_site_tuple_full": None,
        "scratch_literal_source_site_tuple_row": None,
        "scratch_site_parameter_modes": None,
        "scratch_literal_source_site_depth_delta": None,
        "elaborated_scratch_instance_depth_delta": None,
        "physical_BRAM_delta": None,
    }
    if (full.get("status") != "SOURCE_INSTANCE_SITES_PARSED_WITHOUT_ELABORATION"
            or row.get("status") != "SOURCE_INSTANCE_SITES_PARSED_WITHOUT_ELABORATION"
            or not isinstance(left, list) or not isinstance(right, list)
            or not left or not right):
        return result

    def indexed(sites):
        keyed = {}
        for site in sites:
            if not isinstance(site, dict):
                return None
            key = (site.get("parent_module_type"),
                   site.get("module_type"), site.get("instance"))
            if (any(not isinstance(v, str) or not v for v in key)
                    or key in keyed):
                return None
            keyed[key] = site
        return keyed

    a = indexed(left)
    b = indexed(right)
    if a is None or b is None or set(a) != set(b):
        return result
    observations = []
    for key in sorted(a):
        before = a[key].get("source_site_literal_parameter_tuple_if_supported")
        after = b[key].get("source_site_literal_parameter_tuple_if_supported")
        if (not isinstance(before, dict) or not isinstance(after, dict)
                or set(before) != {"DataWidth", "AddressWidth", "AddressRange"}
                or set(after) != {"DataWidth", "AddressWidth", "AddressRange"}
                or any(type(item) is not int or item <= 0
                       for item in (*before.values(), *after.values()))):
            return result
        observations.append({
            "parent_module_type": key[0],
            "module_type": key[1],
            "instance": key[2],
            "source_literal_parameters_full": before,
            "source_literal_parameters_row": after,
            "parameter_mode_full": a[key]["parameter_site_status"],
            "parameter_mode_row": b[key]["parameter_site_status"],
            "source_depth_delta_row_minus_full": (
                after["AddressRange"] - before["AddressRange"]),
            "elaborated_instance_count": None,
            "physical_BRAM_delta": None,
        })
    result["scratch_source_site_pairs"] = observations
    # Do NOT collapse heterogeneous site dimensions to a single summary
    # or sum source capacities as if they were physical RAM instances.
    full_tuples = [v["source_literal_parameters_full"] for v in observations]
    row_tuples = [v["source_literal_parameters_row"] for v in observations]
    uniform = (
        all(item == full_tuples[0] for item in full_tuples)
        and all(item == row_tuples[0] for item in row_tuples)
        and full_tuples[0]["DataWidth"] == row_tuples[0]["DataWidth"]
        and full_tuples[0]["AddressWidth"] == row_tuples[0]["AddressWidth"]
    )
    if not uniform:
        result["status"] = "SOURCE_SCRATCH_SITES_PAIRED_HETEROGENEOUS_NOT_ELABORATED"
        return result
    result.update({
        "status": "SOURCE_SCRATCH_SITES_PARSED_NOT_ELABORATED",
        "scratch_literal_source_site_tuple_full": full_tuples[0],
        "scratch_literal_source_site_tuple_row": row_tuples[0],
        "scratch_site_parameter_modes": (
            [observations[0]["parameter_mode_full"],
             observations[0]["parameter_mode_row"]]
            if all(item["parameter_mode_full"] ==
                   observations[0]["parameter_mode_full"] and
                   item["parameter_mode_row"] ==
                   observations[0]["parameter_mode_row"]
                   for item in observations) else None),
        "scratch_literal_source_site_depth_delta": (
            row_tuples[0]["AddressRange"] - full_tuples[0]["AddressRange"]),
    })
    return result


def inspect(*, study: str | Path, original_study: str | Path) -> dict:
    # Independent pipelines: R4g verifies original native HDL/ROM receipts;
    # R4i verifies same original source module default declarations.
    g = r4g.inspect(study=study, original_study=original_study)
    i = r4i.inspect(study=study, original_study=original_study)
    if (g["original_R4a_source_revision"] != i["original_R4a_source_revision"]
            or g["original_R4d_source_revision"] != i["original_R4d_source_revision"]
            or g["prespecified_pairs"] != len(r4d.FORMATS)
            or i["prespecified_format_pairs"] != len(r4d.FORMATS)
            or len(g["arms"]) != 2 * len(r4d.FORMATS)
            or len(i["arms"]) != 2 * len(r4d.FORMATS)):
        raise ValueError("R4j original research source population mismatch")
    observations = []
    pairs = []
    for idx, fmt in enumerate(r4d.FORMATS):
        a = inspect_arm(Path(study), g["arms"][2*idx], i["arms"][2*idx])
        b = inspect_arm(Path(study), g["arms"][2*idx + 1], i["arms"][2*idx + 1])
        observations.extend((a, b))
        paired = compare_scratch_source_sites(a, b)
        pair = {
            "format": list(fmt),
            "HLS_estimated_BRAM18K_delta": g["pairs"][idx][
                "HLS_estimated_BRAM18K_delta_row_minus_full"],
            **paired,
        }
        pairs.append(pair)
    return {
        "schema": SCHEMA,
        "original_R4a_source_revision": g["original_R4a_source_revision"],
        "original_R4d_source_revision": g["original_R4d_source_revision"],
        "prespecified_format_pairs": len(pairs),
        "original_RTL_source_arms": sum(
            a["status"] == "SOURCE_INSTANCE_SITES_PARSED_WITHOUT_ELABORATION"
            for a in observations),
        "arms": observations,
        "pairs": pairs,
        "scientific_limits": [
            "A source module instance declaration is NOT elaborated hardware.",
            "Three original scratch source sites are not proof of three mapped RAM cells.",
            "A per-site 80-entry reduction is not automatically a 240-element physical saving.",
            "Pairing by source instance name does not establish generate scope multiplicity.",
            "Generate conditionals, hierarchical multiplicity and port connectivity are not resolved.",
            "Literal named parameter overrides may be visible, but defparams and symbolic/positional forms are UNKNOWN.",
            "No proof of mapped RAMB granularity or actual SRAM occupancy.",
            "Source scratch default-depth change of 80 entries is distinct from source-IR 240 cell saving.",
            "HLS BRAM estimate 13 vs 13 is independent of source-site parameter analysis.",
            "No native tool calls, no physical E4 and no original evidence writes.",
        ],
        "native_vendor_calls_during_analysis": 0,
        "physical_E4": None,
    }


def run(*, study: str | Path, original_study: str | Path,
        output: str | Path | None = None) -> dict:
    result = inspect(study=study, original_study=original_study)
    if output is None:
        return result
    dest = Path(output)
    resolved = dest.resolve()
    if any(resolved == root or root in resolved.parents for root in (
            Path(study).resolve(), Path(original_study).resolve())):
        raise ValueError("R4j output must be outside original evidence")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R4j output must be create-only")
    dest.mkdir(parents=True, exist_ok=False)
    body = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (dest / "report.json").write_text(body)
    (dest / "artifact_sha256.json").write_text(json.dumps(
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
        output=args.output), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
