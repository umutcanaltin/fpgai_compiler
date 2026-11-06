"""R4n: read-only source-hash-attested RTL memory inspection of R4l arms.

The source-controlled R4k/R4l pair passed E1, E2 and E3 on the user's
workstation. R4n uses the ORIGINAL Vitis-generated Verilog and ROM files,
provenance audited through R4l and R4g. It reuses the existing conservative
R2x/R4h/R4i/R4j lexical recognizers, but never elaborates a Verilog design.
No new Vitis, Vivado, simulation or synthesis calls; original files read only.

These are SOURCE DECLARATION observations, not physical RAMB/URAM counts,
netlist memory bindings, hardware-cycle schedules or achieved clock.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments import cst_r4g_rtl_memory_bridge as r4g
from fpgai_research.experiments import cst_r4h_hls_memory_source_crosswalk as r4h
from fpgai_research.experiments import cst_r4i_rtl_memory_parameter_defaults as r4i
from fpgai_research.experiments import cst_r4j_memory_instance_source_sites as r4j
from fpgai_research.experiments import cst_r4l_schedule_matched_storage_native as r4l
from fpgai_research.experiments import cst_r4e_numeric_format_native as r4e
from fpgai_research.experiments.cst_r2w_rtl_memory_forensics import _limited_text

SCHEMA = "fpgai-research/R4n-qualified-new-R4l-RTL-memory-source-audit/v1"
ARMS = ("full_storage", "row_storage")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RTL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+\-]*\.(?:v|sv)\Z")


def _signed_sources(pkg: Path, attested_rtl: dict) -> dict[str, str]:
    hashes = attested_rtl.get("rtl_file_sha256")
    if (attested_rtl.get("rtl_observation_status") !=
            "ATTESTED_STATIC_TYPE_GRAPH_ONLY"
            or not isinstance(hashes, dict) or not hashes
            or len(hashes) > 4096):
        raise ValueError("R4n original RTL inventory not qualified")
    root = (pkg / "fpgai_hls_project" / "solution1" / "syn" / "verilog")
    if any(part.is_symlink() for part in (
            root, root.parent, root.parent.parent, root.parent.parent.parent,
            pkg)):
        raise ValueError("R4n symlinked original RTL source root")
    if not root.is_dir():
        raise ValueError("R4n missing original Vitis RTL files")
    result = {}
    for name, digest in sorted(hashes.items()):
        if (not isinstance(name, str)
                or _RTL_NAME.fullmatch(name) is None
                or not isinstance(digest, str) or _SHA.fullmatch(digest) is None):
            raise ValueError("R4n invalid original source-inventory name/hash")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("R4n attested original RTL file missing or symlinked")
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("R4n original Vitis RTL source SHA mismatch")
        result[name] = _limited_text(path)
    return result


def _parameters(sources: dict[str, str], rtl: dict) -> dict:
    names = rtl.get("provisionally_reachable_memory_module_types")
    hashes = rtl.get("rtl_file_sha256")
    if not isinstance(names, list) or not isinstance(hashes, dict):
        raise ValueError("R4n original memory module type inventory missing")
    profiles = []
    for typename in names:
        if (not isinstance(typename, str)
                or re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", typename)
                   is None):
            raise ValueError("R4n invalid memory source module type")
        filename = typename + ".v"
        if filename not in sources:
            entry = {
                "module_type": typename,
                "source_status": "UNKNOWN_NO_SAME_NAMED_SOURCE_FILE",
                "source_sha256": None,
                "signature": None,
            }
        else:
            entry = {
                "module_type": typename,
                "source_status": "SOURCE_SHA_ATTESTED_TEXT_ONLY",
                "source_sha256": hashes[filename],
                "signature": r4h.memory_source_signatures(sources[filename]),
            }
        profiles.append(r4i.parameter_defaults(entry))
    return {
        "candidate_memory_module_count": len(names),
        "source_module_parameter_profiles": profiles,
        "source_parameter_defaults_known": {
            p["module_type"]: p for p in profiles
            if p["status"] == "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY"
        },
        "unknown_parameter_modules": [
            p["module_type"] for p in profiles
            if p["status"] != "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY"
        ],
    }


def _site_profiles(sources: dict[str, str], rtl: dict,
                   parameters: dict) -> dict:
    names = rtl.get("provisionally_reachable_memory_module_types")
    if not isinstance(names, list) or not names:
        return {
            "status": "UNKNOWN_NO_MEMORY_SOURCE_TYPES",
            "source_memory_sites": None,
            "scratch_source_sites": None,
            "source_override_diagnostics": None,
            "static_expected_memory_sites": None,
        }
    scan = r4j.scan_source_sites(sources, names)
    observations = scan.get("instances")
    result = {
        "status": scan["status"],
        "source_memory_sites": None,
        "scratch_source_sites": None,
        "source_override_diagnostics": scan.get("unknown_sites"),
        "static_expected_memory_sites": scan.get("r2x_expected_memory_sites"),
        "source_defparam_present_anywhere": scan.get(
            "source_defparam_present_anywhere"),
    }
    if observations is None:
        return result
    known = parameters["source_parameter_defaults_known"]
    expanded = []
    for site in observations:
        default = known.get(site["module_type"])
        literal_default = (
            {
                "DataWidth": default["default_DataWidth_bits"],
                "AddressWidth": default["default_AddressWidth_bits"],
                "AddressRange": default["default_AddressRange_entries"],
            }
            if default is not None else None
        )
        effective_source_literal = None
        if (literal_default is not None
                and scan["status"] ==
                    "SOURCE_INSTANCE_SITES_PARSED_WITHOUT_ELABORATION"
                and site["parameter_site_status"] in (
                    "NO_EXPLICIT_OVERRIDE_AT_SOURCE_SITE",
                    "SOURCE_LITERAL_NAMED_OVERRIDES_ONLY")
                and not scan.get("source_defparam_present_anywhere")):
            effective_source_literal = {
                **literal_default,
                **(site["literal_named_overrides"] or {}),
            }
            if (any(type(v) is not int or v <= 0
                    for v in effective_source_literal.values())
                    or effective_source_literal["AddressRange"] >
                        (1 << effective_source_literal["AddressWidth"])):
                effective_source_literal = None
        expanded.append({
            **site,
            "source_module_default_tuple": literal_default,
            "source_site_literal_parameter_tuple_if_supported":
                effective_source_literal,
            "effective_elaborated_instance_parameters": None,
        })
    result["source_memory_sites"] = expanded
    result["scratch_source_sites"] = [
        item for item in expanded
        if item["module_type"] == r4j._SCRATCH
    ]
    return result


def inspect_arm(pkg: Path, native: dict) -> dict:
    arm = native.get("arm")
    if arm not in ARMS:
        raise ValueError("R4n unexpected original R4l source arm")
    result = {
        "arm": arm,
        "status": "UNKNOWN_E2_OR_E3_NOT_QUALIFIED",
        "E2_qualified": native.get("E2_qualified") is True,
        "E3_qualified": native.get("E3_qualified") is True,
        "HLS_estimated_BRAM18K": (
            (native.get("HLS_estimated_resources") or {}).get("bram_18k")
            if native.get("E3_qualified") is True else None),
        "source_cpp_sha256": native.get("source_sha256"),
        "rtl_source_receipt": None,
        "original_RTL_file_count": None,
        "original_ROM_file_count": None,
        "original_ROM_content_sha256_multiset": None,
        "static_module_profile": None,
        "memory_parameters": None,
        "source_sites": None,
        "source_files_modified": False,
        "elaborated_instance_count": None,
        "mapped_BRAM18K": None,
        "physical_clock_ns": None,
    }
    if not (result["E2_qualified"] and result["E3_qualified"]):
        return result
    # R4g independently validates the Vitis-invocation source identity,
    # COMPLETE RTL and ROM digests, then reuses R2x static module parsing.
    rtl = r4g.inspect_arm(pkg, observation={
        "format": [32, 6],
        "arm": arm,
        "E3_qualified": True,
        "original_source_sha256": native["source_sha256"],
        "HLS_estimated_resources": native["HLS_estimated_resources"],
    })
    result["rtl_source_receipt"] = {
        "status": rtl["rtl_observation_status"],
        "rtl_file_sha256": rtl["rtl_file_sha256"],
        "rom_file_sha256": rtl["rom_file_sha256"],
    }
    result["original_RTL_file_count"] = rtl["rtl_file_count"]
    result["original_ROM_file_count"] = rtl["rom_file_count"]
    if rtl["rtl_observation_status"] != "ATTESTED_STATIC_TYPE_GRAPH_ONLY":
        result["status"] = "UNKNOWN_MISSING_OR_UNSUPPORTED_ATTESTED_RTL"
        return result
    sources = _signed_sources(pkg, rtl)
    parameters = _parameters(sources, rtl)
    sites = _site_profiles(sources, rtl, parameters)
    result.update({
        "status": (
            "ATTESTED_STATIC_SOURCE_AND_MEMORY_SITES_NOT_ELABORATED"
            if sites["status"] ==
                "SOURCE_INSTANCE_SITES_PARSED_WITHOUT_ELABORATION"
            else "ATTESTED_STATIC_SOURCE_WITH_UNKNOWN_MEMORY_SITES"),
        "original_ROM_content_sha256_multiset": dict(
            sorted(Counter(rtl["rom_file_sha256"].values()).items())),
        "static_module_profile": rtl["static"],
        "memory_parameters": parameters,
        "source_sites": sites,
    })
    return result


def compare_arms(full: dict, row: dict) -> dict:
    if full.get("arm") != "full_storage" or row.get("arm") != "row_storage":
        raise ValueError("R4n storage pair order/membership invalid")
    known = all(
        o["status"] == "ATTESTED_STATIC_SOURCE_AND_MEMORY_SITES_NOT_ELABORATED"
        for o in (full, row))
    result = {
        "status": "UNKNOWN_INCOMPLETE_OR_UNSUPPORTED_RTL_MEMORY_SITES",
        "both_RTL_SHA_attested": known,
        "ROM_content_sha_multiset_identical": None,
        "scratch_source_parameter_sites": None,
        "scratch_source_depth_delta_row_minus_full": None,
        "HLS_estimated_BRAM18K_delta_row_minus_full": (
            row["HLS_estimated_BRAM18K"] - full["HLS_estimated_BRAM18K"]
            if full["E3_qualified"] and row["E3_qualified"]
            and type(full["HLS_estimated_BRAM18K"]) is int
            and type(row["HLS_estimated_BRAM18K"]) is int else None),
        "static_module_type_deltas_row_minus_full": None,
        "mapped_BRAM18K_delta": None,
        "placed_routed_clock_delta": None,
    }
    if not known:
        return result
    pair = r4j.compare_scratch_source_sites(
        {
            "status": full["source_sites"]["status"],
            "scratch_source_sites": full["source_sites"]["scratch_source_sites"],
        },
        {
            "status": row["source_sites"]["status"],
            "scratch_source_sites": row["source_sites"]["scratch_source_sites"],
        },
    )
    result.update({
        "status": "SOURCE_ATTESTED_STATIC_RTL_MEMORY_PAIR_NOT_ELABORATED",
        "both_RTL_SHA_attested": True,
        "ROM_content_sha_multiset_identical": (
            full["original_ROM_content_sha256_multiset"] ==
            row["original_ROM_content_sha256_multiset"]),
        "scratch_source_parameter_sites": pair,
        "scratch_source_depth_delta_row_minus_full":
            pair["scratch_literal_source_site_depth_delta"],
        "static_module_type_deltas_row_minus_full": {
            key: (
                row["static_module_profile"][key] -
                full["static_module_profile"][key]
                if type(row["static_module_profile"].get(key)) is int
                and type(full["static_module_profile"].get(key)) is int
                else None)
            for key in r4g.STATIC_KEYS
        },
    })
    return result


def inspect(*, r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path) -> dict:
    # Even a qualified R4l E3 report alone is not enough for RTL source
    # claims: R4g separately audits SHA-attested original RTL/ROM outputs.
    native = r4l.audit_native(
        study=r4k_study, original_r4d=original_r4d,
        original_r4a=original_r4a)
    if (native.get("E1_matched_pair_accepted") is not True
            or native.get("paired_HLS_source_arms") != 2
            or not isinstance(native.get("observations"), list)
            or [o.get("arm") for o in native["observations"]] != list(ARMS)):
        raise ValueError("R4n complete original R4k/R4l matched population missing")
    base = Path(r4k_study)
    observations = [
        inspect_arm(base / arm / "vitis_source", native["observations"][i])
        for i, arm in enumerate(ARMS)
    ]
    result = {
        "schema": SCHEMA,
        "original_R4a_source_revision": native["original_R4a_source_revision"],
        "original_R4d_source_revision": native["original_R4d_source_revision"],
        "R4k_original_source_revision": native["source_revision"],
        "original_E2_E3_qualified_pair": native["both_E2_qualified"] and
                                          native["both_E3_qualified"],
        "R4l_original_arms": len(observations),
        "R4l_original_RTL_source_arms_attested": sum(
            o["status"] in (
                "ATTESTED_STATIC_SOURCE_AND_MEMORY_SITES_NOT_ELABORATED",
                "ATTESTED_STATIC_SOURCE_WITH_UNKNOWN_MEMORY_SITES")
            for o in observations),
        "arms": observations,
        "pair": compare_arms(*observations),
        "scientific_limits": [
            "Original HLS-generated Verilog module defaults and source instance declarations are NOT elaboration.",
            "Source memory sites do NOT certify actual RAMB counts or inferred BRAM ports.",
            "HLS 13-vs-13 BRAM18K estimate is independent of static source module counts.",
            "Source schedule equivalence does NOT imply identical HLS hardware-cycle schedule.",
            "Changed 32/6 source scratch capacity is distinct from physical FPGA storage allocation.",
            "No Vitis, Vivado, netlist, board-throughput, power or physical E4 measurement.",
            "Two R4l original native cases and the original R4a/R4d evidence remain unchanged.",
        ],
        "native_vendor_calls_during_read_only_analysis": 0,
        "original_evidence_modified": False,
        "physical_E4": None,
    }
    return result


def run(*, r4k_study: str | Path, original_r4d: str | Path,
        original_r4a: str | Path, output: str | Path | None = None) -> dict:
    data = inspect(r4k_study=r4k_study, original_r4d=original_r4d,
                   original_r4a=original_r4a)
    if output is None:
        return data
    target = Path(output)
    dest = target.resolve()
    original_roots = tuple(
        Path(p).resolve() for p in (r4k_study, original_r4d, original_r4a))
    if any(dest == root or root in dest.parents for root in original_roots):
        raise ValueError("R4n output must be outside original evidence")
    if target.exists() or target.is_symlink():
        raise FileExistsError("R4n output is create-only")
    target.mkdir(parents=True, exist_ok=False)
    body = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (target / "report.json").write_text(body)
    (target / "artifact_sha256.json").write_text(json.dumps({
        "report.json": sha256(body.encode()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4k-study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(json.dumps(run(
        r4k_study=args.r4k_study,
        original_r4d=args.original_r4d,
        original_r4a=args.original_r4a,
        output=args.output,
    ), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
