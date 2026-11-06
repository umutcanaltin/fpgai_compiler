"""R4g: source-attested, READ-ONLY static RTL memory analysis of R4e arms.

Reuses existing R2x Verilog static type-graph parser and canonical Vitis
RTL/ROM inventory; does NOT parse/elaborate the entire Verilog language.
The returned module types and candidate primitive names are NOT physical
RAMB/URAM usage or post-synthesis cells. Missing or ambiguous HDL is
UNKNOWN, never zero memory. No native tool is invoked.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments import cst_r4e_numeric_format_native as r4e
from fpgai_research.experiments import cst_r2x_rtl_hierarchy_analysis as r2x
from fpgai_research.experiments.cst_r2w_rtl_memory_forensics import _limited_text
from fpgai_research.toolchains.vivado.rtl_inputs import inventory_vitis_rtl

SCHEMA = "fpgai-research/R4g-original-32bit-HLS-static-RTL-memory-bridge/v1"
STATIC_KEYS = (
    "reachable_module_type_count",
    "reachable_memory_module_type_count",
    "reachable_memory_module_type_instances",
    "reachable_scratch_module_type_count",
    "candidate_reachable_primitive_instantiations",
    "reachable_reg_memory_candidate_count",
)


def inspect_arm(pkg: Path, *, observation: dict) -> dict:
    """Check native invocation's exact RTL and ROM hashes before parsing."""
    if not isinstance(observation, dict):
        raise ValueError("R4g invalid native audit observation")
    result = {
        "format": observation.get("format"),
        "arm": observation.get("arm"),
        "E3_qualified": observation.get("E3_qualified") is True,
        "HLS_estimated_BRAM18K": (
            (observation.get("HLS_estimated_resources") or {}).get("bram_18k")
            if observation.get("E3_qualified") is True else None),
        "rtl_sha_attested": False,
        "rtl_observation_status": "UNKNOWN_E3_NOT_QUALIFIED",
        "rtl_file_count": None,
        "rom_file_count": None,
        "rtl_file_sha256": None,
        "rom_file_sha256": None,
        "source_module_parse_status": None,
        "static": {key: None for key in STATIC_KEYS},
        "provisionally_reachable_scratch_module_types": None,
        "provisionally_reachable_memory_module_types": None,
        "original_RTL_case_modified": False,
        "physical_BRAM": None,
        "physical_achieved_Fmax": None,
    }
    if observation.get("E3_qualified") is not True:
        return result
    if pkg.is_symlink() or not pkg.is_dir():
        raise ValueError("R4g qualified original native case path missing/symlinked")
    receipt_file = pkg / "vitis_invocation.json"
    if receipt_file.is_symlink() or not receipt_file.is_file():
        raise ValueError("R4g qualified original invocation missing/symlinked")
    receipt = json.loads(_limited_text(receipt_file))
    if (not isinstance(receipt, dict)
            or receipt.get("tool") != "vitis_hls"
            or receipt.get("schema") != "fpgai-research/vendor-invocation/v1"
            or receipt.get("source_sha256")
                != observation.get("original_source_sha256")
            or receipt.get("completed_successfully") is not True
            or receipt.get("inputs_stable_during_execution") is not True):
        raise ValueError("R4g source-attested original vendor receipt invalid")
    rtl_root = pkg / "fpgai_hls_project" / "solution1" / "syn" / "verilog"
    if rtl_root.is_symlink():
        raise ValueError("R4g symlinked native RTL directory")
    if not rtl_root.is_dir():
        result["rtl_observation_status"] = "UNKNOWN_MISSING_GENERATED_RTL"
        return result
    expected_rtl = receipt.get("rtl_output_sha256")
    expected_rom = receipt.get("rom_output_sha256")
    if (not isinstance(expected_rtl, dict) or not expected_rtl
            or not isinstance(expected_rom, dict)):
        result["rtl_observation_status"] = "UNKNOWN_NO_NATIVE_RTL_SHA_RECEIPT"
        return result
    actual = inventory_vitis_rtl(pkg)
    if (actual["rtl_sha256"] != expected_rtl
            or actual["rom_sha256"] != expected_rom):
        raise ValueError("R4g original Vitis RTL/ROM SHA mismatch")
    sources = {}
    for filename in sorted(actual["rtl_sha256"]):
        sources[filename] = _limited_text(rtl_root / filename)
    # The R2x parser explicitly withholds reachability when unsupported
    # syntax appears. Never infer a physical memory count from its output.
    parsed = r2x.parse_static_modules(sources, top="fpgai_model")
    valid = parsed["parse_status"] == "STATIC_TYPE_GRAPH_ONLY"
    result.update({
        "rtl_sha_attested": True,
        "rtl_observation_status": (
            "ATTESTED_STATIC_TYPE_GRAPH_ONLY" if valid
            else "UNKNOWN_UNSUPPORTED_STATIC_VERILOG"),
        "rtl_file_count": len(actual["rtl_sha256"]),
        "rom_file_count": len(actual["rom_sha256"]),
        "rtl_file_sha256": actual["rtl_sha256"],
        "rom_file_sha256": actual["rom_sha256"],
        "source_module_parse_status": parsed["parse_status"],
        "static": {
            key: parsed[key] if valid else None for key in STATIC_KEYS},
        "provisionally_reachable_scratch_module_types": (
            parsed["provisionally_reachable_scratch_module_types"]
            if valid else None),
        "provisionally_reachable_memory_module_types": (
            parsed["provisionally_reachable_memory_module_types"]
            if valid else None),
    })
    return result


def compare_pair(full: dict, row: dict, *, format: list[int]) -> dict:
    if (full.get("format") != format or row.get("format") != format
            or full.get("arm") != "full_H"
            or row.get("arm") != "row_H"):
        raise ValueError("R4g unmatched source-format/arm comparison")
    both_e3 = full["E3_qualified"] and row["E3_qualified"]
    both_static = (both_e3
                   and full["rtl_observation_status"]
                       == "ATTESTED_STATIC_TYPE_GRAPH_ONLY"
                   and row["rtl_observation_status"]
                       == "ATTESTED_STATIC_TYPE_GRAPH_ONLY")
    a_bram = full["HLS_estimated_BRAM18K"]
    b_bram = row["HLS_estimated_BRAM18K"]
    if both_e3 and (type(a_bram) is not int or type(b_bram) is not int):
        raise ValueError("R4g qualified E3 comparison lacks BRAM estimate")
    return {
        "format": format,
        "both_E3_qualified": both_e3,
        "both_RTL_SHA_attested": (
            full["rtl_sha_attested"] and row["rtl_sha_attested"]),
        "both_static_type_graphs_known": both_static,
        "status": ("STATIC_RTL_HLS_ESTIMATES_ONLY" if both_static
                   else "UNKNOWN_MISSING_OR_UNSUPPORTED_STATIC_RTL"),
        "HLS_estimated_BRAM18K_delta_row_minus_full": (
            b_bram - a_bram if both_e3 else None),
        "candidate_RTL_structure_delta_row_minus_full": {
            key: (row["static"][key] - full["static"][key])
                 if both_static and type(row["static"][key]) is int
                 and type(full["static"][key]) is int else None
            for key in STATIC_KEYS
        },
        "physical_BRAM_delta": None,
        "physical_clock_delta": None,
    }


def inspect(*, study: str | Path, original_study: str | Path) -> dict:
    """Read-only full three-format population, with original R4e audit."""
    native = r4e.audit_native(study=study, original_study=original_study)
    if native.get("source_attested_arms") != 6:
        raise ValueError("R4g incomplete original R4d population")
    observations = native.get("observations")
    if not isinstance(observations, list) or len(observations) != 6:
        raise ValueError("R4g original R4e observations incomplete")
    root = Path(study)
    rows, pairs = [], []
    for i, (bits, integer) in enumerate(r4d.FORMATS):
        folder = f"format_{bits:02d}_{integer:02d}"
        selected = []
        for j, arm in enumerate(("full_H", "row_H")):
            record = observations[2 * i + j]
            if record.get("format") != [bits, integer] or record.get("arm") != arm:
                raise ValueError("R4g original format/arm order inconsistent")
            result = inspect_arm(root / folder / arm / "vitis_source",
                                 observation=record)
            rows.append(result)
            selected.append(result)
        pairs.append(compare_pair(*selected, format=[bits, integer]))
    return {
        "schema": SCHEMA,
        "original_R4d_source_revision": native["source_revision"],
        "original_R4a_source_revision": native["original_R4a_source_revision"],
        "prespecified_pairs": 3,
        "native_E3_qualified_pairs": native["E3_qualified_pairs"],
        "original_RTL_SHA_attested_arms": sum(
            arm["rtl_sha_attested"] for arm in rows),
        "static_type_graph_pairs_known": sum(
            pair["both_static_type_graphs_known"] for pair in pairs),
        "arms": rows,
        "pairs": pairs,
        "interpretation_limits": [
            "A static module-type graph is not an elaborated or synthesized netlist.",
            "Named scratch/RAM source modules are not physical RAMB resources.",
            "Unrecognized generated RTL makes structure UNKNOWN, not zero.",
            "HLS BRAM18K estimates are not placed/routed BRAM counts.",
            "No Vitis, Vivado, simulation or native compilation was launched.",
            "No conclusion that scratch saving alone caused logic differences.",
        ],
        "original_files_modified": False,
        "new_native_Vitis_calls": 0,
        "new_native_Vivado_calls": 0,
        "physical_E4": None,
    }


def run(*, study: str | Path, original_study: str | Path,
        output: str | Path | None = None) -> dict:
    data = inspect(study=study, original_study=original_study)
    if output is not None:
        target = Path(output)
        resolved = target.resolve()
        for root in (Path(study).resolve(), Path(original_study).resolve()):
            if resolved == root or root in resolved.parents:
                raise ValueError("R4g output must not be inside original evidence")
        if target.exists() or target.is_symlink():
            raise FileExistsError("R4g report output must be create-only")
        target.mkdir(parents=True, exist_ok=False)
        report = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
        (target / "report.json").write_text(report)
        (target / "artifact_sha256.json").write_text(json.dumps({
            "report.json": sha256(report.encode()).hexdigest()
        }, indent=2, sort_keys=True) + "\n")
    return data


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--original-study", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = run(study=args.study, original_study=args.original_study,
                 output=args.output)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
