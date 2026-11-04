"""Re-audit two native hls4ml E1 source projects for *effective* single-knob isolation.

This is an immutable, read-only source-level audit. It does not run Vitis,
check native arithmetic or imply a cross-compiler performance advantage.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory


SCHEMA = "fpgai-research/hls4ml-effective-pair-audit/v1"
RECEIPT_SCHEMA = "fpgai-research/hls4ml-baseline/v2"
DIGEST = set("0123456789abcdef")


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= DIGEST


def _load_receipt(p):
    p = Path(p)
    if p.is_symlink() or not p.is_file() or not 0 < p.stat().st_size <= 2 * 1024 * 1024:
        raise ValueError("Missing ordinary bounded E1 receipt")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if (not isinstance(obj, dict) or obj.get("schema") != RECEIPT_SCHEMA
            or obj.get("evidence") != "E1-source-only"
            or not _sha(obj.get("model_sha256"))
            or obj.get("hls4ml_version") != "1.3.0"
            or obj.get("backend") != "Vitis"
            or obj.get("io_type") != "io_parallel"
            or obj.get("native_synthesis") != "NOT_EXECUTED"):
        raise ValueError("E1 source provenance does not match pinned hls4ml 1.3.0 Vitis baseline")
    if (not isinstance(obj.get("hls_config"), dict)
            or not isinstance(obj.get("hls_config_requested_before_conversion"), dict)):
        raise ValueError("Missing requested/effective hls4ml configurations")
    if (obj.get("hls_config_changed_during_conversion")
            is not (obj["hls_config"] != obj["hls_config_requested_before_conversion"])):
        raise ValueError("Saved conversion mutation flag is inconsistent")
    original = Path(obj.get("model_path", ""))
    if (original.is_symlink() or not original.is_file()
            or not 0 < original.stat().st_size <= 128 * 1024 * 1024
            or sha256(original.read_bytes()).hexdigest() != obj["model_sha256"]):
        raise ValueError("Actual ONNX source differs from pinned E1 receipt")
    source_files = obj.get("generated_cpp_source_inventory")
    if not isinstance(source_files, list) or not 1 <= len(source_files) <= 128:
        raise ValueError("Expected bounded generated C++ inventory")
    digest_map = {}
    for item in source_files:
        if not isinstance(item, dict):
            raise ValueError("Invalid C++ inventory entry")
        name, digest = item.get("relative_path"), item.get("sha256")
        if not isinstance(name, str) or not _sha(digest):
            raise ValueError("C++ file identity or hash missing")
        rel = Path(name)
        if (rel.is_absolute() or ".." in rel.parts or not rel.parts
                or rel.suffix not in (".cpp", ".cc", ".cxx")
                or name in digest_map):
            raise ValueError("Unsafe or duplicated generated C++ name")
        file = p.parent / rel
        if file.is_symlink() or not file.is_file() or sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError("Generated C++ does not match E1 inventory")
        digest_map[name] = digest
    # Generated HLS implementation controls can reside in model-owned
    # headers, even when every .cpp file remains byte-identical.
    from fpgai_research.experiments.hls4ml_codegen_delta import _model_files
    _, firmware = _model_files(p.parent)
    return obj, sha256(p.read_bytes()).hexdigest(), digest_map, firmware


def _leaves(obj, prefix=()):
    if isinstance(obj, dict):
        if not obj:
            yield prefix, {}
        for key, value in obj.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Non-string/empty hls4ml configuration key")
            yield from _leaves(value, prefix + (key,))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _leaves(value, prefix + (str(i),))
    elif obj is None or type(obj) in (str, int, float, bool):
        yield prefix, obj
    else:
        raise ValueError("Unsupported hls4ml config value")


def _diff(left, right):
    a, b = dict(_leaves(left)), dict(_leaves(right))
    return [{"path": list(key),
             "a": a.get(key, "<ABSENT>"),
             "b": b.get(key, "<ABSENT>")}
            for key in sorted(a.keys() | b.keys()) if a.get(key, "<ABSENT>") != b.get(key, "<ABSENT>")]


def audit_hls4ml_effective_pair(a_receipt, b_receipt, expected_path):
    if (not isinstance(expected_path, list) or not 2 <= len(expected_path) <= 5
            or any(not isinstance(x, str) or not x for x in expected_path)):
        raise ValueError("Explicit expected effective configuration path required")
    a, ah, acpp, afirmware = _load_receipt(a_receipt)
    b, bh, bcpp, bfirmware = _load_receipt(b_receipt)
    control_fields = ("model_sha256", "hls4ml_version", "onnx_version", "part",
                      "clock_ns", "precision", "reuse_factor", "backend", "io_type",
                      "derived_candidate_full_e0_lineage", "configuration_granularity",
                      "name_config_policy")
    differences = [name for name in control_fields if a.get(name) != b.get(name)]
    requested = _diff(a["hls_config_requested_before_conversion"],
                      b["hls_config_requested_before_conversion"])
    effective = _diff(a["hls_config"], b["hls_config"])
    expected = tuple(expected_path)
    one_requested = len(requested) == 1 and tuple(requested[0]["path"]) == expected
    one_effective = len(effective) == 1 and tuple(effective[0]["path"]) == expected
    isolated = not differences and one_requested and one_effective
    cpp_changed = sorted({*acpp, *bcpp} -
                         {name for name in acpp if acpp.get(name) == bcpp.get(name)})
    firmware_changed = sorted(
        name for name in (set(afirmware) | set(bfirmware))
        if afirmware.get(name) != bfirmware.get(name)
    )
    headers_changed = [
        name for name in firmware_changed
        if Path(name).suffix.lower() in (".h", ".hpp")
    ]
    # A request may update multiple converter-layer aliases and yet leave
    # emitted computational C++ byte-identical. The alias relationship
    # is only a hypothesis: it requires independently verified graph/RTL
    # structure before attributing a hardware effect.
    possible_name_alias = bool(
        not differences and one_requested and len(effective) == 2
        and all(row["path"][:1] == ["LayerName"] and
                row["path"][-1] == "ReuseFactor" and
                row["a"] == requested[0]["a"] and
                row["b"] == requested[0]["b"]
                for row in effective)
        and any(row["path"][-2] == "Dense_" + expected[-2] for row in effective)
        and any(row["path"][-2] == expected[-2] for row in effective)
    )
    return {
        "schema": SCHEMA,
        "evidence": "E1-generated-source-config-only",
        "a_receipt_sha256": ah,
        "b_receipt_sha256": bh,
        "common_original_model_source_sha256": a["model_sha256"] if not differences else None,
        "expected_changed_path": expected_path,
        "nonmatching_experimental_controls": differences,
        "requested_differences": requested,
        "effective_differences": effective,
        "effective_single_knob_isolated": isolated,
        "confounded_extra_effective_changes": bool(len(effective) != 1 or not one_effective),
        "generated_cpp_files_changed": cpp_changed,
        "generated_cpp_file_change_count": len(cpp_changed),
        "generated_firmware_inventory_matches": set(afirmware)==set(bfirmware),
        "generated_firmware_file_change_count": len(firmware_changed),
        "generated_header_file_change_count": len(headers_changed),
        "generated_firmware_file_changes_observed": bool(firmware_changed),
        "model_header_changes_do_not_prove_causal_hardware_effect": True,
        "requested_one_effective_two_name_alias_candidates": possible_name_alias,
        "effective_config_changed_without_generated_cpp_change":
            bool(effective and not cpp_changed),
        "alias_identity_or_hardware_effect_proven": False,
        "same_source_model_weights_controlled_at_e1": "model_sha256" not in differences,
        "fpga_internal_parallelism_or_reuse_equivalence_proven": False,
        "numerical_e2_qualified": False,
        "vendor_e3_or_e4_executed": False,
        "cross_compiler_superiority_eligible": False,
    }


def export_hls4ml_effective_pair(a_receipt, b_receipt, expected_path, output):
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("E1 audit output must be new")
    result = audit_hls4ml_effective_pair(a_receipt, b_receipt, expected_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hls4ml-effective-audit-", dir=target.parent) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        (stage / "effective_pair.json").write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("E1 audit destination appeared")
        stage.rename(target)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="Baseline E1 research_baseline.json")
    p.add_argument("--b", required=True, help="Intervention E1 research_baseline.json")
    p.add_argument("--expected-path-json", required=True,
                   help='JSON list such as ["LayerName","matmul","ReuseFactor"]')
    p.add_argument("--output", required=True, help="Fresh immutable audit folder")
    args = p.parse_args(argv)
    expected = json.loads(args.expected_path_json)
    result = export_hls4ml_effective_pair(args.a, args.b, expected, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
