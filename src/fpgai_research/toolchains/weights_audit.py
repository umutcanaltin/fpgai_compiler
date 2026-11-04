"""Read-only comparison of ONNX float32 constants with generated FPGAI HLS C++.

This checks the exact emitted initializer literals, original model provenance,
and source manifest; it does NOT prove post-synthesis memory-cell contents,
bit-exact fixed-point values, or any physical FPGA deployment.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import sys


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked source provenance file: {path.name}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Invalid source provenance document: {path.name}")
    return result


def _payload_digest(constants) -> str:
    """Hash typed, named float32 payloads without exposing their raw values."""
    digest = sha256()
    for tensor in constants:
        encoded = tensor.name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(tensor.shape).to_bytes(4, "little"))
        for dim in tensor.shape:
            digest.update(int(dim).to_bytes(8, "little"))
        for value in tensor.data:
            digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def audit_weight_constants(onnx_path: str | Path, case_dir: str | Path) -> dict:
    """Fail closed on wrong ONNX file, edited source, or changed initializer values.

    Compare each full initializer text line against the *canonical emitter*
    for the ONNX Model IR tensor order. No regex approximation of individual
    numbers; even a one-digit changed constant makes the audit fail.
    """
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.model_ir.semantics import functional_fingerprint

    source_model = Path(onnx_path).expanduser().resolve(strict=True)
    case = Path(case_dir).expanduser()
    if case.is_symlink() or not case.is_dir():
        raise ValueError("Expected existing non-symlinked case directory")
    case = case.resolve(strict=True)
    manifest = _json(case / "manifest.json")
    provenance = _json(case / "inputs_provenance.json")
    cpp = case / "model.cpp"
    if cpp.is_symlink() or not cpp.is_file():
        raise ValueError("Missing or symlinked generated model.cpp")
    if provenance.get("onnx_sha256") != sha256(source_model.read_bytes()).hexdigest():
        raise ValueError("Supplied ONNX file differs from the source recorded for this case")
    source_bytes = cpp.read_bytes()
    source_hash = sha256(source_bytes).hexdigest()
    if manifest.get("cpp_sha256") != source_hash:
        raise ValueError("Generated C++ has changed since its original emission receipt")
    model = load_onnx(str(source_model))
    if manifest.get("functional_sha256") != functional_fingerprint(model):
        raise ValueError("Imported ONNX graph or constant payload disagrees with case manifest")
    width = manifest.get("precision_bits")
    integer = manifest.get("integer_bits")
    if (type(width) is not int or type(integer) is not int or width < 2
            or not 0 < integer <= width):
        raise ValueError("Unknown or invalid synthesized fixed-point format")
    cpp_text = source_bytes.decode("utf-8")
    declaration = f"using data_t = ap_fixed<{width}, {integer}, AP_RND, AP_SAT>;"
    if declaration not in cpp_text:
        raise ValueError("Generated HLS source precision differs from source manifest")
    constants = [t for t in model.tensors if t.data is not None]
    # The generated emitter enumerates all tensors, including runtime inputs
    # and intermediate tensors, when assigning tensor_N identifiers.
    expected_lines = [
        "static const data_t tensor_" + str(i) + "[" + str(len(tensor.data)) +
        "] = {" + ", ".join(f"data_t({value:.9g})" for value in tensor.data) + "};"
        for i, tensor in enumerate(model.tensors) if tensor.data is not None
    ]
    actual_lines = [
        line for line in cpp_text.splitlines()
        if re.match(r"^static const data_t\b", line)
    ]
    if len(actual_lines) != len(expected_lines):
        raise ValueError(
            "Number of C++ constant declarations does not match ONNX initializers"
        )
    for position, (actual, expected) in enumerate(zip(actual_lines, expected_lines)):
        if actual != expected:
            # No actual or expected numerical values leave the audit.
            raise ValueError(
                f"Generated C++ initializer mismatch at constant declaration {position}"
            )
    return {
        "schema": "fpgai-research/onnx-hls-constant-audit/v1",
        "status": "ONNX_FLOAT32_CONSTANT_LITERALS_VERIFIED",
        "case": str(case),
        "onnx_file_matches_prepared_case": True,
        "model_ir_graph_and_constant_hash_matches_case": True,
        "generated_cpp_sha256_matches_manifest": True,
        "constant_literals_match_imported_onnx_values": True,
        "float32_constant_tensor_count": len(constants),
        "float32_constant_value_count": sum(len(t.data) for t in constants),
        "float32_constant_payload_sha256": _payload_digest(constants),
        "hls_fixed_point_format": f"ap_fixed<{width},{integer},AP_RND,AP_SAT>",
        "fixed_point_quantization_or_saturation_possible": True,
        "rtl_memory_cell_values_independently_verified": False,
        "fpga_bitstream_generated_by_this_audit": False,
        "physical_fpga_weights_verified": False,
        "raw_weight_values_printed": False,
        "note": (
            "Verifies original ONNX float32 constants as compiler-generated "
            "C++ literals. Vitis may quantize, optimize or reorganize constants. "
            "No on-device storage/readback or board inference is claimed."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Original ONNX model for this case")
    parser.add_argument("--case", required=True, help="Existing prepared FPGAI case directory")
    args = parser.parse_args(argv)
    try:
        report = audit_weight_constants(args.onnx, args.case)
    except (OSError, ValueError, UnicodeError, OverflowError, struct.error) as exc:
        print(
            f"FPGAI weights audit failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
