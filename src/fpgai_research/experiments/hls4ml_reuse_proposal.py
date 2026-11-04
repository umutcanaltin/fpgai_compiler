"""Select one supported MatMul reuse control from a private hls4ml census.

This is a reproducible *proposal*, not a legality guarantee or realized
architecture setting; the actual E1 generator must still accept the request.
"""
from __future__ import annotations

import json
from pathlib import Path


def propose_reuse(census_dir, output, *, factor=2):
    if type(factor) is not int or factor < 2 or factor > 4096:
        raise ValueError("Tuned reuse factor must be an integer 2..4096")
    source = Path(census_dir) / "layer_controls_private.json"
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 1024 * 1024:
        raise ValueError("Missing or unsafe immutable private hls4ml census")
    receipt = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "fpgai-research/hls4ml-layer-control-census/v1"
            or receipt.get("hls4ml_version") != "1.3.0"
            or receipt.get("actual_onnx_parser_and_backend_config_generator_called") is not True
            or receipt.get("native_vendor_tools_executed") is not False):
        raise ValueError("No qualified per-layer hls4ml source census")
    allowed = sorted(
        row["layer_name"] for row in receipt.get("matmul_layers", [])
        if isinstance(row, dict) and row.get("present_in_hls4ml_config") is True
        and row.get("configurable_reuse_factor") is True
        and isinstance(row.get("layer_name"), str) and row["layer_name"]
    )
    if not allowed:
        raise ValueError("No source-attested MatMul layer exposes configurable ReuseFactor")
    requested = {allowed[0]: factor}
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable tuned hls4ml control proposal already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    # The exact generated layer identifier is private and must not be copied
    # into public manuscript prose without separate release approval.
    with target.open("x", encoding="utf-8") as file:
        json.dump(requested, file, sort_keys=True, indent=2)
        file.write("\n")
    return {
        "model_kind": receipt["model_kind"],
        "selected_control_count": 1,
        "requested_reuse_factor": factor,
        "source_only_proposal": True,
        "hls4ml_e1_execution_verified": False,
        "native_e2_e3_e4_verified": False,
    }
