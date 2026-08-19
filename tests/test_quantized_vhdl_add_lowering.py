from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.quantization import emit_quantized_add_int8x4_vhdl_package


def _q(scale: float):
    return {
        "spec": {"bits": 8, "scheme": "symmetric", "granularity": "per_tensor", "signed": True, "axis": None, "rounding": "nearest", "saturation": "saturate"},
        "scale": scale, "zero_point": 0, "observed_min": -1.0, "observed_max": 1.0,
    }


def test_emit_specialized_quantized_add_vhdl_package(tmp_path: Path) -> None:
    partition = {
        "schema": "fpgai.quantized-residual-operator-partition/v1",
        "partition_type": "residual_add_relu",
        "add": {
            "left_quantization": _q(0.125),
            "right_quantization": _q(0.25),
            "output_quantization": _q(0.5),
            "lowering": {
                "left_zero": 0, "left_multiplier": 536870912, "left_shift": 31,
                "right_zero": 0, "right_multiplier": 1073741824, "right_shift": 31,
                "output_zero": 0, "qmin": -128, "qmax": 127,
                "rounding_mode": 0, "saturation_mode": 0,
            },
        },
    }
    package = emit_quantized_add_int8x4_vhdl_package(tmp_path / "pkg", partition)
    contract = implementation_contract_from_manifest(package)
    assert contract.top == "quantized_add_int8x4_vhdl"
    source = (package / "rtl" / "quantized_add_int8x4_vhdl.vhd").read_text()
    assert "constant LEFT_MULTIPLIER  : integer := 536870912;" in source
    assert "constant RIGHT_MULTIPLIER : integer := 1073741824;" in source
    assert "function add_lane" in source
    assert "left_q := requant_lane" in source
