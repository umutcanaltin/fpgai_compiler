from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.quantization import emit_quantized_relu_int8x4_vhdl_package


def _q(scale: float):
    return {
        "spec": {
            "bits": 8,
            "scheme": "symmetric",
            "granularity": "per_tensor",
            "signed": True,
            "axis": None,
            "rounding": "nearest",
            "saturation": "saturate",
        },
        "scale": scale,
        "zero_point": 0,
        "observed_min": -1.0,
        "observed_max": 1.0,
    }


def test_emit_specialized_quantized_relu_vhdl_package_supports_requantization(tmp_path: Path) -> None:
    partition = {
        "schema": "fpgai.quantized-residual-operator-partition/v1",
        "partition_type": "residual_add_relu",
        "relu": {
            "input_quantization": _q(0.0162),
            "output_quantization": _q(0.0160),
            "lowering": {
                "input_zero": 0,
                "multiplier": 1087163597,
                "shift": 30,
                "output_zero": 0,
                "qmin": -128,
                "qmax": 127,
                "rounding_mode": 0,
                "saturation_mode": 0,
            },
        },
    }
    package = emit_quantized_relu_int8x4_vhdl_package(tmp_path / "pkg", partition)
    contract = implementation_contract_from_manifest(package)
    assert contract.top == "quantized_relu_int8x4_vhdl"
    source = (package / "rtl" / "quantized_relu_int8x4_vhdl.vhd").read_text(encoding="utf-8")
    assert "constant INPUT_ZERO    : integer := 0;" in source
    assert "constant MULTIPLIER    : integer := 1087163597;" in source
    assert "constant SHIFT         : natural := 30;" in source
    assert "centered := resize(raw, 32) - to_signed(INPUT_ZERO, 32);" in source
    assert "product := centered * to_signed(MULTIPLIER, 32);" in source
    assert "with_zero := shifted + to_signed(OUTPUT_ZERO, 64);" in source


def test_generated_relu_does_not_assume_identical_quantization_domains(tmp_path: Path) -> None:
    partition = {
        "schema": "fpgai.quantized-residual-operator-partition/v1",
        "partition_type": "residual_add_relu",
        "relu": {
            "input_quantization": _q(0.02),
            "output_quantization": _q(0.01),
            "lowering": {
                "input_zero": 0,
                "multiplier": 2,
                "shift": 0,
                "output_zero": 0,
                "qmin": -128,
                "qmax": 127,
                "rounding_mode": 0,
                "saturation_mode": 0,
            },
        },
    }
    package = emit_quantized_relu_int8x4_vhdl_package(tmp_path / "pkg2", partition)
    source = (package / "rtl" / "quantized_relu_int8x4_vhdl.vhd").read_text(encoding="utf-8")
    assert "constant MULTIPLIER    : integer := 2;" in source
    assert "constant SHIFT         : natural := 0;" in source


def test_generated_relu_derives_lowering_for_legacy_partition_artifact(tmp_path: Path) -> None:
    partition = {
        "schema": "fpgai.quantized-residual-operator-partition/v1",
        "partition_type": "residual_add_relu",
        "relu": {
            "input_quantization": _q(0.0162),
            "output_quantization": _q(0.0160),
        },
    }
    package = emit_quantized_relu_int8x4_vhdl_package(tmp_path / "legacy_pkg", partition)
    source = (package / "rtl" / "quantized_relu_int8x4_vhdl.vhd").read_text(encoding="utf-8")
    assert "constant MULTIPLIER" in source
    assert "constant SHIFT" in source
    assert "centered := resize(raw, 32) - to_signed(INPUT_ZERO, 32);" in source
    contract = implementation_contract_from_manifest(package)
    assert contract.top == "quantized_relu_int8x4_vhdl"
