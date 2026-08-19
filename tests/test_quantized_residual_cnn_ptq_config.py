import subprocess
import sys
from pathlib import Path
import yaml


def test_quantized_residual_cnn_ptq_config_is_explicit_and_public():
    path = Path("configs/examples/quantized_residual_cnn_ptq.yml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    q = raw["numerics"]["quantization"]
    assert q["mode"] == "ptq"
    assert q["weights"]["bits"] == 8
    assert q["weights"]["granularity"] == "per_channel"
    assert q["weights"]["axis"] == 0
    assert q["activations"]["bits"] == 8
    assert q["activations"]["granularity"] == "per_tensor"
    assert q["accumulators"]["bits"] == 32
    assert q["calibration"]["samples"] > 0


def test_quantized_residual_cnn_ptq_runner_supports_direct_script_execution():
    result = subprocess.run(
        [sys.executable, "scripts/run_quantized_residual_cnn_ptq.py", "--help"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--skip-hls" in result.stdout


def test_quantized_residual_cnn_vhdl_add_partition_is_yaml_selectable():
    raw = yaml.safe_load(Path("configs/examples/quantized_residual_cnn_ptq_mixed_add.yml").read_text(encoding="utf-8"))
    mixed = raw["targets"]["mixed_backend"]
    assert mixed["residual_add_backend"] == "vhdl"
    assert mixed["final_relu_backend"] == "vhdl"
    assert raw["targets"]["hls"]["control_protocol"] == "ap_ctrl_none"


def test_quantized_residual_cnn_backend_matrix_configs_are_explicit():
    cases = {
        "quantized_residual_cnn_ptq_hls_add_vhdl_relu.yml": ("hls", "vhdl"),
        "quantized_residual_cnn_ptq_vhdl_add_hls_relu.yml": ("vhdl", "hls"),
    }
    for filename, expected in cases.items():
        raw = yaml.safe_load(Path("configs/examples", filename).read_text(encoding="utf-8"))
        mixed = raw["targets"]["mixed_backend"]
        assert (mixed["residual_add_backend"], mixed["final_relu_backend"]) == expected
