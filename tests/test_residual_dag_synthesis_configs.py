from pathlib import Path
import yaml

def test_residual_dag_synthesis_and_vivado_configs_are_maintained():
    synth=yaml.safe_load(Path("configs/examples/mixed_external_residual_add_synth.yml").read_text())
    vivado=yaml.safe_load(Path("configs/examples/mixed_external_residual_add_vivado.yml").read_text())
    assert synth["build"]["stages"]["hls_synthesis"] is True
    assert vivado["build"]["stages"]["hls_synthesis"] is True
    assert vivado["build"]["stages"]["vivado_project"] is True
    assert vivado["build"]["stages"]["vivado_implementation"] is True
    assert vivado["build"]["stages"]["bitstream"] is False
