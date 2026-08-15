from pathlib import Path
import yaml


def _load(name):
    return yaml.safe_load(Path('configs/examples',name).read_text())


def test_residual_cnn_csim_config_enables_numeric_validation():
    raw=_load('mixed_external_residual_cnn_csim.yml')
    assert raw['ecosystem']['validation']['enabled'] is True
    assert raw['ecosystem']['validation']['run_vitis_csim'] is True
    assert len(raw['ecosystem']['validation']['input_values']) == 16
    assert raw['build']['stages']['hls_synthesis'] is False


def test_residual_cnn_synth_and_vivado_configs_stage_real_tools():
    synth=_load('mixed_external_residual_cnn_synth.yml')
    vivado=_load('mixed_external_residual_cnn_vivado.yml')
    assert synth['build']['stages']['hls_synthesis'] is True
    assert vivado['build']['stages']['hls_synthesis'] is True
    assert vivado['build']['stages']['vivado_implementation'] is True
