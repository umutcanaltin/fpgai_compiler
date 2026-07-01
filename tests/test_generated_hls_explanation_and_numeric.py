from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import pytest
import yaml

from fpgai.validation.numeric import emit_numeric_validation_report


def _write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack('<' + 'f' * len(values), *values))


def test_numeric_validation_compares_inference_output_files(tmp_path: Path) -> None:
    inp = tmp_path / 'inputs.bin'
    ref = tmp_path / 'outputs_ref.bin'
    hw = tmp_path / 'outputs_hw.bin'
    _write_f32(inp, [1.0, 2.0])
    _write_f32(ref, [0.25, -0.5, 1.0])
    _write_f32(hw, [0.25, -0.5, 1.0])

    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode='inference',
        source_generated=True,
        inference_reference_artifacts={
            'inputs_bin': inp,
            'outputs_ref': ref,
            'outputs_hw': hw,
        },
    )
    payload = json.loads(artifacts['numeric_validation_json'].read_text(encoding='utf-8'))

    assert payload['status'] == 'passed'
    assert payload['passed'] is True
    assert payload['paper_claim_allowed']['numeric_correctness'] is True
    assert payload['inference']['output_compare']['max_abs_error'] == 0.0
    assert payload['inference']['output_compare']['cosine_similarity'] == pytest.approx(1.0)


def _load_inference_config() -> dict:
    data = yaml.safe_load(Path('configs/examples/inference_compile.yml').read_text(encoding='utf-8'))
    assert isinstance(data, dict)
    return data


def _load_training_config() -> dict:
    for p in [
        Path('paper_experiments/full_pipeline_gate/sprint26_paper_matrix/configs/training_kv260_aggressive_fx8_3.yml'),
        Path('paper_experiments/full_pipeline_gate/sprint27h_full_rerun/configs_hls/training_kv260_aggressive_fx8_3.yml'),
        Path('configs/examples/training_compile_smoke.yml'),
    ]:
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                return data
    pytest.skip('training config not available')


def _cpp_only(raw: dict) -> None:
    raw.setdefault('toolchain', {}).setdefault('vitis_hls', {})['enabled'] = False
    raw.setdefault('build', {})['stages'] = {
        'cpp': True,
        'testbench': True,
        'hls_project': False,
        'hls_synthesis': False,
        'vivado_project': False,
        'vivado_implementation': False,
        'bitstream': False,
        'runtime_package': True,
        'reports': True,
    }


def _make_config(raw: dict, tmp_path: Path):
    cfg_path = tmp_path / 'compile.yml'
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
    from fpgai.config.loader import load_config

    return load_config(str(cfg_path))


def _compile_raw(raw: dict, tmp_path: Path):
    pytest.importorskip('onnx')
    from fpgai.engine.compiler import Compiler

    return Compiler(_make_config(raw, tmp_path)).compile()


def test_compile_emits_explainable_hls_reports_for_inference_cpp_only(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'explain_inference')
    raw.setdefault('weights', {})['mode'] = 'import_export'
    raw.setdefault('runtime', {})['sequence'] = ['import_weights', {'run_inference': {'repeat': 2}}, 'export_weights']
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    reports = out_dir / 'reports'

    explanation = json.loads((reports / 'generated_hls_explanation.json').read_text(encoding='utf-8'))
    decisions = json.loads((reports / 'hardware_design_decisions.json').read_text(encoding='utf-8'))
    manifest = json.loads((out_dir / 'manifest.json').read_text(encoding='utf-8'))

    assert explanation['generated_files']['top_source_exists'] is True
    assert explanation['source_evidence']['checks']['top_function_deeplearn'] is True
    assert explanation['decisions']['runtime_sequence']['sequence'][0]['command'] == 'import_weights'
    assert decisions['top_name'] == 'deeplearn'
    assert (reports / 'generated_hls_explanation.md').exists()
    assert (reports / 'hardware_design_decisions.md').exists()
    assert (reports / 'codegen_review_checklist.md').exists()
    assert manifest['generated_hls_explanation_artifacts']['generated_hls_explanation_json'].endswith('reports/generated_hls_explanation.json')


def test_training_numeric_validation_records_gradient_export_status_and_explanation(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'explain_training_gradient_export')
    raw.setdefault('pipeline', {})['mode'] = 'training_on_device'
    raw.setdefault('data_movement', {}).setdefault('gradients', {})['export'] = {
        'interface': 'm_axi',
        'transport': 'ps_runtime',
        'policy': 'tiled',
    }
    raw.setdefault('runtime', {})['sequence'] = [{'run_training': {'steps': 1}}, 'export_gradients']
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    numeric = json.loads((out_dir / 'reports/numeric_validation.json').read_text(encoding='utf-8'))
    explanation = json.loads((out_dir / 'reports/generated_hls_explanation.json').read_text(encoding='utf-8'))

    assert numeric['training']['reference']['status'] == 'generated'
    assert numeric['gradient_export']['requested'] is True
    assert numeric['gradient_export']['policy'] == 'tiled'
    assert numeric['gradient_export']['status'] in {'generated_not_captured_by_testbench', 'covered_by_training_gradient_compare'}
    assert explanation['source_evidence']['checks']['runtime_mode_export_gradients'] is True
    assert explanation['source_evidence']['checks']['m_axi_gradient_port'] is True


def test_numeric_validation_compares_optimizer_state_files(tmp_path: Path) -> None:
    ref_v = tmp_path / 'optimizer_state_after_ref.bin'
    got_v = tmp_path / 'optimizer_state_after.bin'
    _write_f32(ref_v, [0.1, -0.2, 0.3])
    _write_f32(got_v, [0.1, -0.2, 0.3])

    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode='training_on_device',
        source_generated=True,
        optimizer_state_artifacts={
            'requested': True,
            'optimizer': 'momentum',
            'storage': 'bram',
            'expected_tensors': ['velocity'],
            'comparisons': {
                'velocity_after': {'ref': ref_v, 'got': got_v},
            },
        },
    )
    payload = json.loads(artifacts['numeric_validation_json'].read_text(encoding='utf-8'))

    opt_state = payload['optimizer_state_validation']
    assert opt_state['requested'] is True
    assert opt_state['optimizer'] == 'momentum'
    assert opt_state['status'] == 'compared'
    assert opt_state['passed'] is True
    assert opt_state['comparisons']['velocity_after']['max_abs_error'] == 0.0


def test_training_numeric_validation_records_adam_optimizer_state_debt(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'training_adam_state_numeric')
    raw.setdefault('pipeline', {})['mode'] = 'training_on_device'
    opt_cfg = raw.setdefault('training', {}).setdefault('optimizer', {})
    opt_cfg['type'] = 'adam'
    opt_cfg['learning_rate'] = 0.001
    opt_cfg['beta1'] = 0.9
    opt_cfg['beta2'] = 0.999
    opt_cfg['epsilon'] = 1.0e-8
    raw['training'].setdefault('storage', {})['optimizer_state'] = 'bram'
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    numeric = json.loads((out_dir / 'reports/numeric_validation.json').read_text(encoding='utf-8'))

    opt_state = numeric['optimizer_state_validation']
    assert opt_state['requested'] is True
    assert opt_state['optimizer'] == 'adam'
    assert opt_state['storage'] == 'bram'
    assert opt_state['expected_tensors'] == ['first_moment', 'second_moment']
    assert opt_state['status'] == 'generated_not_captured_by_testbench'
    assert opt_state['passed'] is False
