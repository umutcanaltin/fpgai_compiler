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

    tb_source = (out_dir / 'hls/src/tb.cpp').read_text(encoding='utf-8')
    assert 'FPGAI CSim automatic gradient-export capture' in tb_source
    assert 'FPGAI_MODE_EXPORT_GRADIENTS' in tb_source
    assert 'gradients_after.bin' in tb_source
    assert 'gradients_export.bin' in tb_source


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


def test_numeric_validation_reports_optimizer_state_artifact_missing_when_reference_has_no_capture(tmp_path: Path) -> None:
    ref_v = tmp_path / 'optimizer_state_after_ref.bin'
    _write_f32(ref_v, [0.1, -0.2, 0.3])

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
                'packed_optimizer_state_after': {'ref': ref_v, 'got': tmp_path / 'optimizer_state_after.bin'},
            },
        },
    )
    payload = json.loads(artifacts['numeric_validation_json'].read_text(encoding='utf-8'))

    opt_state = payload['optimizer_state_validation']
    assert opt_state['status'] == 'artifact_missing'
    assert opt_state['passed'] is False
    assert opt_state['comparisons']['packed_optimizer_state_after']['ref_exists'] is True
    assert opt_state['comparisons']['packed_optimizer_state_after']['got_exists'] is False


def test_numeric_validation_reports_sgd_optimizer_state_not_applicable(tmp_path: Path) -> None:
    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode='training_on_device',
        source_generated=True,
        optimizer_state_artifacts={
            'requested': False,
            'optimizer': 'sgd',
            'storage': 'none',
            'expected_tensors': [],
            'comparisons': {},
        },
    )
    payload = json.loads(artifacts['numeric_validation_json'].read_text(encoding='utf-8'))

    opt_state = payload['optimizer_state_validation']
    assert opt_state['status'] == 'not_applicable'
    assert opt_state['passed'] is False


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
    assert opt_state['bias_correction'] is False
    assert numeric['training']['reference']['optimizer_type'] == 'adam'
    assert numeric['training']['reference']['optimizer_state_before_ref_bin'] is not None
    assert numeric['training']['reference']['optimizer_state_after_ref_bin'] is not None
    assert Path(numeric['training']['reference']['optimizer_state_before_ref_bin']).exists()
    assert Path(numeric['training']['reference']['optimizer_state_after_ref_bin']).exists()
    assert opt_state['status'] == 'artifact_missing'
    assert opt_state['passed'] is False


def test_compile_numeric_validation_compares_captured_optimizer_state_files(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    out_dir = tmp_path / 'training_adam_state_captured'
    raw.setdefault('project', {})['out_dir'] = str(out_dir)
    raw.setdefault('pipeline', {})['mode'] = 'training_on_device'
    opt_cfg = raw.setdefault('training', {}).setdefault('optimizer', {})
    opt_cfg['type'] = 'adam'
    opt_cfg['learning_rate'] = 0.001
    opt_cfg['beta1'] = 0.9
    opt_cfg['beta2'] = 0.999
    opt_cfg['epsilon'] = 1.0e-8
    raw['training'].setdefault('storage', {})['optimizer_state'] = 'bram'
    dm = raw.setdefault('data_movement', {}).setdefault('optimizer_state', {})
    dm['export'] = {'interface': 'm_axi', 'transport': 'ps_runtime', 'policy': 'full'}
    _cpp_only(raw)

    # Simulate a CSim/runtime capture that already wrote the optimizer-state payload.
    # The compiler must consume these files in numeric_validation.json instead of
    # leaving the optimizer state at generated_not_captured_by_testbench.
    (out_dir / 'training_reference').mkdir(parents=True, exist_ok=True)
    _write_f32(out_dir / 'training_reference' / 'optimizer_state_after_ref.bin', [0.1, -0.2, 0.3, -0.4])
    _write_f32(out_dir / 'optimizer_state_after.bin', [0.1, -0.2, 0.3, -0.4])

    result = _compile_raw(raw, tmp_path)
    compiled_out = Path(result.out_dir)
    numeric = json.loads((compiled_out / 'reports/numeric_validation.json').read_text(encoding='utf-8'))
    opt_state = numeric['optimizer_state_validation']
    assert opt_state['requested'] is True
    assert opt_state['status'] == 'compared'
    assert opt_state['passed'] is True
    assert opt_state['comparisons']['packed_optimizer_state_after']['max_abs_error'] == 0.0

    manifest = json.loads((compiled_out / 'runtime_package' / 'package_manifest.json').read_text(encoding='utf-8'))
    assert manifest['runtime_optimizer_state']['captured_state_present'] is True
    assert manifest['runtime_optimizer_state']['reference_state_present'] is True

    tb_source = (compiled_out / 'hls/src/tb.cpp').read_text(encoding='utf-8')
    assert 'FPGAI CSim automatic optimizer-state export capture' in tb_source
    assert 'FPGAI_MODE_EXPORT_OPTIMIZER_STATE' in tb_source
    assert 'optimizer_state_after.bin' in tb_source


def test_compile_numeric_validation_compares_captured_gradient_export_files(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    out_dir = tmp_path / 'training_gradient_export_captured'
    raw.setdefault('project', {})['out_dir'] = str(out_dir)
    raw.setdefault('pipeline', {})['mode'] = 'training_on_device'
    raw.setdefault('data_movement', {}).setdefault('gradients', {})['export'] = {
        'interface': 'm_axi',
        'transport': 'ps_runtime',
        'policy': 'tiled',
    }
    raw.setdefault('runtime', {})['sequence'] = [{'run_training': {'steps': 1}}, 'export_gradients']
    _cpp_only(raw)

    # Simulate a CSim/runtime capture of the dedicated gradient-export payload.
    # These files must survive clean compile and be compared in numeric_validation.json.
    (out_dir / 'training_reference').mkdir(parents=True, exist_ok=True)
    _write_f32(out_dir / 'training_reference' / 'grads_ref.bin', [0.5, -0.25, 0.125, -0.0625])
    _write_f32(out_dir / 'gradients_after.bin', [0.5, -0.25, 0.125, -0.0625])

    result = _compile_raw(raw, tmp_path)
    compiled_out = Path(result.out_dir)
    numeric = json.loads((compiled_out / 'reports/numeric_validation.json').read_text(encoding='utf-8'))
    grad_export = numeric['gradient_export']
    assert grad_export['requested'] is True
    assert grad_export['policy'] == 'tiled'
    assert grad_export['status'] == 'compared'
    assert grad_export['passed'] is True
    assert grad_export['comparisons']['flattened_gradients_export']['max_abs_error'] == 0.0


def test_generated_cpp_reports_enforce_no_unrequested_weight_export(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'cpp_import_without_export')
    raw.setdefault('weights', {})['mode'] = 'import'
    raw.setdefault('runtime', {})['sequence'] = ['import_weights', 'run_inference']
    raw.setdefault('codegen', {})['readability'] = 'high'
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    reports = out_dir / 'reports'
    source = (out_dir / 'hls/src/deeplearn.cpp').read_text(encoding='utf-8')
    readability = json.loads((reports / 'generated_cpp_readability.json').read_text(encoding='utf-8'))
    validation = json.loads((reports / 'generated_cpp_validation.json').read_text(encoding='utf-8'))
    manifest = json.loads((out_dir / 'manifest.json').read_text(encoding='utf-8'))

    assert 'FPGAI_MODE_IMPORT_WEIGHTS' in source
    assert 'FPGAI_MODE_EXPORT_WEIGHTS' not in source
    assert readability['status'] == 'passed'
    assert validation['status'] == 'passed'
    assert readability['requested_features']['import_weights_mode'] is True
    assert readability['requested_features']['export_weights_mode'] is False
    assert readability['present_features']['export_weights_mode'] is False
    assert readability['unrequested_present_features'] == []
    assert manifest['generated_hls_explanation_artifacts']['generated_cpp_readability_json'].endswith('reports/generated_cpp_readability.json')
    assert manifest['generated_hls_explanation_artifacts']['generated_cpp_validation_json'].endswith('reports/generated_cpp_validation.json')


def test_generated_training_cpp_omits_unrequested_gradient_and_optimizer_exports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'cpp_training_no_exports')
    raw.setdefault('pipeline', {})['mode'] = 'training_on_device'
    raw.setdefault('runtime', {})['sequence'] = ['run_training']
    raw.setdefault('data_movement', {})['gradients'] = {}
    raw.setdefault('data_movement', {})['optimizer_state'] = {}
    raw.setdefault('codegen', {})['readability'] = 'high'
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / 'hls/src/deeplearn.cpp').read_text(encoding='utf-8')
    readability = json.loads((out_dir / 'reports/generated_cpp_readability.json').read_text(encoding='utf-8'))
    validation = json.loads((out_dir / 'reports/generated_cpp_validation.json').read_text(encoding='utf-8'))

    assert 'FPGAI_MODE_RUN_TRAINING' in source
    assert 'FPGAI_MODE_EXPORT_GRADIENTS' not in source
    assert 'FPGAI_MODE_EXPORT_OPTIMIZER_STATE' not in source
    assert readability['requested_features']['export_gradients_mode'] is False
    assert readability['requested_features']['export_optimizer_state_mode'] is False
    assert readability['present_features']['export_gradients_mode'] is False
    assert readability['present_features']['export_optimizer_state_mode'] is False
    assert readability['unrequested_present_features'] == []
    assert validation['status'] == 'passed'


def test_codegen_readability_level_changes_generated_cpp_comment_density(tmp_path: Path) -> None:
    compact = copy.deepcopy(_load_inference_config())
    compact.setdefault('project', {})['out_dir'] = str(tmp_path / 'cpp_readability_compact')
    compact.setdefault('weights', {})['mode'] = 'embedded'
    compact.setdefault('runtime', {})['sequence'] = ['run_inference']
    compact.setdefault('codegen', {})['readability'] = 'compact'
    _cpp_only(compact)

    high = copy.deepcopy(_load_inference_config())
    high.setdefault('project', {})['out_dir'] = str(tmp_path / 'cpp_readability_high')
    high.setdefault('weights', {})['mode'] = 'embedded'
    high.setdefault('runtime', {})['sequence'] = ['run_inference']
    high.setdefault('codegen', {})['readability'] = 'high'
    _cpp_only(high)

    compact_result = _compile_raw(compact, tmp_path)
    high_result = _compile_raw(high, tmp_path)
    compact_report = json.loads((Path(compact_result.out_dir) / 'reports/generated_cpp_readability.json').read_text(encoding='utf-8'))
    high_report = json.loads((Path(high_result.out_dir) / 'reports/generated_cpp_readability.json').read_text(encoding='utf-8'))
    compact_source = (Path(compact_result.out_dir) / 'hls/src/deeplearn.cpp').read_text(encoding='utf-8')
    high_source = (Path(high_result.out_dir) / 'hls/src/deeplearn.cpp').read_text(encoding='utf-8')

    assert compact_report['readability_level'] == 'compact'
    assert high_report['readability_level'] == 'high'
    assert high_report['comment_line_count'] > compact_report['comment_line_count']
    assert 'FPGAI generated HLS top' not in compact_source[:512]
    assert 'FPGAI generated HLS top' in high_source[:512]
    assert compact_report['status'] == 'passed'
    assert high_report['status'] == 'passed'


def test_hls_truth_reports_cpp_only_are_not_requested(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'hls_truth_cpp_only')
    raw.setdefault('runtime', {})['sequence'] = ['run_inference']
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    hls_truth = json.loads((out_dir / 'reports/hls_synthesis_report.json').read_text(encoding='utf-8'))
    estimate = json.loads((out_dir / 'reports/estimate_vs_hls.json').read_text(encoding='utf-8'))
    manifest = json.loads((out_dir / 'manifest.json').read_text(encoding='utf-8'))

    assert hls_truth['status'] == 'not_requested'
    assert hls_truth['claimed_success'] is False
    assert hls_truth['paper_safe'] is False
    assert estimate['status'] == 'not_requested'
    assert estimate['paper_safe'] is False
    assert estimate['comparison'] is None
    assert manifest['hls_truth_artifacts']['hls_synthesis_report_json'].endswith('reports/hls_synthesis_report.json')
    assert manifest['hls_truth_artifacts']['estimate_vs_hls_json'].endswith('reports/estimate_vs_hls.json')


def test_hls_truth_parser_compares_fake_csynth_xml(tmp_path: Path) -> None:
    from fpgai.reporting.hls_truth import emit_hls_truth_reports

    out_dir = tmp_path / 'hls_truth_fake'
    report_dir = out_dir / 'hls/fpgai_hls_proj/sol1/syn/report'
    report_dir.mkdir(parents=True)
    (report_dir / 'deeplearn_csynth.xml').write_text(
        '''<profile>
          <AreaEstimates><Resources>
            <LUT>13190</LUT><FF>15662</FF><DSP>21</DSP><BRAM_18K>19</BRAM_18K>
          </Resources></AreaEstimates>
          <PerformanceEstimates><SummaryOfOverallLatency>
            <Average-caseLatency>44340</Average-caseLatency>
          </SummaryOfOverallLatency></PerformanceEstimates>
        </profile>''',
        encoding='utf-8',
    )

    class FakeDesignResult:
        pass

    design_result = FakeDesignResult()
    design_result.results_json = out_dir / 'design_space/results.json'
    design_result.results_json.parent.mkdir(parents=True)
    design_result.results_json.write_text(json.dumps({
        'recommended_balanced': {
            'predicted_lut': 4239,
            'predicted_ff': 5751,
            'predicted_dsp': 40,
            'predicted_bram18': 18,
            'predicted_cycles': 14460,
            'predicted_latency_ms': 0.0723,
        }
    }), encoding='utf-8')

    artifacts = emit_hls_truth_reports(
        out_dir=out_dir,
        hls_dir=out_dir / 'hls',
        build_stages={'hls_project': True, 'hls_synthesis': True},
        hls_run=None,
        design_result=design_result,
        clock_mhz=200.0,
    )

    hls_truth = json.loads(artifacts.hls_synthesis_report_json.read_text(encoding='utf-8'))
    estimate = json.loads(artifacts.estimate_vs_hls_json.read_text(encoding='utf-8'))

    assert hls_truth['status'] == 'parsed'
    assert hls_truth['paper_safe'] is True
    assert hls_truth['actual']['lut'] == 13190
    assert hls_truth['actual']['ff'] == 15662
    assert hls_truth['actual']['dsp'] == 21
    assert hls_truth['actual']['bram18'] == 19
    assert hls_truth['actual']['latency_cycles'] == pytest.approx(44340.0)
    assert hls_truth['actual']['latency_ms'] == pytest.approx(0.2217)
    assert estimate['status'] == 'compared'
    assert estimate['paper_safe'] is True
    assert estimate['comparison']['resources']['lut']['estimated'] == 4239
    assert estimate['comparison']['resources']['lut']['hls'] == 13190
    assert estimate['comparison']['latency']['ms']['hls'] == pytest.approx(0.2217)


def test_hls_synthesis_missing_tool_reports_tool_missing_without_fake_success(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'hls_tool_missing')
    raw.setdefault('runtime', {})['sequence'] = ['run_inference']
    raw.setdefault('build', {})['stages'] = {
        'cpp': True,
        'testbench': True,
        'hls_project': True,
        'hls_synthesis': True,
        'vivado_project': False,
        'vivado_implementation': False,
        'bitstream': False,
        'runtime_package': True,
        'reports': True,
    }
    raw.setdefault('toolchain', {}).setdefault('vitis_hls', {})['enabled'] = True
    raw.setdefault('backends', {}).setdefault('hls', {}).setdefault('vitis', {})['exe'] = '__fpgai_missing_vitis_hls__'

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    hls_truth = json.loads((out_dir / 'reports/hls_synthesis_report.json').read_text(encoding='utf-8'))
    estimate = json.loads((out_dir / 'reports/estimate_vs_hls.json').read_text(encoding='utf-8'))

    assert hls_truth['status'] == 'tool_missing'
    assert hls_truth['claimed_success'] is False
    assert hls_truth['paper_safe'] is False
    assert hls_truth['hls_run']['returncode'] == 127
    assert estimate['status'] == 'tool_missing'
    assert estimate['paper_safe'] is False


def _set_fixed_precision(raw: dict, *, mode: str, activation: int, weight: int, bias: int, accum: int) -> None:
    numerics = raw.setdefault('numerics', {})
    numerics['precision_mode'] = mode
    numerics['defaults'] = {
        'activation': {'type': 'ap_fixed', 'total_bits': activation, 'int_bits': max(1, min(6, activation))},
        'weight': {'type': 'ap_fixed', 'total_bits': weight, 'int_bits': max(1, min(6, weight))},
        'bias': {'type': 'ap_fixed', 'total_bits': bias, 'int_bits': max(1, min(10, bias))},
        'accum': {'type': 'ap_fixed', 'total_bits': accum, 'int_bits': max(1, min(10, accum))},
    }


def test_precision_effect_report_materializes_manual_precision_in_generated_types(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'precision_effect_fx8')
    raw.setdefault('runtime', {})['sequence'] = ['run_inference']
    _set_fixed_precision(raw, mode='fx8_test', activation=8, weight=8, bias=16, accum=16)
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    report = json.loads((out_dir / 'reports/precision_effect.json').read_text(encoding='utf-8'))
    types_h = (out_dir / 'hls/include/fpgai_types.h').read_text(encoding='utf-8')
    manifest = json.loads((out_dir / 'manifest.json').read_text(encoding='utf-8'))

    assert report['status'] == 'validated'
    assert report['precision']['resolved'] == 'fx8_test'
    assert report['precision']['source'] == 'manual_yaml'
    assert 'numerics.defaults.activation' in report['precision']['manual_yaml_sources']
    assert report['precision']['bits']['activation'] == 8
    assert report['precision']['bits']['weight'] == 8
    assert report['generated_artifacts']['type_header_exists'] is True
    assert report['generated_artifacts']['type_changed'] is True
    assert report['generated_artifacts']['expected_patterns_present']['activation'] is True
    assert 'typedef ap_fixed<8,' in types_h
    assert 'typedef ap_fixed<16,' in types_h
    assert report['resource_effect']['paper_safe_hls_claim'] is False
    assert manifest['precision_effect_artifacts']['precision_effect_json'].endswith('reports/precision_effect.json')


def test_precision_effect_distinguishes_fx8_and_fx16_layouts(tmp_path: Path) -> None:
    fx8 = copy.deepcopy(_load_inference_config())
    fx8.setdefault('project', {})['out_dir'] = str(tmp_path / 'precision_effect_fx8_compare')
    fx8.setdefault('runtime', {})['sequence'] = ['run_inference']
    _set_fixed_precision(fx8, mode='fx8_compare', activation=8, weight=8, bias=16, accum=16)
    _cpp_only(fx8)

    fx16 = copy.deepcopy(_load_inference_config())
    fx16.setdefault('project', {})['out_dir'] = str(tmp_path / 'precision_effect_fx16_compare')
    fx16.setdefault('runtime', {})['sequence'] = ['run_inference']
    _set_fixed_precision(fx16, mode='fx16_compare', activation=16, weight=16, bias=24, accum=24)
    _cpp_only(fx16)

    r8 = _compile_raw(fx8, tmp_path)
    r16 = _compile_raw(fx16, tmp_path)
    p8 = Path(r8.out_dir)
    p16 = Path(r16.out_dir)
    report8 = json.loads((p8 / 'reports/precision_effect.json').read_text(encoding='utf-8'))
    report16 = json.loads((p16 / 'reports/precision_effect.json').read_text(encoding='utf-8'))
    types8 = (p8 / 'hls/include/fpgai_types.h').read_text(encoding='utf-8')
    types16 = (p16 / 'hls/include/fpgai_types.h').read_text(encoding='utf-8')

    assert report8['precision']['bits']['activation'] == 8
    assert report16['precision']['bits']['activation'] == 16
    assert report8['precision']['bits'] != report16['precision']['bits']
    assert types8 != types16
    assert 'typedef ap_fixed<8,' in types8
    assert 'typedef ap_fixed<16,' in types16
    assert report8['truth_boundary']['paper_safe_hls_claim_requires_estimate_vs_hls_compared'] is True
    assert report16['truth_boundary']['paper_safe_hls_claim_requires_estimate_vs_hls_compared'] is True


def test_parallel_pipeline_effect_materializes_manual_hls_pragmas_and_macros(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'parallel_pipeline_effect_manual')
    raw.setdefault('runtime', {})['sequence'] = ['run_inference']
    raw.setdefault('optimization', {}).setdefault('pipeline', {})['ii'] = 2
    raw['optimization'].setdefault('parallel', {})['pe'] = 4
    raw['optimization']['parallel']['simd'] = 2
    raw['optimization']['parallel']['partition_factor'] = 4
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    report = json.loads((out_dir / 'reports/parallel_pipeline_effect.json').read_text(encoding='utf-8'))
    manifest = json.loads((out_dir / 'manifest.json').read_text(encoding='utf-8'))
    hls_source = "\n".join(
        path.read_text(encoding='utf-8')
        for path in (out_dir / 'hls').rglob('*')
        if path.suffix in {'.cpp', '.h', '.hpp'}
    )

    assert report['artifact_kind'] == 'parallel_pipeline_effect'
    assert report['status'] == 'validated'
    assert report['manual_yaml_sources']['optimization.pipeline.ii'] == 2
    assert report['manual_yaml_sources']['optimization.parallel.pe'] == 4
    assert report['pipeline']['status'] == 'applied'
    assert report['pipeline']['resolved'] == 2
    assert report['parallelization']['dense_out_unroll']['status'] == 'applied'
    assert report['parallelization']['dense_out_unroll']['resolved'] == 4
    assert report['hls_effect']['paper_safe_hls_claim'] is False
    assert '#define FPGAI_PIPELINE_II 2' in hls_source or '#define FPGAI_PIPELINE_II 2' in report['pipeline']['evidence']
    assert '#define FPGAI_DENSE_OUT_UNROLL 4' in hls_source or '#define FPGAI_DENSE_OUT_UNROLL 4' in report['parallelization']['dense_out_unroll']['evidence']
    assert '#define FPGAI_DENSE_IN_UNROLL 2' in hls_source or '#define FPGAI_DENSE_IN_UNROLL 2' in report['parallelization']['dense_in_unroll']['evidence']
    assert '#define FPGAI_DENSE_PARTITION_WEIGHTS 4' in hls_source or '#define FPGAI_DENSE_PARTITION_WEIGHTS 4' in report['parallelization']['dense_weight_partition']['evidence']
    assert '#pragma HLS PIPELINE' in hls_source or '#pragma HLS PIPELINE' in report['pipeline']['evidence']
    assert '#pragma HLS UNROLL' in hls_source or any('#pragma HLS UNROLL' in e for e in report['parallelization']['dense_out_unroll']['evidence'])
    assert '#pragma HLS ARRAY_PARTITION' in hls_source or any('#pragma HLS ARRAY_PARTITION' in e for e in report['parallelization']['dense_weight_partition']['evidence'])
    assert manifest['parallel_pipeline_effect_artifacts']['parallel_pipeline_effect_json'].endswith('reports/parallel_pipeline_effect.json')


def test_parallel_pipeline_effect_records_not_requested_defaults_without_manual_claim(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault('project', {})['out_dir'] = str(tmp_path / 'parallel_pipeline_effect_defaults')
    raw.setdefault('runtime', {})['sequence'] = ['run_inference']
    _cpp_only(raw)

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    report = json.loads((out_dir / 'reports/parallel_pipeline_effect.json').read_text(encoding='utf-8'))

    assert report['manual_yaml_sources'] == {}
    assert report['pipeline']['requested'] is False
    assert report['resource_latency_hygiene']['unrequested_manual_claims'] == []
    assert report['parallelization']['dense_out_unroll']['status'] in {'not_requested', 'compiler_default'}
    assert report['hls_effect']['paper_safe_hls_claim'] is False
