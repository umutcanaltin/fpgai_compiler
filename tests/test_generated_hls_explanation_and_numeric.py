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


def test_numeric_validation_inference_failed_tolerance_is_explicit(tmp_path: Path) -> None:
    inp = tmp_path / "inputs.bin"
    ref = tmp_path / "outputs_ref.bin"
    hw = tmp_path / "outputs_hw.bin"
    _write_f32(inp, [1.0, 2.0])
    _write_f32(ref, [1.0, 0.0])
    _write_f32(hw, [1.5, 0.0])

    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=True,
        hls_ok=True,
        raw_config={"benchmark": {"compare": {"precision_aware": False, "max_abs_error": 0.01}}},
        inference_reference_artifacts={
            "inputs_bin": inp,
            "outputs_ref": ref,
            "outputs_hw": hw,
        },
    )
    payload = json.loads(artifacts["numeric_validation_json"].read_text(encoding="utf-8"))

    assert payload["status"] == "failed_tolerance"
    assert payload["passed"] is False
    assert payload["paper_claim_allowed"]["numeric_correctness"] is False
    assert payload["inference"]["output_compare"]["status"] == "compared"
    assert payload["inference"]["output_compare"]["passed"] is False
    assert "max_abs_error" in payload["reason"]


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


def test_inference_reference_artifact_helper_uses_existing_hls_output_and_input_bin(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    import numpy as np
    monkeypatch.setitem(
        sys.modules,
        "fpgai.toolchain",
        types.SimpleNamespace(build_xilinx_tool_command=lambda *args, **kwargs: []),
    )
    from fpgai.engine.compiler import _emit_inference_reference_artifacts

    out_dir = tmp_path / "inference_out"
    out_dir.mkdir()
    np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tofile(out_dir / "input.bin")
    np.asarray([0.3, 0.7], dtype=np.float32).tofile(out_dir / "output.bin")

    class _Meta:
        def __init__(self, name: str, shape: list[int]) -> None:
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_inputs(self):
            return [_Meta("input", [1, 4])]

        def get_outputs(self):
            return [_Meta("output", [1, 2])]

        def run(self, outputs, feed):
            x = next(iter(feed.values()))
            assert tuple(x.shape) == (1, 4)
            return [np.asarray([[0.3, 0.7]], dtype=np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=_Session))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")

    artifacts = _emit_inference_reference_artifacts(out_dir, model_path=model, hls_ok=True)

    assert artifacts["status"] == "available"
    assert Path(artifacts["outputs_ref"]).exists()
    assert Path(artifacts["outputs_hw"]) == out_dir / "output.bin"


def test_numeric_validation_inference_uses_precision_aware_limits(tmp_path: Path) -> None:
    inp = tmp_path / "input.bin"
    ref = tmp_path / "outputs_ref.bin"
    hw = tmp_path / "output.bin"
    _write_f32(inp, [1.0, 2.0])
    _write_f32(ref, [1.0, 0.0])
    # Error is too large for the historic 1e-3 threshold but acceptable for fx8_3
    # precision-aware reporting (LSB = 2^-5, max limit >= 4 LSB = 0.125).
    _write_f32(hw, [1.0625, 0.0])

    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=True,
        hls_ok=True,
        raw_config={
            "numerics": {
                "defaults": {
                    "activation": {"total_bits": 8, "int_bits": 3},
                },
            },
            "benchmark": {"compare": {"precision_aware": True}},
        },
        inference_reference_artifacts={
            "inputs_bin": inp,
            "outputs_ref": ref,
            "outputs_hw": hw,
        },
    )
    payload = json.loads(artifacts["numeric_validation_json"].read_text(encoding="utf-8"))

    assert payload["status"] == "passed"
    assert payload["paper_claim_allowed"]["numeric_correctness"] is True
    compare = payload["inference"]["output_compare"]
    assert compare["max_abs_error"] == pytest.approx(0.0625)
    assert compare["limits"]["max_abs_error_limit"] >= 0.125


def test_numeric_validation_emits_classification_decision_metrics(tmp_path: Path) -> None:
    inp = tmp_path / "inputs.bin"
    ref = tmp_path / "outputs_ref.bin"
    hw = tmp_path / "outputs_hw.bin"
    labels = tmp_path / "labels.txt"
    _write_f32(inp, [0.0, 1.0])
    # Two samples, three classes. Reference predicts [1, 2]; generated predicts [1, 0].
    _write_f32(ref, [0.1, 0.8, 0.1, 0.1, 0.2, 0.7])
    _write_f32(hw, [0.2, 0.7, 0.1, 0.6, 0.3, 0.1])
    labels.write_text("1\n2\n", encoding="utf-8")

    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        raw_config={
            "validation": {
                "task": "classification",
                "labels": str(labels),
                "dataset": "sample_labels",
                "decision_thresholds": {"max_accuracy_drop_pct": 60.0},
            },
            "benchmark": {"compare": {"precision_aware": False, "max_abs_error": 1.0, "min_cosine_similarity": 0.0}},
        },
        inference_reference_artifacts={"inputs_bin": inp, "outputs_ref": ref, "outputs_hw": hw},
    )
    payload = json.loads(artifacts["numeric_validation_json"].read_text(encoding="utf-8"))
    task = payload["inference"]["task_quality"]

    assert task["task"] == "classification"
    assert task["status"] == "compared"
    assert task["labels_status"] == "provided"
    assert task["reference_top1_accuracy"] == pytest.approx(1.0)
    assert task["generated_top1_accuracy"] == pytest.approx(0.5)
    assert task["top1_accuracy_drop_pct"] == pytest.approx(-50.0)
    assert task["prediction_agreement_vs_reference"] == pytest.approx(0.5)
    assert task["decision_status"] in {"aggressive_compression", "acceptable_tradeoff", "not_recommended_for_quality"}


def test_numeric_validation_emits_regression_decision_metrics(tmp_path: Path) -> None:
    inp = tmp_path / "inputs.bin"
    ref = tmp_path / "outputs_ref.bin"
    hw = tmp_path / "outputs_hw.bin"
    targets = tmp_path / "targets.csv"
    _write_f32(inp, [0.0])
    _write_f32(ref, [1.0, 2.0, 3.0])
    _write_f32(hw, [1.1, 1.9, 3.2])
    targets.write_text("1.0,2.0,3.0", encoding="utf-8")

    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        raw_config={
            "validation": {
                "task": "regression",
                "targets": str(targets),
                "decision_thresholds": {"max_mae_increase": 0.25, "max_rmse_increase": 0.25},
            },
            "benchmark": {"compare": {"precision_aware": False, "max_abs_error": 1.0, "min_cosine_similarity": 0.0}},
        },
        inference_reference_artifacts={"inputs_bin": inp, "outputs_ref": ref, "outputs_hw": hw},
    )
    payload = json.loads(artifacts["numeric_validation_json"].read_text(encoding="utf-8"))
    task = payload["inference"]["task_quality"]

    assert task["task"] == "regression"
    assert task["targets_status"] == "provided"
    assert task["target_mae_reference"] == pytest.approx(0.0)
    assert task["target_mae_generated"] == pytest.approx((0.1 + 0.1 + 0.2) / 3, rel=1e-5)
    assert task["mae_increase"] == pytest.approx(task["target_mae_generated"], rel=1e-5)
    assert task["decision_status"] in {"recommended_quality", "acceptable_tradeoff", "aggressive_compression"}


def test_numeric_validation_classifies_empty_generated_output_as_execution_artifact_invalid(tmp_path: Path) -> None:
    ref = tmp_path / "ref.bin"
    got = tmp_path / "got.bin"
    import struct
    ref.write_bytes(struct.pack("<ff", 0.25, 0.75))
    got.write_bytes(b"")

    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=True,
        hls_ok=True,
        inference_reference_artifacts={"outputs_ref": ref, "outputs_hw": got},
    )
    payload = json.loads(artifacts["numeric_validation_json"].read_text(encoding="utf-8"))
    assert payload["status"] == "execution_artifact_invalid"
    assert payload["passed"] is False
    assert payload["inference"]["output_compare"]["status"] == "empty_generated_output"
    assert payload["inference"]["task_quality"]["decision_status"] != "recommended_quality"


def test_dataset_npz_normalization_emits_manifest_and_selected_labels(tmp_path: Path) -> None:
    import numpy as np
    from fpgai.validation.dataset import emit_dataset_artifacts

    dataset = tmp_path / "samples.npz"
    np.savez(
        dataset,
        inputs=np.arange(20, dtype=np.float32).reshape(5, 4),
        labels=np.asarray([0, 1, 2, 1, 0], dtype=np.int64),
    )
    out = tmp_path / "build"
    artifacts = emit_dataset_artifacts(
        out,
        raw_config={
            "validation": {
                "task": "classification",
                "dataset": {
                    "source": "npz",
                    "path": str(dataset),
                    "inputs_key": "inputs",
                    "labels_key": "labels",
                    "sample_selection": {"offset": 1, "count": 3},
                },
            }
        },
    )

    assert artifacts["status"] == "available"
    assert artifacts["sample_count"] == 3
    assert artifacts["input_shape"] == [4]
    manifest = json.loads(Path(artifacts["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 3
    assert manifest["input_words_per_sample"] == 4
    assert np.load(artifacts["labels_path"]).tolist() == [1, 2, 1]


def test_inference_reference_artifacts_execute_normalized_dataset_batch(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    import numpy as np
    monkeypatch.setitem(
        sys.modules,
        "fpgai.toolchain",
        types.SimpleNamespace(build_xilinx_tool_command=lambda *args, **kwargs: []),
    )
    from fpgai.engine.compiler import _emit_inference_reference_artifacts

    out = tmp_path / "out"
    out.mkdir()
    # Dataset-backed reference generation must use validation/dataset/inputs.bin
    # and must not require a legacy root-level input.bin artifact.
    assert not (out / "input.bin").exists()
    # Simulate CSim output for three samples and two classes.
    np.asarray([0.1, 0.9, 0.8, 0.2, 0.3, 0.7], dtype=np.float32).tofile(out / "output.bin")
    dataset = tmp_path / "batch.npz"
    np.savez(
        dataset,
        inputs=np.asarray([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]], dtype=np.float32),
        labels=np.asarray([1, 0, 1], dtype=np.int64),
    )

    class _Meta:
        def __init__(self, name: str, shape: list[object]) -> None:
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_inputs(self):
            return [_Meta("input", [None, 4])]

        def get_outputs(self):
            return [_Meta("output", [None, 2])]

        def run(self, outputs, feed):
            x = next(iter(feed.values()))
            assert tuple(x.shape) == (3, 4)
            return [np.asarray([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], dtype=np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=_Session))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    raw = {"validation": {"task": "classification", "dataset": {"source": "npz", "path": str(dataset)}}}

    artifacts = _emit_inference_reference_artifacts(out, model_path=model, hls_ok=True, raw_config=raw)

    assert artifacts["status"] == "available"
    assert artifacts["sample_count"] == 3
    assert artifacts["output_shape_per_sample"] == (2,)
    assert Path(artifacts["outputs_ref"]).stat().st_size == 3 * 2 * 4
    assert Path(artifacts["labels_path"]).exists()

    report = emit_numeric_validation_report(
        out,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=True,
        hls_ok=True,
        raw_config=raw,
        inference_reference_artifacts=artifacts,
    )
    payload = json.loads(report["numeric_validation_json"].read_text(encoding="utf-8"))
    task = payload["inference"]["task_quality"]
    assert task["sample_count"] == 3
    assert task["labels_status"] == "provided"
    assert task["prediction_agreement_vs_reference"] == pytest.approx(1.0)
    assert task["reference_top1_accuracy"] == pytest.approx(1.0)
    assert task["generated_top1_accuracy"] == pytest.approx(1.0)
    assert task["decision_reason"] == "accuracy drop 0% is within the recommended threshold"


def test_dataset_binary_requires_explicit_sample_shape(tmp_path: Path) -> None:
    import numpy as np
    from fpgai.validation.dataset import emit_dataset_artifacts

    path = tmp_path / "inputs.bin"
    np.arange(8, dtype=np.float32).tofile(path)
    artifacts = emit_dataset_artifacts(
        tmp_path / "out",
        raw_config={"validation": {"dataset": {"source": "binary", "path": str(path)}}},
    )
    assert artifacts["status"] == "invalid"
    assert "sample_shape" in artifacts["reason"]


def test_inference_reference_artifacts_execute_unbatched_onnx_input_for_dataset(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    import numpy as np
    monkeypatch.setitem(
        sys.modules,
        "fpgai.toolchain",
        types.SimpleNamespace(build_xilinx_tool_command=lambda *args, **kwargs: []),
    )
    from fpgai.engine.compiler import _emit_inference_reference_artifacts

    out = tmp_path / "out"
    out.mkdir()
    np.asarray([0.1, 0.9, 0.8, 0.2, 0.3, 0.7], dtype=np.float32).tofile(out / "output.bin")
    dataset = tmp_path / "batch.npz"
    np.savez(
        dataset,
        inputs=np.asarray([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]], dtype=np.float32),
        labels=np.asarray([1, 0, 1], dtype=np.int64),
    )

    class _Meta:
        def __init__(self, name: str, shape: list[object]) -> None:
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def get_inputs(self):
            return [_Meta("input", [4])]

        def get_outputs(self):
            return [_Meta("output", [2])]

        def run(self, outputs, feed):
            x = next(iter(feed.values()))
            assert tuple(x.shape) == (4,)
            self.calls += 1
            table = [
                np.asarray([0.1, 0.9], dtype=np.float32),
                np.asarray([0.8, 0.2], dtype=np.float32),
                np.asarray([0.3, 0.7], dtype=np.float32),
            ]
            return [table[self.calls - 1]]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=_Session))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    raw = {"validation": {"task": "classification", "dataset": {"source": "npz", "path": str(dataset)}}}

    artifacts = _emit_inference_reference_artifacts(out, model_path=model, hls_ok=True, raw_config=raw)

    assert artifacts["status"] == "available"
    assert artifacts["sample_count"] == 3
    assert artifacts["output_shape_per_sample"] == (2,)
    assert Path(artifacts["outputs_ref"]).stat().st_size == 3 * 2 * 4


def test_inference_reference_artifacts_reshape_flat_dataset_sample_to_static_onnx_image_shape(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    import numpy as np

    monkeypatch.setitem(
        sys.modules,
        "fpgai.toolchain",
        types.SimpleNamespace(build_xilinx_tool_command=lambda *args, **kwargs: []),
    )
    from fpgai.engine.compiler import _emit_inference_reference_artifacts

    out = tmp_path / "out"
    out.mkdir()
    np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float32).tofile(out / "output.bin")
    dataset = tmp_path / "batch.npz"
    np.savez(
        dataset,
        inputs=np.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32),
        labels=np.asarray([1, 0], dtype=np.int64),
    )

    class _Meta:
        def __init__(self, name: str, shape: list[object]) -> None:
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def get_inputs(self):
            return [_Meta("input", [1, 1, 2, 2])]

        def get_outputs(self):
            return [_Meta("output", [1, 2])]

        def run(self, outputs, feed):
            x = next(iter(feed.values()))
            assert tuple(x.shape) == (1, 1, 2, 2)
            self.calls += 1
            table = [
                np.asarray([[0.1, 0.9]], dtype=np.float32),
                np.asarray([[0.8, 0.2]], dtype=np.float32),
            ]
            return [table[self.calls - 1]]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=_Session))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    raw = {"validation": {"task": "classification", "dataset": {"source": "npz", "path": str(dataset)}}}

    artifacts = _emit_inference_reference_artifacts(out, model_path=model, hls_ok=True, raw_config=raw)

    assert artifacts["status"] == "available"
    assert artifacts["sample_count"] == 2
    assert artifacts["output_shape_per_sample"] == (2,)
    assert Path(artifacts["outputs_ref"]).stat().st_size == 2 * 2 * 4


def test_dataset_numeric_validation_emits_execution_and_class_diagnostics(tmp_path: Path) -> None:
    import json
    import struct
    from fpgai.validation.numeric import emit_numeric_validation_report

    out = tmp_path / "dataset_numeric"
    reports = out / "reports"
    reports.mkdir(parents=True)
    ref = out / "reference.bin"
    got = out / "output.bin"
    labels = out / "labels.npy"
    import numpy as np
    values = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    ref.write_bytes(struct.pack("<6f", *values))
    got.write_bytes(struct.pack("<6f", *values))
    np.save(labels, np.asarray([1, 0], dtype=np.int64))
    (reports / "hls_dataset_execution.json").write_text(json.dumps({
        "sample_count_requested": 2,
        "sample_count_executed": 2,
        "inference_invocation_count": 2,
        "output_values_per_sample": 3,
        "generated_output_words": 6,
        "weight_import_count": 0,
        "weight_export_count": 0,
    }), encoding="utf-8")

    emit_numeric_validation_report(
        out,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=True,
        hls_ok=True,
        raw_config={"validation": {"task": "classification", "labels": str(labels), "dataset": {"source": "npz"}}},
        inference_reference_artifacts={"outputs_ref": ref, "outputs_hw": got, "labels_path": labels},
    )
    payload = json.loads((reports / "numeric_validation.json").read_text())
    task = payload["inference"]["task_quality"]
    assert task["execution_validation"]["passed"] is True
    assert Path(task["confusion_matrix_path"]).exists()
    assert Path(task["per_class_accuracy_path"]).exists()
    assert task["generated_per_class_accuracy"][0]["accuracy"] == 1.0


def test_dataset_numeric_validation_rejects_incomplete_execution_record(tmp_path: Path) -> None:
    import json
    import struct
    from fpgai.validation.numeric import emit_numeric_validation_report

    out = tmp_path / "dataset_numeric_bad"
    reports = out / "reports"
    reports.mkdir(parents=True)
    ref = out / "reference.bin"
    got = out / "output.bin"
    labels = out / "labels.npy"
    import numpy as np
    values = [0.0, 1.0, 1.0, 0.0]
    ref.write_bytes(struct.pack("<4f", *values))
    got.write_bytes(struct.pack("<4f", *values))
    np.save(labels, np.asarray([1, 0], dtype=np.int64))
    (reports / "hls_dataset_execution.json").write_text(json.dumps({
        "sample_count_requested": 2,
        "sample_count_executed": 1,
        "inference_invocation_count": 1,
        "output_values_per_sample": 2,
        "generated_output_words": 2,
    }), encoding="utf-8")

    emit_numeric_validation_report(
        out,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=True,
        hls_ok=True,
        raw_config={"validation": {"task": "classification", "labels": str(labels), "dataset": {"source": "npz"}}},
        inference_reference_artifacts={"outputs_ref": ref, "outputs_hw": got, "labels_path": labels},
    )
    payload = json.loads((reports / "numeric_validation.json").read_text())
    task = payload["inference"]["task_quality"]
    assert task["status"] == "execution_record_invalid"
    assert task["execution_validation"]["passed"] is False


def test_torchvision_mnist_balanced_selection_emits_provenance(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    import numpy as np
    from fpgai.validation.dataset import emit_dataset_artifacts

    class _FakeMNIST:
        classes = [str(index) for index in range(10)]

        def __init__(self, *, root: str, train: bool, download: bool) -> None:
            assert train is False
            assert download is False
            self.data = np.arange(20 * 28 * 28, dtype=np.uint8).reshape(20, 28, 28)
            self.targets = np.repeat(np.arange(10, dtype=np.int64), 2)

    fake_torchvision = types.SimpleNamespace(
        __version__="0.test",
        datasets=types.SimpleNamespace(MNIST=_FakeMNIST, FashionMNIST=_FakeMNIST),
    )
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)

    artifacts = emit_dataset_artifacts(
        tmp_path / "out",
        raw_config={
            "validation": {
                "task": "classification",
                "dataset": {
                    "source": "torchvision",
                    "name": "MNIST",
                    "root": str(tmp_path / "datasets"),
                    "split": "test",
                    "download": False,
                    "sample_selection": {
                        "mode": "balanced_per_class",
                        "count": 10,
                        "seed": 7,
                        "per_class_count": 1,
                    },
                    "preprocessing": {
                        "normalize": True,
                        "flatten": True,
                    },
                },
            }
        },
    )

    assert artifacts["status"] == "available"
    assert artifacts["sample_count"] == 10
    assert artifacts["input_shape"] == [784]
    labels = np.load(artifacts["labels_path"])
    assert sorted(labels.tolist()) == list(range(10))
    inputs = np.load(artifacts["inputs_path"])
    assert inputs.shape == (10, 784)
    assert float(inputs.min()) >= 0.0
    assert float(inputs.max()) <= 1.0
    manifest = json.loads(Path(artifacts["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["source"] == "torchvision"
    assert manifest["provenance"]["dataset_name"] == "MNIST"
    assert manifest["provenance"]["split"] == "test"
    assert manifest["provenance"]["download"] is False
    assert manifest["selection"]["mode"] == "balanced_per_class"
    assert manifest["class_distribution"] == {str(index): 1 for index in range(10)}


def test_torchvision_adapter_reports_missing_optional_dependency(tmp_path: Path, monkeypatch) -> None:
    import importlib
    from fpgai.validation.dataset import emit_dataset_artifacts

    original = importlib.import_module

    def _import(name: str, *args, **kwargs):
        if name == "torchvision":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _import)
    artifacts = emit_dataset_artifacts(
        tmp_path / "out",
        raw_config={
            "validation": {
                "dataset": {
                    "source": "torchvision",
                    "name": "MNIST",
                    "root": str(tmp_path / "datasets"),
                    "download": False,
                }
            }
        },
    )

    assert artifacts["status"] == "invalid"
    assert "optional 'datasets' dependencies" in artifacts["reason"]
