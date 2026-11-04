"""A complete and attested CSim run must evaluate every prepared sample/output."""
from hashlib import sha256
import json
import stat
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis


def prepare(tmp_path, expected_samples=4, expected_outputs=3):
    model=tmp_path/'model.cpp'
    model.write_text('void fpgai_model() {}')
    bench=tmp_path/'tb.cpp'
    bench.write_text('// numeric bench')
    (tmp_path/'run_hls.tcl').write_text('exit\n')
    (tmp_path/'manifest.json').write_text(json.dumps({
        'cpp_sha256':sha256(model.read_bytes()).hexdigest(),
        'testbench_sha256':sha256(bench.read_bytes()).hexdigest(),
        'csim_sample_count':expected_samples,
        'csim_outputs_per_sample':expected_outputs,
    }))
    return tmp_path


def mock(tmp_path, samples, outputs):
    executable=tmp_path/'mock_vitis'
    executable.write_text('#!/bin/sh\n'
      f'echo "FPGAI_METRICS samples={samples} outputs={outputs} cosine=1 rmse=0 maxabs=0 failures=0"\n'
      'echo "FPGAI_CSIM_STATUS=completed"\n')
    executable.chmod(executable.stat().st_mode|stat.S_IXUSR)
    return str(executable)


def test_mock_csim_has_metrics_but_not_native_evidence(tmp_path):
    p=prepare(tmp_path)
    run_vitis(p,executable=mock(tmp_path,4,3))
    r=summarize_case(str(p))
    assert r['csim_passed'] is None
    assert r['evidence']=='E1-HLS-source'


def test_csim_missing_samples_does_not_gain_e2(tmp_path):
    p=prepare(tmp_path)
    run_vitis(p,executable=mock(tmp_path,1,3))
    r=summarize_case(str(p))
    assert r['evidence']=='E1-HLS-source'
    assert any('identity/version' in w for w in r['warnings'])


def test_csim_wrong_output_count_does_not_gain_e2(tmp_path):
    p=prepare(tmp_path)
    run_vitis(p,executable=mock(tmp_path,4,1))
    r=summarize_case(str(p))
    assert r['evidence']=='E1-HLS-source'
    assert any('identity/version' in w for w in r['warnings'])
