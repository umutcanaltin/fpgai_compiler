"""Old, mismatched and freshly attested C-simulation logs are distinguishable."""
from hashlib import sha256
import json
from pathlib import Path
import stat

from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis


def _case(root: Path):
    (root/'model.cpp').write_text('void placeholder() {}\n')
    (root/'tb.cpp').write_text('// testbench\n')
    (root/'run_hls.tcl').write_text('exit\n')
    (root/'manifest.json').write_text(json.dumps({
        'cpp_sha256': sha256((root/'model.cpp').read_bytes()).hexdigest(),
        'testbench_sha256': sha256((root/'tb.cpp').read_bytes()).hexdigest(),
    }))
    return root


def _tool(path, output):
    path.write_text('#!/bin/sh\n' + output + '\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_fresh_mock_stdout_does_not_prove_native_csimulation(tmp_path):
    root = _case(tmp_path)
    fake = _tool(tmp_path/'mock_hls',
        'echo "FPGAI_METRICS samples=1 outputs=1 cosine=1 rmse=0 maxabs=0 failures=0"; '
        'echo "FPGAI_CSIM_STATUS=completed"')
    run_vitis(root, executable=fake)
    record = summarize_case(str(root))
    assert record['evidence'] == 'E1-HLS-source'
    assert record['csim_passed'] is None
    assert any('identity/version' in msg for msg in record['warnings'])


def test_modified_stdout_cannot_be_promoted(tmp_path):
    root = _case(tmp_path)
    fake = _tool(tmp_path/'mock_hls',
        'echo "FPGAI_METRICS samples=1 outputs=1 cosine=1 rmse=0 maxabs=0 failures=0"; '
        'echo "FPGAI_CSIM_STATUS=completed"')
    run_vitis(root, executable=fake)
    (root/'logs'/'vitis.stdout.log').write_text(
        (root/'logs'/'vitis.stdout.log').read_text() + '\nModified after invocation\n')
    record = summarize_case(str(root))
    assert record['evidence'] == 'E1-HLS-source'
    assert any('fresh log' in w for w in record['warnings'])


def test_stale_csim_report_not_reused_by_unrelated_tool_run(tmp_path):
    root = _case(tmp_path)
    old = root/'fpgai_hls_project'/'solution1'/'csim'/'report'/'fpgai_model_csim.log'
    old.parent.mkdir(parents=True)
    old.write_text('FPGAI_METRICS samples=1 outputs=1 cosine=1 rmse=0 maxabs=0 failures=0\n'
                   'FPGAI_CSIM_STATUS=completed\n')
    fake = _tool(tmp_path/'mock_hls', 'echo synthesis_only')
    receipt = run_vitis(root, executable=fake)
    assert 'csim_log' not in receipt['fresh_report_sha256']
    record = summarize_case(str(root))
    assert record['evidence'] == 'E1-HLS-source'
    assert any('fresh log' in w for w in record['warnings'])
