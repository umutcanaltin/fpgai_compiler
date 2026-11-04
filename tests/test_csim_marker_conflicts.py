"""Contradictory C-simulation markers and impossible metrics cannot establish E2."""
import pytest
from fpgai_research.toolchains.vitis_hls.report import discover_csim_success


def log(tmp_path, text):
    folder = tmp_path / 'logs'
    folder.mkdir()
    (folder / 'vitis.stdout.log').write_text(text)
    return discover_csim_success(tmp_path)


def test_failure_marker_overrides_success_marker(tmp_path):
    record = log(tmp_path,
        'FPGAI_METRICS samples=1 outputs=1 cosine=1 rmse=0 maxabs=0 failures=0\n'
        'FPGAI_CSIM_STATUS=completed\nFPGAI_CSIM_STATUS=failed\n')
    assert record['marker'] == 'FPGAI_CSIM_STATUS=failed'
    assert record['passed'] is False


@pytest.mark.parametrize('field', ('cosine=2', 'rmse=-1', 'maxabs=-0.01'))
def test_impossible_metrics_rejected(tmp_path, field):
    metric = {'cosine=2': 'cosine=2 rmse=0 maxabs=0',
              'rmse=-1': 'cosine=1 rmse=-1 maxabs=0',
              'maxabs=-0.01': 'cosine=1 rmse=0 maxabs=-0.01'}[field]
    receipt = log(tmp_path, f'FPGAI_METRICS samples=1 outputs=1 {metric} failures=0\nFPGAI_CSIM_STATUS=completed\n')
    assert receipt['metrics_missing'] is True
