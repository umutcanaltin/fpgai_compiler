"""Trained-model HLS source preparation never invents private evidence."""
import pytest

from fpgai_research.experiments.cst_trained_mlp_native_prepare import prepare


@pytest.mark.parametrize("n", [0,5,-1,True])
def test_native_source_prep_rejects_unsupported_sample_count(tmp_path,n):
    with pytest.raises(ValueError,match="sample_count"):
        prepare(trained_directory=tmp_path/"missing",output=tmp_path/"study",
                sample_count=n)


def test_original_trained_assets_are_required_and_no_synthetic_replacement(tmp_path):
    output=tmp_path/"study"
    with pytest.raises(ValueError,match="Missing or linked trained"):
        prepare(trained_directory=tmp_path/"missing",output=output,sample_count=1)
    assert not output.exists()


@pytest.mark.parametrize("period,uncertainty",[
    (5.0,8.64), (0.0,0.0), (float("nan"),1.0)])
def test_invalid_original_clock_constraints_rejected(tmp_path,period,uncertainty):
    with pytest.raises(ValueError,match="clock"):
        prepare(trained_directory=tmp_path/"missing",output=tmp_path/"study",
                clock_ns=period,clock_uncertainty_ns=uncertainty)
