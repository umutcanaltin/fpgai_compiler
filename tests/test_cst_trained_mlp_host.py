"""Trained MNIST CST experiment cannot fabricate private weights or evidence."""
import pytest

from fpgai_research.experiments.cst_trained_mlp_host import run


@pytest.mark.parametrize("count", [0, 9, -1, 1.5])
def test_trained_host_rejects_invalid_sample_budget(tmp_path, count):
    with pytest.raises(ValueError, match="host_samples"):
        run(trained_directory=tmp_path/"missing", output=tmp_path/"result",
            host_samples=count)


def test_real_trained_assets_required_and_no_output_is_created(tmp_path):
    result = tmp_path/"result"
    with pytest.raises(ValueError, match="Missing or linked trained"):
        run(trained_directory=tmp_path/"missing", output=result,
            host_samples=1)
    assert not result.exists()


@pytest.mark.parametrize("atol,rtol", [
    (0.0, .001), (-1.0, .001), (.001, 0.0), (.02, .001)
])
def test_numerical_qualification_requires_justified_bounded_tolerance(
        tmp_path, atol, rtol):
    with pytest.raises(ValueError, match="tolerance"):
        run(trained_directory=tmp_path/"missing", output=tmp_path/"result",
            host_samples=1, atol=atol, rtol=rtol)
