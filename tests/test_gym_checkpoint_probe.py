"""Pure no-native tests of fixed-future GNN checkpoint feasibility evaluation."""
import pytest


def test_fixed_future_feasibility_metrics_require_hls_truth_and_holdout():
    from rl_environment.gym_checkpoint_probe import _metrics
    stats = _metrics([.1, .3, .7, .9], [0,0,1,1])
    assert stats["cases"] == 4
    assert stats["qualified"] == 2
    assert stats["feasibility_fraction"] == .5
    assert stats["roc_auc"] == 1.0
    assert stats["raw_weighted_brier"] == pytest.approx(.05)
    assert stats["constant_zero_brier_reference"] == .5
    assert stats["corrected_brier"] is not None
    assert _metrics([.1,.2], [0,0])["roc_auc"] is None


def test_fixed_future_checkpoint_probe_is_create_only_and_validates_source(
        tmp_path):
    from rl_environment.gym_checkpoint_probe import probe
    with pytest.raises(FileNotFoundError):
        probe(tmp_path/"missing_original",tmp_path/"new_output")
    assert not (tmp_path/"new_output").exists()
