"""CPU-only research reward-profile tests, never original HLS measurements."""
from __future__ import annotations

from math import isfinite

import pytest

from rl_environment.reward import (
    DEFAULT_HARDWARE_REFERENCE_NS, HARDWARE_REWARD_MODES,
    hardware_reward_contract, hardware_trial_reward,
    latency_only_reward,
)


def _reward(mode, *, previous=None, best=None, latency=None, qualified=False):
    return hardware_trial_reward(
        mode=mode, previous_best_ns=previous, updated_best_ns=best,
        qualified=qualified, evaluated_latency_ns=latency,
        reference_ns=10000.,
    )


@pytest.mark.parametrize("mode", HARDWARE_REWARD_MODES)
def test_failed_hls_actions_produce_original_negative_feedback(mode):
    assert _reward(mode, qualified=False) == -.05
    assert _reward(mode, previous=9000., best=9000., qualified=False) == -.05
    with pytest.raises(ValueError, match="no qualified latency"):
        _reward(mode, qualified=False, latency=2000.)


def test_legacy_reward_mode_remains_exactly_identical_to_original_baseline():
    assert _reward("best_improvement", best=9072., latency=9072.,
                   qualified=True) == latency_only_reward(
                       None, 9072., qualified=True
                   ) == -.005
    assert _reward("best_improvement", previous=10000., best=9000.,
                   latency=9000., qualified=True) == latency_only_reward(
                       10000., 9000., qualified=True
                   )
    assert _reward("best_improvement", previous=10000., best=10000.,
                   latency=12000., qualified=True) == -.005


def test_dense_reward_is_monotone_in_each_original_qualified_latency():
    first = _reward("dense_latency", best=9072., latency=9072., qualified=True)
    faster = _reward("dense_latency", previous=9072., best=6000.,
                     latency=6000., qualified=True)
    slower = _reward("dense_latency", previous=9072., best=9072.,
                     latency=15000., qualified=True)
    assert isfinite(first)
    assert 0. < slower < first < faster < 1.
    assert first == 10000. / (10000. + 9072.) - .005
    assert DEFAULT_HARDWARE_REFERENCE_NS == 10000.


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_reward_reference_rejects_ambiguous_or_invalid_values(value):
    with pytest.raises(ValueError, match="positive and finite"):
        hardware_reward_contract(mode="dense_latency", reference_ns=value)


def test_reward_requires_actual_audited_positive_feasible_latency():
    with pytest.raises(ValueError, match="actual positive"):
        _reward("dense_latency", best=1., latency=None, qualified=True)
    with pytest.raises(ValueError, match="actual positive"):
        _reward("best_improvement", best=1., latency=-1., qualified=True)
    with pytest.raises(ValueError, match="original HLS qualification"):
        hardware_trial_reward(
            mode="best_improvement", previous_best_ns=None,
            updated_best_ns=None, qualified=1, evaluated_latency_ns=None,
        )
