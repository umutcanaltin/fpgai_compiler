"""Scientific decision-interface campaign: complete denominators and no RL claims."""
from __future__ import annotations

import csv
import json

import pytest

from fpgai_research.experiments.cst_decision_campaign import run


def test_prespecified_policy_neutral_campaign_and_original_numerics(tmp_path):
    target=tmp_path/"decision"
    result=run(output=target,graphs=2,host_checks=1,seed=37)
    assert result["graphs"]==2
    assert result["candidate_actions"]==26
    assert result["independent_restricted_oracle_agreements"]==26
    assert result["valid_actions"]+result["rejected_actions"]==26
    assert result["false_legal"]==0
    assert result["false_illegal"]==0
    assert result["host_numerical_passes"]==1
    assert result["actual_mapped_value_lifetime_passes"]==1
    assert result["policy_trajectories"]==4
    assert result["native_vitis_calls"]==0
    assert result["trained_rl_policies"]==0
    with (target/"cases.csv").open() as file:
        assert len(list(csv.DictReader(file)))==26
    assert json.loads((target/"summary.json").read_text())==result
    assert len(json.loads((target/"policies.json").read_text()))==4
    assert len(json.loads((target/"qualified_source_checks.json").read_text()))==1
    with pytest.raises(FileExistsError,match="Create-only"):
        run(output=target,graphs=1,host_checks=0)


@pytest.mark.parametrize("graphs,checks,seed",[
    (0,0,37),(5,0,37),(1,2,37),(1,-1,37),(1,0,-1),
])
def test_campaign_rejects_invalid_scientific_limits(tmp_path,graphs,checks,seed):
    with pytest.raises(ValueError,match="graphs|host_checks|seed"):
        run(output=tmp_path/"invalid",graphs=graphs,
            host_checks=checks,seed=seed)
