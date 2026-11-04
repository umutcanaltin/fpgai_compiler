"""Audit guardrails are CPU-only; no Vitis and no evidence mutation."""
import pytest
from rl_environment.audit import audit, _expected_episodes
from rl_environment.cohort import make_cohort


def test_audit_expected_episode_interleaving():
    cohort=make_cohort(training=2,validation=1,test=1,
                       seed=20261030,max_vitis_calls=2)
    rows=list(_expected_episodes(cohort,1))
    assert len(rows)==8
    assert [(row[0],row[2]) for row in rows]==[
        ("train","online_drl"),
        ("validation_after_00001","online_drl"),
        ("validation_random_after_00001","random"),
        ("train","online_drl"),
        ("validation_after_00002","online_drl"),
        ("validation_random_after_00002","random"),
        ("test_rl","online_drl"),
        ("test_random","random"),
    ]


def test_audit_requires_real_original_campaign(tmp_path):
    with pytest.raises(ValueError,match="study directory"):
        audit(tmp_path/"missing")
    study=tmp_path/"incomplete"
    study.mkdir()
    with pytest.raises(ValueError,match="campaign.json"):
        audit(study)


def test_audit_rejects_fake_source_proxy_campaign(tmp_path):
    import json
    root=tmp_path/"fake"
    root.mkdir()
    (root/"campaign.json").write_text(json.dumps({
        "schema":"fpgai-research/real-online-post-hls-campaign/v1",
        "source_proxy_reward_used":True,
        "no_post_vivado_execution":True,
    }))
    with pytest.raises(ValueError,match="proxy/native confusion"):
        audit(root)
