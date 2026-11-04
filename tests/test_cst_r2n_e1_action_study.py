"""R2n source-only C3 reproducibility, negative and original GCC controls."""
from __future__ import annotations

import json
import shutil

import pytest

from fpgai_research.experiments import cst_r2n_e1_action_study as r2n


@pytest.mark.skipif(shutil.which("g++") is None, reason="Original GCC required")
def test_original_r2n_one_graph_actual_cpp_e1_and_unknown(tmp_path):
    out = tmp_path / "original"
    result = r2n.run(output=out, graphs=1)
    assert result["schema"] == r2n.SCHEMA
    assert result["graph_depths"] == [2]
    assert result["all_candidate_actions"] == 3
    assert result["e1_verified_actions"] == 1
    assert result["e1_host_executions_attempted"] == 1
    assert result["rejected_actions"] == 1
    assert result["source_only_uncertified_actions"] == 1
    assert result["failed_e1_actions"] == 0
    assert result["native_vitis_calls"] == 0
    assert result["rl_policies_trained"] == 0
    assert json.loads((out / "summary.json").read_text()) == result
    cases = json.loads((out / "case_evidence.json").read_text())
    assert len(cases) == 1 and cases[0]["graph_depth"] == 2
    assert len(cases[0]["actions"]) == 3
    qualified = cases[0]["actions"][0]
    assert qualified["status"] == "E1_GCC_SOURCE_QUALIFIED"
    assert len(qualified["source_trace_sha256"]) == 64
    assert qualified["host_max_absolute_error"] >= 0
    assert cases[0]["actions"][1]["status"] == "REJECTED"
    assert cases[0]["actions"][2]["status"] == "UNKNOWN_E0_CERTIFICATE"
    with pytest.raises(FileExistsError, match="create-only"):
        r2n.run(output=out, graphs=1)


def test_rejected_unknown_and_failed_e1_remain_separate_in_denominator(
        monkeypatch, tmp_path):
    from fpgai_research.experiments import cst_r2n_e1_action_study as module
    calls = []
    def fake(state, actions, *, runtime_input, max_e1_executions):
        calls.append(len(actions))
        return {
            "parent_state_sha256": "a" * 64,
            "rows": [
                {"status": "FAILED_E1_SOURCE_EXECUTION"},
                {"status": "REJECTED"},
                {"status": "UNKNOWN_E0_CERTIFICATE"},
            ],
            "e1_executions_attempted": 1,
            "native_vitis_calls": 0,
            "rl_policies_trained": 0,
            "physical_feasibility_checked": False,
        }
    monkeypatch.setattr(module, "source_executed_action_mask", fake)
    out = tmp_path / "new"
    result = r2n.run(output=out, graphs=3)
    assert calls == [3, 3, 3]
    assert result["e1_verified_actions"] == 0
    assert result["failed_e1_actions"] == 3
    assert result["source_only_uncertified_actions"] == 3
    assert result["rejected_actions"] == 3
    assert result["e1_host_executions_attempted"] == 3
    assert result["all_candidate_actions"] == 9
    assert json.loads((out / "summary.json").read_text()) == result


def test_graph_count_and_create_only_preflight_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="1 through 3"):
        r2n.run(output=tmp_path / "bad0", graphs=0)
    with pytest.raises(ValueError, match="1 through 3"):
        r2n.run(output=tmp_path / "bad4", graphs=4)
    with pytest.raises(ValueError, match="1 through 3"):
        r2n.run(output=tmp_path / "badbool", graphs=True)
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        r2n.run(output=existing)


def test_native_or_trained_claims_are_never_accepted(monkeypatch, tmp_path):
    from fpgai_research.experiments import cst_r2n_e1_action_study as module
    def forged(*args, **kwargs):
        return {
            "parent_state_sha256": "a" * 64,
            "rows": [{"status": "E1_GCC_SOURCE_QUALIFIED"}] * 3,
            "e1_executions_attempted": 3,
            "native_vitis_calls": 1,
            "rl_policies_trained": 0,
            "physical_feasibility_checked": False,
        }
    monkeypatch.setattr(module, "source_executed_action_mask", forged)
    out = tmp_path / "not_created"
    with pytest.raises(ValueError, match="native or trained"):
        r2n.run(output=out, graphs=1)
    assert not out.exists()



def test_original_source_git_preflight_blocks_dirty_checkout_before_gcc(
        monkeypatch, tmp_path):
    from types import SimpleNamespace
    import fpgai_research.experiments.cst_r2n_e1_action_study as module
    original = module.subprocess.run
    def dirty(command, **kwargs):
        if "status" in command:
            return SimpleNamespace(stdout=" M src/fpgai_research/example.py\n")
        return original(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", dirty)
    def forbidden(*args, **kwargs):
        pytest.fail("No GCC action study may start on dirty tracked source")
    monkeypatch.setattr(module, "source_executed_action_mask", forbidden)
    out = tmp_path / "blocked"
    with pytest.raises(RuntimeError, match="clean tracked Git revision"):
        r2n.run(output=out, graphs=1)
    assert not out.exists()
