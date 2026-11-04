"""Source-only extension-contract and compiler-refactor tests, no Vitis runs."""
from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys

import pytest


def test_from_checkout_module_cli_wins_over_older_pythonpath(tmp_path):
    """Reproduce native CLI shadowed by a previous sibling editable checkout."""
    root = Path(__file__).resolve().parents[1]
    stale = tmp_path / "outdated_checkout"
    pkg = stale / "fpgai_research/toolchains/vitis_hls"
    pkg.mkdir(parents=True)
    for p in (pkg.parents[1] / "__init__.py",
              pkg.parent / "__init__.py",
              pkg / "__init__.py"):
        p.write_text("")
    (pkg / "package.py").write_text(
        "def write_vitis_package(output, model, emitted, runtime_input):\n"
        "    raise AssertionError('stale compiler must not load')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(stale)
    proc = subprocess.run(
        [sys.executable, "-m", "rl_environment.runtime_contract"],
        cwd=root, env=env, text=True, capture_output=True, timeout=45,
        check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    output = json.loads(proc.stdout)
    assert output["source_package_matches_checkout"] is True
    assert output["clock_uncertainty_keyword_supported"] is True
    assert output["original_vitis_calls"] == 0
    assert output["resolved_compiler_package"] == str(
        (root / "src/fpgai_research/toolchains/vitis_hls/package.py").resolve())


def test_source_registered_rl_algorithms_expose_fresh_agent_contract(monkeypatch):
    from rl_environment.agents import (
        create_agent, available_agents, register_agent)
    from rl_environment.agents import registry

    assert "graph_actor_critic" in available_agents()
    with pytest.raises(ValueError, match="already registered"):
        register_agent("graph_actor_critic", lambda **kwargs: None)
    with pytest.raises(ValueError, match="Unknown agent"):
        create_agent("unregistered_policy", seed=101)
    monkeypatch.setattr(registry, "_FACTORIES",
                        dict(registry._FACTORIES))
    register_agent("test_invalid_policy", lambda **kwargs: object())
    with pytest.raises(TypeError, match="must implement choose"):
        create_agent("test_invalid_policy", seed=101)
    pytest.importorskip("torch")
    agent = create_agent("graph_actor_critic", seed=101)
    assert callable(agent.choose)
    assert callable(agent.update)
    assert callable(agent.save_private)
    assert callable(agent.load_private)


def test_source_registered_experiment_configs_share_stage_contract(monkeypatch):
    from rl_environment.experiments import (
        available_experiments, experiment_config, register_experiment)
    from rl_environment.experiments import registry
    from rl_environment.gym_study import plan

    assert {f"curriculum_l{i}" for i in range(7)}.issubset(
        set(available_experiments()))
    kwargs = dict(train=100, validation=1, test=1, calls=10, seed=111)
    cfg = experiment_config("curriculum_l0", **kwargs)
    result = plan(**cfg, agent_algorithm="graph_actor_critic",
                  experiment_id="curriculum_l0")
    assert result["agent_algorithm"] == "graph_actor_critic"
    assert result["experiment_id"] == "curriculum_l0"
    assert result["training_episode_minimum"] == 100
    assert result["maximum_original_vitis_invocations"] == 1090
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    register_experiment(
        "research_variant", lambda **values:
        experiment_config("curriculum_l1", **values))
    variant = experiment_config("research_variant", **kwargs)
    assert variant["curriculum_stage"] == "L1"
    assert plan(**variant, experiment_id="research_variant")[
        "curriculum_stage"] == "L1"
    with pytest.raises(ValueError, match="already registered"):
        register_experiment("research_variant", lambda **values: {})
    with pytest.raises(ValueError, match="not registered"):
        plan(**cfg, experiment_id="typo_experiment")


def test_environment_factory_keeps_real_gym_backend_and_no_implicit_tool(tmp_path):
    pytest.importorskip("gymnasium")
    from rl_environment.environments import create_environment
    from rl_environment.cohort import make_cohort

    task = make_cohort(
        training=1, validation=1, test=1, seed=101,
        max_vitis_calls=10, clock_ns=None)["train"][0]
    env = create_environment(
        "graph_native_hls", task, tmp_path / "factory_trial")
    obs, _ = env.reset(seed=101)
    assert env.backend.calls == 0
    assert obs["node_mask"].sum() == len(task.model.operators)
    with pytest.raises(ValueError, match="Unsupported environment"):
        create_environment("unapproved_synthetic_qor", task, tmp_path/"blocked")


def test_refactored_compiler_pipeline_produces_identical_hls_source_and_receipt():
    from rl_environment.cohort import make_cohort
    from fpgai_research.experiments.architecture_knobs_v2 import (
        build_catalog, resolve_candidate, compile_candidate)
    from fpgai_research.compiler import compile_architecture
    from fpgai_research.codegen.hls.emitter import emit_hls
    from fpgai_research.model_ir.lowering import lower

    task = make_cohort(
        training=1, validation=1, test=1, seed=111, max_vitis_calls=2,
        clock_ns=None)["train"][0]
    catalog = build_catalog(task.model)
    architecture = resolve_candidate(
        task.model, catalog, catalog.baseline)
    direct = emit_hls(lower(architecture))
    compiled = compile_architecture(architecture)
    from_knobs, _, _ = compile_candidate(
        task.model, catalog, catalog.baseline)
    assert compiled.refinement_receipt["stage_invariants_verified"] is True
    assert compiled.emission.cpp == direct.cpp == from_knobs.cpp
    assert compiled.emission.manifest == direct.manifest == from_knobs.manifest


def test_vitis_script_refactor_preserves_public_exact_tcl_bytes():
    from fpgai_research.toolchains.vitis_hls import package, tcl
    for arguments in (
        {}, {"clock_ns": 32.0},
        {"clock_ns": 32.0, "clock_uncertainty_ns": 8.64},
        {"clock_ns": 10., "run_csim": False, "run_csynth": True},
    ):
        assert package.make_vitis_tcl(**arguments) == tcl.make_vitis_tcl(
            **arguments)
        assert package.make_vitis_tcl(**arguments).endswith("\n")


def test_new_operator_cannot_be_labeled_native_ready_by_passing_codegen_alone():
    from fpgai_research.compiler.operator_contribution import (
        OperatorContributionChecklist)

    missing = OperatorContributionChecklist(
        name="NewLayer", frontend_shape_and_attributes=True,
        functional_model_ir=True, reference_numerical_semantics=False,
        architecture_legality=False, lowering_preserves_refinement=False,
        hls_source_emitter=True, source_and_csim_tests=False,
        independently_audited_native_hls_example=False)
    assert "reference_numerical_semantics" in missing.missing()
    assert "independently_audited_native_hls_example" in missing.missing()
    with pytest.raises(ValueError, match="not qualified"):
        missing.assert_native_ready()
