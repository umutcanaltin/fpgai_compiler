"""Scientific single-command compiler suite: safety, reproducibility and CLI.

Does not execute Vitis, modify RL, call GitHub Actions, or require tokens.
The user runs source-only tests on the local Linux workstation.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    file = Path(__file__).resolve().parents[1] / "experiments/compiler_ir/series.py"
    spec = importlib.util.spec_from_file_location("compiler_ir_series", file)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_paper_series_covers_every_available_compiler_experiment(tmp_path):
    sut = _module()
    steps = sut.plan(tmp_path, "paper", onnx=True)
    names = [s.name for s in steps]
    assert len(names) == len(set(names))
    assert {"compiler_tests", "onnx_crosscheck", "array", "row", "chain",
            "cpu_paper_table", "schedule_mutations", "independent_oracle",
            "scaling", "nonlinear", "tensor_source", "tensor_source_audit",
            "tensor_source_report", "row_block_source",
            "row_block_source_report"} <= set(names)
    assert "tensor_native_execute" not in names
    assert "row_block_native_execute" not in names
    assert all("rl_environment" not in " ".join(s.command) for s in steps)
    assert all("--execute-native" not in s.command for s in steps)
    steps_by_name = {step.name: step for step in steps}
    assert steps_by_name["cpu_paper_table"].dependencies == (
        "array", "row", "chain")
    assert "--max-cases" in steps_by_name["array"].command
    assert "96" in steps_by_name["array"].command
    assert "fpgai_research.experiments.cst_native_campaign" in (
        " ".join(steps_by_name["tensor_source"].command))
    assert list(tmp_path.iterdir()) == []


def test_full_series_requires_explicit_native_selection_and_models(tmp_path):
    sut = _module()
    steps = sut.plan(
        tmp_path, "quick", trained_mlp="/private/mnist_mlp",
        trained_cnn="/private/mnist_cnn", with_native=True)
    names = {s.name for s in steps}
    assert {"trained_mlp_preflight", "trained_mlp_host",
            "trained_mlp_native_source", "trained_cnn_preflight",
            "tensor_native_execute", "row_block_native_execute",
            "tensor_native_report", "row_block_native_report"} <= names
    native = [s for s in steps if s.evidence == "E2/E3-original-vitis"]
    assert len(native) == 2
    assert all("--execute-native" in s.command for s in native)
    assert all(s.dependencies for s in native)
    assert list(tmp_path.iterdir()) == []


def test_plan_cli_is_read_only_and_disallows_unconfirmed_native(capsys, tmp_path):
    sut = _module()
    r = sut.main(["--profile", "quick", "--plan", "--skip-onnx",
                  "--output", str(tmp_path/"unused")])
    assert r == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["profile"] == "quick"
    assert shown["source_revision"]
    assert not (tmp_path/"unused").exists()
    assert sut.main(["--plan", "--with-native",
                     "--output", str(tmp_path/"no_native")]) == 2
    assert not (tmp_path/"no_native").exists()


def test_source_output_attestation_does_not_follow_symlinks(tmp_path):
    sut = _module()
    folder = tmp_path/"real"
    folder.mkdir()
    (folder/"data.json").write_text('{"status":"tested"}\n')
    output = sut._artifacts(folder)
    assert len(output) == 1
    assert output[0]["bytes"] > 0
    assert len(output[0]["sha256"]) == 64
    (folder/"malicious").symlink_to(folder/"data.json")
    with pytest.raises(ValueError, match="Symlink"):
        sut._artifacts(folder)


def test_config_profiles_are_bounded_and_not_free_form_shell_scripts():
    sut = _module()
    conf = sut._load_config()
    assert conf["schema"].endswith("v1")
    assert conf["profiles"]["quick"]["array_cases"] < conf["profiles"]["paper"]["array_cases"]
    assert conf["native"]["clock_uncertainty_ns"] < conf["native"]["clock_ns"]
    assert "rl_environment" not in sut.CONFIG.read_text()


def test_series_pins_source_and_rejects_dirty_checkout(monkeypatch, tmp_path, capsys):
    sut = _module()
    original = sut._git
    def dirty(*args):
        if args[0] == "status":
            return " M src/fpgai_research/feature.py"
        return original(*args)
    monkeypatch.setattr(sut, "_git", dirty)
    result = sut.main(["--profile", "quick", "--output", str(tmp_path/"study")])
    assert result == 2
    assert not (tmp_path/"study").exists()
    assert "Tracked changes" in capsys.readouterr().err


def test_cst_three_paper_contributions_are_in_both_original_cpu_profiles(tmp_path):
    """C1/C2 original experiments feed C3's policy-neutral interface tests."""
    sut = _module()
    for profile in ("quick","paper"):
        stages = sut.plan(tmp_path,profile,onnx=False)
        by_name = {stage.name:stage for stage in stages}
        assert {"compiler_tests","nchw_real_source_rw_e1",
                "two_phi_readers_e1","two_phi_counterfactual_e1",
                "decision_interface","rl_compatible_ir_r3_r4",
                "rl_compatible_ir_e1_action_qualification",
                "multi_reader_certified_actions_e1",
                "unified_policy_interface_e1",
                "accounting_baseline_sensitivity_e0",
                "emitter_shadow_mutation_e1",
                "phi_scaling_sensitivity_e0",
                "tensor_exact_comparator_e0",
                "hls_mechanism_factorial_source_e1",
                "evidence_index"}.issubset(by_name)
        assert by_name["rl_compatible_ir_r3_r4"].dependencies == (
            "compiler_tests","decision_interface")
        assert by_name["rl_compatible_ir_e1_action_qualification"].dependencies == (
            "compiler_tests","decision_interface","rl_compatible_ir_r3_r4")
        assert "g++" in by_name["rl_compatible_ir_e1_action_qualification"].executables
        assert by_name["multi_reader_certified_actions_e1"].dependencies == (
            "compiler_tests","rl_compatible_ir_e1_action_qualification")
        assert "g++" in by_name["multi_reader_certified_actions_e1"].executables
        assert "--graphs" in by_name["multi_reader_certified_actions_e1"].command
        assert ("3" if profile == "paper" else "1") in (
            by_name["multi_reader_certified_actions_e1"].command)
        assert by_name["unified_policy_interface_e1"].dependencies == (
            "compiler_tests","rl_compatible_ir_e1_action_qualification",
            "multi_reader_certified_actions_e1")
        assert "g++" in by_name["unified_policy_interface_e1"].executables
        assert by_name["accounting_baseline_sensitivity_e0"].dependencies == (
            "compiler_tests","multi_reader_certified_actions_e1")
        assert by_name["accounting_baseline_sensitivity_e0"].executables == ()
        assert "--execute-native" not in (
            by_name["accounting_baseline_sensitivity_e0"].command)
        assert by_name["emitter_shadow_mutation_e1"].dependencies == (
            "compiler_tests","accounting_baseline_sensitivity_e0")
        assert by_name["emitter_shadow_mutation_e1"].executables == ("g++",)
        assert "--graphs" in by_name["emitter_shadow_mutation_e1"].command
        assert "--mutants-per-graph" in by_name["emitter_shadow_mutation_e1"].command
        assert ("36" if profile=="paper" else "4") in (
            by_name["emitter_shadow_mutation_e1"].command)
        assert by_name["phi_scaling_sensitivity_e0"].dependencies == (
            "compiler_tests","accounting_baseline_sensitivity_e0")
        assert by_name["phi_scaling_sensitivity_e0"].executables == ()
        assert "--shape-limit" in by_name["phi_scaling_sensitivity_e0"].command
        assert ("7" if profile == "paper" else "1") in (
            by_name["phi_scaling_sensitivity_e0"].command)
        assert by_name["tensor_exact_comparator_e0"].dependencies == (
            "compiler_tests","accounting_baseline_sensitivity_e0")
        assert by_name["tensor_exact_comparator_e0"].executables == ()
        assert by_name["hls_mechanism_factorial_source_e1"].dependencies == (
            "compiler_tests","tensor_exact_comparator_e0")
        assert by_name["hls_mechanism_factorial_source_e1"].executables == ()
        assert "--execute-native" not in (
            by_name["hls_mechanism_factorial_source_e1"].command)
        assert by_name["two_phi_counterfactual_e1"].dependencies == (
            "two_phi_readers_e1",)
        command = by_name["two_phi_counterfactual_e1"].command
        assert "--input-csv" in command
        assert str(tmp_path/"two_phi_readers_e1"/"cases.csv") in command
        assert "--output" in command
        assert by_name["evidence_index"] == stages[-1]
        for name in ("two_phi_counterfactual_e1","rl_compatible_ir_r3_r4",
                     "rl_compatible_ir_e1_action_qualification",
                     "multi_reader_certified_actions_e1",
                     "unified_policy_interface_e1",
                     "accounting_baseline_sensitivity_e0",
                     "emitter_shadow_mutation_e1",
                     "phi_scaling_sensitivity_e0",
                     "tensor_exact_comparator_e0",
                     "hls_mechanism_factorial_source_e1"):
            step=by_name[name]
            assert step.output is not None
            assert "--execute-native" not in step.command
            assert "rl_environment" not in " ".join(step.command)
    assert list(tmp_path.iterdir())==[]
