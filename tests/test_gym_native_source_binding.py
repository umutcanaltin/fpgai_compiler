"""Prevent stale installed fpgai_research packages from consuming native HLS budgets."""
import inspect
import json

import pytest


def test_real_compiler_package_is_current_checkout_and_supports_native_uncertainty():
    from rl_environment.runtime_contract import verify_local_compiler_package
    result = verify_local_compiler_package()
    assert result["source_package_matches_checkout"]
    assert result["clock_uncertainty_keyword_supported"]
    assert result["original_vitis_calls"] == 0


def test_detect_stale_installed_package_before_any_vendor_calls(monkeypatch):
    from fpgai_research.toolchains.vitis_hls import package as pkg
    from rl_environment.runtime_contract import verify_local_compiler_package
    monkeypatch.setattr(pkg, "__file__", "/old_install/fpgai_research/toolchains/vitis_hls/package.py")
    with pytest.raises(RuntimeError, match="outdated/different installed"):
        verify_local_compiler_package()


def test_detect_old_compiler_writer_signature_before_any_vendor_calls(monkeypatch):
    from fpgai_research.toolchains.vitis_hls import package as pkg
    from rl_environment.runtime_contract import verify_local_compiler_package

    def old_writer(output, model, emitted, runtime_input, clock_ns=5.):
        raise AssertionError("This old signature must never be invoked")

    monkeypatch.setattr(pkg, "write_vitis_package", old_writer)
    with pytest.raises(RuntimeError, match="lacks clock_uncertainty_ns"):
        verify_local_compiler_package()


def test_smoke_preflight_cannot_leave_half_created_study_on_stale_package(
        monkeypatch, tmp_path):
    from rl_environment import runtime_contract
    from rl_environment.gym_study import run

    def stale():
        raise RuntimeError("outdated/different installed compiler")

    monkeypatch.setattr(runtime_contract, "verify_local_compiler_package", stale)
    dest = tmp_path / "never_created_native_study"
    with pytest.raises(RuntimeError, match="outdated/different installed"):
        run(
            output=dest, training=2, validation=1, test=1,
            calls_per_network=2, clocks_ns=(32.,), seeds=(111,),
            freeze_numeric_format=True, beginner_pilot=True,
            curriculum_stage="L0", curriculum_mode="smoke",
            execute_native=True, max_total_vitis_calls=22,
            timeout_seconds=7200)
    assert not dest.exists()


def test_complete_native_packaging_accepts_uncertainty_and_writes_source_evidence(
        tmp_path):
    """Integration test of ACTUAL write_vitis_package, without running Vitis."""
    from rl_environment.cohort import make_cohort
    from fpgai_research.experiments.architecture_knobs_v2 import (
        build_catalog, compile_candidate)
    from fpgai_research.toolchains.vitis_hls.package import write_vitis_package

    task = make_cohort(
        training=1, validation=1, test=1,
        max_vitis_calls=2, seed=111, clock_ns=None,
        fixed_clock_ablation_ns=32., freeze_numeric_format=True,
        beginner_pilot=True, curriculum_stage="L0")["train"][0]
    catalog = build_catalog(task.model)
    emitted, _, _ = compile_candidate(task.model, catalog, catalog.baseline)
    case = tmp_path / "actual_native_package_without_tool"
    result = write_vitis_package(
        str(case), task.model, emitted, task.samples,
        part=task.part, clock_ns=32.,
        tolerance=task.max_absolute_error,
        clock_uncertainty_ns=8.64)
    assert result["explicit_clock_uncertainty_ns"] == 8.64
    assert json.loads((case / "manifest.json").read_text())[
        "explicit_clock_uncertainty_ns"] == 8.64
    script = (case / "run_hls.tcl").read_text()
    assert "create_clock -period 32 -name default" in script
    assert "set_clock_uncertainty 8.640" in script
    assert (case / "model.cpp").is_file()
    assert (case / "tb.cpp").is_file()
    assert not (case / "vitis_invocation.json").exists()



def test_python_module_cli_prefers_this_checkout_over_old_other_research_checkout(
        tmp_path):
    """Reproduce user's real native vs pytest import mismatch in a new process."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    old = tmp_path / "fpgai_research_rl_native_v8_20261030" / "src"
    fake_package = old / "fpgai_research"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        'raise RuntimeError("STALE_RESEARCH_COMPILER_WAS_IMPORTED")\n')
    env = os.environ.copy()
    # The historical .pth/PYTHONPATH can legitimately mention multiple
    # checkouts; the current RL entrypoint must pick its own sibling src/.
    env["PYTHONPATH"] = os.pathsep.join(
        (str(old), str(root), env.get("PYTHONPATH", "")))
    run = subprocess.run(
        [sys.executable, "-m", "rl_environment.runtime_contract"],
        cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == (
        "LOCAL_CURRENT_SOURCE_VITIS_PACKAGING_BINDING_VERIFIED")
    assert Path(result["resolved_compiler_package"]).resolve() == (
        root / "src" / "fpgai_research" / "toolchains" /
        "vitis_hls" / "package.py").resolve()
    assert result["original_vitis_calls"] == 0
