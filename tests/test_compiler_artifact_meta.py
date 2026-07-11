from __future__ import annotations

import json
from types import SimpleNamespace

from fpgai.compiler.compiler import Compiler
from pathlib import Path


def _cfg(out_dir, *, training_enabled=False):
    return SimpleNamespace(
        numerics=SimpleNamespace(
            kind="fixed",
            activation="ap_fixed<16,6>",
        ),
        pipeline=SimpleNamespace(
            mode="inference",
        ),
        raw={
            "project": {
                "out_dir": str(out_dir),
            },
            "training": {
                "enabled": training_enabled,
                "optimizer": {
                    "learning_rate": 0.025,
                },
            },
        },
    )


def test_compile_artifact_loads_codegen_tiling_meta(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "generated"
    hls_dir = out_dir / "hls"
    hls_dir.mkdir(parents=True)

    codegen_meta = {
        "pipeline_mode": "inference",
        "compile_plan_present": True,
        "tiling_analysis": {
            "path": "reports/tiling_analysis.json",
            "tiled_layer_count": 1,
        },
        "tiling_resource_estimate": {
            "path": "reports/tiling_resource_estimate.json",
            "tile_buffer_bits": 512,
        },
        "tiling_performance_estimate": {
            "path": "reports/tiling_performance_estimate.json",
            "estimated_macs": 128,
        },
    }
    (hls_dir / "codegen_meta.json").write_text(
        json.dumps(codegen_meta),
        encoding="utf-8",
    )

    calls = {}

    def fake_engine(**kwargs):
        calls.update(kwargs)
        return None

    monkeypatch.setattr(
        "fpgai.compiler.compiler.fpgai_engine",
        fake_engine,
    )

    artifact = Compiler.from_cfg(
        _cfg(out_dir)
    ).compile(
        input_data=[0.0],
        first_layer_shape=(1,),
        output_shape=(1,),
        onnx_path="model.onnx",
        verbose=True,
    )

    assert artifact.success is True
    assert artifact.out_dir == str(out_dir)
    assert calls["precision"] == "ap_fixed<16,6>"
    assert calls["learning_rate"] == 0.1
    assert calls["verbose"] is True

    assert artifact.meta["codegen_meta_present"] is True
    assert artifact.meta["codegen_meta"] == codegen_meta
    assert artifact.meta["tiling_analysis"]["tiled_layer_count"] == 1
    assert artifact.meta["tiling_resource_estimate"]["tile_buffer_bits"] == 512
    assert artifact.meta["tiling_performance_estimate"]["estimated_macs"] == 128


def test_compile_artifact_meta_reports_absent_codegen_meta(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "generated"

    monkeypatch.setattr(
        "fpgai.compiler.compiler.fpgai_engine",
        lambda **kwargs: None,
    )

    artifact = Compiler.from_cfg(
        _cfg(out_dir)
    ).compile(
        input_data=[0.0],
        first_layer_shape=(1,),
        output_shape=(1,),
        onnx_path="model.onnx",
    )

    assert artifact.meta["codegen_meta_present"] is False
    assert artifact.meta["codegen_meta_path"].endswith(
        "generated/hls/codegen_meta.json"
    )
    assert "tiling_analysis" not in artifact.meta


def test_training_learning_rate_is_read_from_dict_optimizer(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "generated"
    calls = {}

    def fake_engine(**kwargs):
        calls.update(kwargs)
        return None

    monkeypatch.setattr(
        "fpgai.compiler.compiler.fpgai_engine",
        fake_engine,
    )

    Compiler.from_cfg(
        _cfg(
            out_dir,
            training_enabled=True,
        )
    ).compile(
        input_data=[0.0],
        first_layer_shape=(1,),
        output_shape=(1,),
        onnx_path="model.onnx",
    )

    assert calls["learning_rate"] == 0.025

def test_compiler_manifest_records_pipeline_stages_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "_build_pipeline_stages" in source
    assert '"pipeline_stages": self._build_pipeline_stages(**kwargs)' in source
    assert '"vivado_project"' in source
    assert '"bitstream"' in source
    assert '"runtime_package"' in source

def test_compiler_pipeline_stage_helper_returns_expected_names(tmp_path):
    from types import SimpleNamespace

    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)
    compiler.cfg = SimpleNamespace(
        raw={
            "backends": {
                "hls": {"enabled": False},
                "host_cpp": {"enabled": False},
            }
        }
    )

    compile_plan = SimpleNamespace(
        layer_plans=[],
        architecture_signature="test-signature",
    )
    graph = SimpleNamespace(
        ops=[],
        params={},
    )
    memory_plan = SimpleNamespace(placements=[])
    communication_plan = SimpleNamespace(edges=[])

    stages = compiler._build_pipeline_stages(
        graph=graph,
        compile_plan=compile_plan,
        memory_plan=memory_plan,
        communication_plan=communication_plan,
        descriptors=[],
        hls_run=None,
        training_plan=None,
        quant_result=None,
        sweep_result=None,
        design_result=None,
        estimate_vs_hls_result=None,
        hls_module_breakdown_result=None,
    )

    by_name = {stage["name"]: stage for stage in stages}

    assert by_name["load_config"]["status"] == "done"
    assert by_name["import_model"]["status"] == "done"
    assert by_name["analyze_model"]["status"] == "done"
    assert by_name["plan_architecture"]["status"] == "done"
    assert by_name["generate_host_cpp"]["status"] == "skipped"
    assert by_name["generate_hls_project"]["status"] == "skipped"
    assert by_name["run_hls"]["status"] == "skipped"
    assert by_name["training_artifacts"]["status"] == "skipped"
    assert by_name["vivado_project"]["status"] == "not_requested"
    assert by_name["bitstream"]["status"] == "not_requested"
    assert by_name["runtime_package"]["status"] == "skipped"
    assert by_name["runtime_package"]["detail"] == "Runtime package was not emitted."

def test_compiler_emits_prediction_artifacts_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "_emit_prediction_artifacts" in source
    assert "write_model_inspection_report" in source
    assert "estimate_resources_from_descriptors" in source
    assert "estimate_performance" in source
    assert "out_dir / \"reports\"" in source
    assert "prediction_artifacts=prediction_artifacts" in source
    assert "\"prediction_artifacts\": kwargs.get(\"prediction_artifacts\")" in source


def test_design_space_manifest_payload_includes_recommendations(tmp_path):
    import json
    from types import SimpleNamespace

    from fpgai.engine.compiler import Compiler

    design_dir = tmp_path / "design_space"
    design_dir.mkdir()

    results_json = design_dir / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "format": "fpgai.design_space.v2",
                "analytical_models": {
                    "resources": "operator_structural_v2",
                    "performance": "operator_execution_schedule_v2",
                },
                "recommendation_policy": {
                    "require_prediction_match": True,
                    "minimum_cosine": 0.999,
                },
                "recommended_smallest_valid": {"name": "fx8"},
                "recommended_balanced": {"name": "fx12"},
                "recommended_best_accuracy": {"name": "fx16"},
                "recommendation_scope": "configured_candidates_only",
                "search_enabled": False,
                "recommendation_kind": "estimate_based_recommendation",
                "dse_truth": {"configured_candidates_only": True},
            }
        ),
        encoding="utf-8",
    )

    result = SimpleNamespace(
        out_dir=design_dir,
        results_json=results_json,
        summary_txt=design_dir / "summary.txt",
        results_csv=design_dir / "results.csv",
    )

    compiler = Compiler.__new__(Compiler)
    payload = compiler._design_space_manifest_payload(result)

    assert payload["prediction_status"] == "estimate"
    assert payload["format"] == "fpgai.design_space.v2"
    assert payload["layer_breakdown_csv"].endswith("layer_breakdown.csv")
    assert payload["analytical_models"]["resources"] == "operator_structural_v2"
    assert payload["recommended_smallest_valid"]["name"] == "fx8"
    assert payload["recommended_balanced"]["name"] == "fx12"
    assert payload["recommended_best_accuracy"]["name"] == "fx16"
    assert payload["recommendation_scope"] == "configured_candidates_only"
    assert payload["search_enabled"] is False
    assert payload["recommendation_kind"] == "estimate_based_recommendation"
    assert payload["dse_truth"]["configured_candidates_only"] is True

def test_hls_artifacts_manifest_payload_groups_hls_outputs(tmp_path):
    from types import SimpleNamespace

    from fpgai.engine.compiler import Compiler

    hls_run = SimpleNamespace(
        ok=True,
        returncode=0,
        stdout_log=tmp_path / "hls" / "logs" / "stdout.log",
        stderr_log=tmp_path / "hls" / "logs" / "stderr.log",
        csynth_report=tmp_path / "hls" / "proj" / "sol1" / "syn" / "report" / "csynth.rpt",
    )

    compiler = Compiler.__new__(Compiler)
    payload = compiler._hls_artifacts_manifest_payload(
        out_dir=tmp_path,
        hls_run=hls_run,
        hls_schedule_summary={"path": "hls_schedule_summary.json"},
        hls_artifact_metadata={"path": "hls_artifact_metadata.json"},
        hls_ii_comparison={"path": "hls_ii_comparison.json"},
    )

    assert payload["hls_ran"] is True
    assert payload["hls_ok"] is True
    assert payload["hls_returncode"] == 0
    assert payload["hls_project_dir"].endswith("hls")
    assert payload["stdout_log"].endswith("stdout.log")
    assert payload["stderr_log"].endswith("stderr.log")
    assert payload["csynth_report"].endswith("csynth.rpt")
    assert payload["schedule_summary"]["path"] == "hls_schedule_summary.json"
    assert payload["artifact_metadata"]["path"] == "hls_artifact_metadata.json"
    assert payload["ii_comparison"]["path"] == "hls_ii_comparison.json"


def test_compiler_manifest_records_training_reference_and_estimate_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "\"training_reference\": None if kwargs[\"training_reference_result\"] is None else {" in source
    assert "\"loss_before\": kwargs[\"training_reference_result\"].loss_before" in source
    assert "\"loss_after\": kwargs[\"training_reference_result\"].loss_after" in source
    assert "\"grads_ref_bin\": str(kwargs[\"training_reference_result\"].grads_flat_path)" in source
    assert "\"weights_before_ref_bin\": str(kwargs[\"training_reference_result\"].weights_before_flat_path)" in source
    assert "\"weights_after_ref_bin\": str(kwargs[\"training_reference_result\"].weights_after_flat_path)" in source
    assert "\"summary_json\": str(kwargs[\"training_reference_result\"].summary_json)" in source
    assert "\"summary_txt\": str(kwargs[\"training_reference_result\"].summary_txt)" in source

    assert "\"training_estimate\": None if kwargs[\"training_estimate_result\"] is None else {" in source
    assert "\"total_param_bytes\": kwargs[\"training_estimate_result\"].total_param_bytes" in source
    assert "\"total_activation_cache_bytes\": kwargs[\"training_estimate_result\"].total_activation_cache_bytes" in source
    assert "\"total_gradient_bytes\": kwargs[\"training_estimate_result\"].total_gradient_bytes" in source
    assert "\"total_optimizer_state_bytes\": kwargs[\"training_estimate_result\"].total_optimizer_state_bytes" in source




def test_params_cpp_does_not_emit_file_scope_bind_storage_pragmas():
    from pathlib import Path

    source = Path("fpgai/backends/hls/emit/params_cpp.py").read_text(
        encoding="utf-8"
    )

    assert "#pragma HLS BIND_STORAGE" not in source
    assert "file-scope BIND_STORAGE disabled" in source


def test_compiler_emits_board_fit_artifacts_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "emit_board_fit_report" in source
    assert "reports_dir" in source
    assert "board_fit_json" in source
    assert "board_fit_markdown" in source
    assert '"targets.platform.board"' in source
    assert '"targets.platform.clocks.0.target_mhz"' in source


def test_hardware_feasibility_has_user_guiding_board_fit_reporter() -> None:
    source = Path("fpgai/reporting/hardware_feasibility.py").read_text(encoding="utf-8")

    assert "def emit_board_fit_report" in source
    assert "def board_fit_markdown" in source
    assert "suggested_yaml_actions" in source
    assert "Truth boundary" in source
    assert "prediction_based" in source


def test_compiler_emits_hardware_knob_contract_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "_emit_hardware_knob_contract_reports" in source
    assert "hardware_knob_contract.json" in source
    assert "hardware_knob_contract.md" in source
    assert "manual_yaml_override" in source
    assert "optimization.parallel.pe" in source
    assert "optimization.parallel.simd" in source
    assert "optimization.pipeline.ii" in source
    assert "optimization.tiling.dense" in source
    assert "optimization.tiling.conv" in source
    assert "targets.platform.fit_policy" in source
    assert '"hardware_knob_contract": hardware_knob_contract' in source


def test_hardware_knob_contract_is_layer_aware_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "first_layer_of_type_value" in source
    assert "has_layer_type" in source
    assert "dense_tiling_effective" in source
    assert "conv_tiling_effective" in source
    assert '"not_applicable"' in source


def test_hardware_knob_contract_labels_board_aware_policy_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "board_aware_policy" in source
    assert "policy_resource_awareness" in source
    assert "board_aware_policy_scaling" in source


def test_hardware_knob_contract_path_helpers_support_list_indexes() -> None:
    import inspect

    import fpgai.engine.compiler as compiler_module

    compiler_cls = None
    for _name, obj in inspect.getmembers(compiler_module, inspect.isclass):
        if hasattr(obj, "_raw_has_path") and hasattr(obj, "_raw_get_path"):
            compiler_cls = obj
            break

    assert compiler_cls is not None

    compiler = object.__new__(compiler_cls)
    raw = {
        "targets": {
            "platform": {
                "clocks": [
                    {
                        "name": "ap_clk",
                        "target_mhz": 200,
                    }
                ]
            }
        }
    }

    assert compiler._raw_has_path(raw, "targets.platform.clocks.0.target_mhz")
    assert compiler._raw_get_path(raw, "targets.platform.clocks.0.target_mhz") == 200
    assert not compiler._raw_has_path(raw, "targets.platform.clocks.1.target_mhz")


def test_hardware_knob_contract_status_treats_numeric_equivalence_as_applied() -> None:
    import inspect

    import fpgai.engine.compiler as compiler_module

    compiler_cls = None
    for _name, obj in inspect.getmembers(compiler_module, inspect.isclass):
        if hasattr(obj, "_contract_status"):
            compiler_cls = obj
            break

    assert compiler_cls is not None
    assert compiler_cls._contract_status(200, 200.0, manual=True) == "applied"
    assert compiler_cls._contract_status(200, 100.0, manual=True) == "changed_or_clamped"
    assert compiler_cls._contract_status(None, 100.0, manual=False) == "applied"
    assert compiler_cls._contract_status(None, None, manual=False) == "unknown"


def test_compiler_has_fit_policy_gate_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "def _resolved_fit_policy" in source
    assert "def _fit_policy_gate" in source
    assert '"fit_policy_gate": fit_policy_gate' in source
    assert "targets.platform.fit_policy" in source
    assert "hardware.fit_policy" in source
    assert "build.fit_policy" in source
    assert "block_over_limit" in source
    assert "policy_source" in source
    assert "requested_policy" in source
    assert "blocked_stages" in source
    assert "vivado_impl" in source
    assert "bitstream" in source
    assert "deployable_runtime_overlay" in source
    assert "fit_policy is enforced through fit_policy_gate" in source


def test_fit_policy_gate_modes_behavior() -> None:
    from types import SimpleNamespace

    from fpgai.engine.compiler import Compiler

    prediction_artifacts = {
        "board_fit": {
            "status": "over_limit",
            "limiting_dimension": "dsp",
            "vivado_allowed": False,
        }
    }

    expected = {
        "report_only": (False, False, "info"),
        "warn": (False, True, "warning"),
        "enforce": (True, False, "error"),
        "block_over_limit": (True, False, "error"),
    }

    for policy, (blocked, warning, severity) in expected.items():
        compiler = object.__new__(Compiler)
        compiler.cfg = SimpleNamespace(
            raw={
                "targets": {
                    "platform": {
                        "fit_policy": policy,
                    }
                }
            }
        )

        gate = compiler._fit_policy_gate(prediction_artifacts)

        assert gate["policy"] == ("enforce" if policy == "block_over_limit" else policy)
        assert gate["policy_source"] == "targets.platform.fit_policy"
        assert gate["requested_policy"] == policy
        assert gate["board_fit_status"] == "over_limit"
        assert gate["board_fit_limiting_dimension"] == "dsp"
        assert gate["vivado_allowed_by_board_fit"] is False
        assert gate["over_limit"] is True
        assert gate["blocked"] is blocked
        assert gate["warning"] is warning
        assert gate["severity"] == severity

    compiler = object.__new__(Compiler)
    compiler.cfg = SimpleNamespace(
        raw={
            "targets": {
                "platform": {
                    "fit_policy": "invalid_policy_name",
                }
            }
        }
    )
    gate = compiler._fit_policy_gate(prediction_artifacts)

    assert gate["policy"] == "report_only"
    assert gate["blocked"] is False
    assert gate["warning"] is False


def test_fit_policy_resolver_supports_build_path_and_priority() -> None:
    from types import SimpleNamespace

    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)
    compiler.cfg = SimpleNamespace(raw={"build": {"fit_policy": "block_over_limit"}})
    assert compiler._resolved_fit_policy() == ("enforce", "build.fit_policy", "block_over_limit")

    compiler.cfg = SimpleNamespace(raw={"hardware": {"fit_policy": "warn"}, "build": {"fit_policy": "block_over_limit"}})
    assert compiler._resolved_fit_policy() == ("warn", "hardware.fit_policy", "warn")

    compiler.cfg = SimpleNamespace(
        raw={
            "targets": {"platform": {"fit_policy": "enforce"}},
            "hardware": {"fit_policy": "warn"},
            "build": {"fit_policy": "report_only"},
        }
    )
    assert compiler._resolved_fit_policy() == ("enforce", "targets.platform.fit_policy", "enforce")

    compiler.cfg = SimpleNamespace(raw={"build": {"fit_policy": "invalid_policy_name"}})
    assert compiler._resolved_fit_policy() == ("report_only", "invalid_fallback", "invalid_policy_name")


def test_fit_policy_gate_uses_build_fit_policy_alias() -> None:
    from types import SimpleNamespace

    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)
    compiler.cfg = SimpleNamespace(raw={"build": {"fit_policy": "block_over_limit"}})
    gate = compiler._fit_policy_gate(
        {
            "board_fit": {
                "status": "over_limit",
                "limiting_dimension": "lut",
                "vivado_allowed": False,
            }
        }
    )

    assert gate["policy"] == "enforce"
    assert gate["policy_source"] == "build.fit_policy"
    assert gate["requested_policy"] == "block_over_limit"
    assert gate["blocked"] is True
    assert "vivado_impl" in gate["blocked_stages"]
    assert "bitstream" in gate["blocked_stages"]


def test_config_loader_validates_fit_policy_enum_in_source() -> None:
    source = Path("fpgai/config/loader.py").read_text(encoding="utf-8")

    assert "def _validate_fit_policy" in source
    assert '"targets.platform.fit_policy"' in source
    assert '"hardware.fit_policy"' in source
    assert '"build.fit_policy"' in source
    assert "block_over_limit" in source
    assert '{"report_only", "warn", "enforce"}' in source
    assert "Invalid fit_policy" in source
    assert "_validate_fit_policy(" in source


def test_compiler_uses_compile_plan_clock_for_outputs_in_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert 'getattr(compile_plan, "clock_mhz"' in source
    assert 'target_clock_mhz = getattr(compile_plan, "clock_mhz"' in source
    assert 'clk_mhz = float(getattr(compile_plan, "clock_mhz"' in source
