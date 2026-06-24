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
    assert '"vivado_bridge"' in source
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
    assert by_name["generate_hls"]["status"] == "skipped"
    assert by_name["run_hls"]["status"] == "skipped"
    assert by_name["training_artifacts"]["status"] == "skipped"
    assert by_name["vivado_bridge"]["status"] == "not_requested"
    assert by_name["runtime_package"]["status"] == "not_implemented"

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
