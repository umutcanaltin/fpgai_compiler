from __future__ import annotations

import json
from types import SimpleNamespace

from fpgai.backends.hls.codegen import emit_hls_stub


class _Section(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def _compile_plan_with_dense_tiling():
    return SimpleNamespace(
        layer_plans=[
            SimpleNamespace(
                node_name="dense0",
                op_type="Dense",
                architecture=SimpleNamespace(
                    precision=_Section(
                        activation_bits=16,
                        weight_bits=8,
                        accumulator_bits=32,
                    ),
                    parallelism=_Section(
                        pe=2,
                        simd=2,
                    ),
                    pipeline=_Section(
                        ii=1,
                    ),
                    tiling=_Section(
                        sizes={
                            "in": 8,
                            "out": 4,
                        }
                    ),
                ),
                input_features=16,
                output_features=8,
            )
        ]
    )


def test_emit_hls_stub_writes_tiling_reports_and_meta(tmp_path) -> None:
    project = emit_hls_stub(
        graph=SimpleNamespace(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "clk_mhz": 200,
            "run_csim": False,
            "run_csynth": False,
            "memory_words_per_cycle": 4,
            "tile_overhead_cycles": 1,
        },
        compile_plan=_compile_plan_with_dense_tiling(),
    )

    reports_dir = project.hls_dir / "reports"

    assert (reports_dir / "tiling_analysis.json").exists()
    assert (reports_dir / "tiling_resource_estimate.json").exists()
    assert (reports_dir / "tiling_performance_estimate.json").exists()

    meta = json.loads(
        (project.hls_dir / "codegen_meta.json").read_text(
            encoding="utf-8"
        )
    )

    assert meta["compile_plan_present"] is True
    assert meta["tiling_analysis"]["path"] == "reports/tiling_analysis.json"
    assert meta["tiling_analysis"]["tiled_layer_count"] == 1
    assert meta["tiling_resource_estimate"]["path"] == "reports/tiling_resource_estimate.json"
    assert meta["tiling_resource_estimate"]["tile_buffer_bits"] == 512
    assert meta["tiling_performance_estimate"]["path"] == "reports/tiling_performance_estimate.json"
    assert meta["tiling_performance_estimate"]["estimated_macs"] == 128


def test_emit_hls_stub_can_write_optional_tiling_sweep(tmp_path) -> None:
    project = emit_hls_stub(
        graph=SimpleNamespace(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "clk_mhz": 200,
            "run_csim": False,
            "run_csynth": False,
            "tiling_candidates_by_layer": {
                "dense0": [
                    {
                        "in": 4,
                        "out": 2,
                    },
                    {
                        "in": 8,
                        "out": 4,
                    },
                ]
            },
            "tiling_sweep_top_k": 1,
        },
        compile_plan=_compile_plan_with_dense_tiling(),
    )

    assert (project.hls_dir / "reports" / "tiling_sweep.json").exists()

    meta = json.loads(
        (project.hls_dir / "codegen_meta.json").read_text(
            encoding="utf-8"
        )
    )

    assert meta["tiling_sweep"]["path"] == "reports/tiling_sweep.json"
    assert meta["tiling_sweep"]["layer_count"] == 1
    assert meta["tiling_sweep"]["best_layer_count"] == 1


def test_emit_hls_stub_without_compile_plan_keeps_tiling_reports_absent(tmp_path) -> None:
    project = emit_hls_stub(
        graph=SimpleNamespace(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "run_csim": False,
            "run_csynth": False,
        },
        compile_plan=None,
    )

    meta = json.loads(
        (project.hls_dir / "codegen_meta.json").read_text(
            encoding="utf-8"
        )
    )

    assert meta["compile_plan_present"] is False
    assert "tiling_analysis" not in meta
    assert not (project.hls_dir / "reports").exists()
