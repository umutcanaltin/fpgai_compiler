from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil

import numpy as np

from fpgai.benchmark.reference import run_onnx_reference
from fpgai.benchmark.compare import compare_outputs, write_metrics
from fpgai.benchmark.reference_intermediate import run_onnx_intermediate_reference
from fpgai.benchmark.intermediate_compare import compare_intermediate_layers
from fpgai.engine.compiler import Compiler


@dataclass(frozen=True)
class BenchmarkResult:
    build_dir: Path
    bench_dir: Path
    passed: bool
    metrics_json: Path
    summary_txt: Path
    reference_output_npy: Path
    hls_output_npy: Path
    quant_metrics_json: Path | None = None
    quant_summary_txt: Path | None = None


def _cfg_get(raw: dict, path: str, default=None):
    cur = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _write_input_bin_from_npy(input_npy: Path, build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(input_npy).astype(np.float32).reshape(-1)
    out = build_dir / "input.bin"
    x.tofile(out)
    return out


def _load_hls_output_bin(output_bin: Path, reference_shape: tuple[int, ...]) -> np.ndarray:
    y = np.fromfile(output_bin, dtype=np.float32)
    return y.reshape(reference_shape)


def _find_output_bin(out_dir: Path) -> Path:
    candidates = [
        out_dir / "hls" / "fpgai_hls_proj" / "sol1" / "csim" / "build" / "output.bin",
        out_dir / "hls" / "output.bin",
        out_dir / "output.bin",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find output.bin from HLS simulation")


def run_compile_correctness_benchmark(
    *,
    config_path: str | Path,
) -> BenchmarkResult:
    compiler = Compiler.from_yaml(str(config_path))
    raw = compiler.cfg.raw

    model_path = Path(compiler.cfg.model.path)
    out_dir = Path(raw["project"]["out_dir"]).resolve()
    bench_dir = out_dir / "bench"

    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)

    seed = int(_cfg_get(raw, "benchmark.seed", 0))
    atol = float(_cfg_get(raw, "benchmark.compare.atol", 1e-2))
    rtol = float(_cfg_get(raw, "benchmark.compare.rtol", 1e-2))
    max_abs_error_limit = _cfg_get(raw, "benchmark.compare.max_abs_error", None)
    mean_abs_error_limit = _cfg_get(raw, "benchmark.compare.mean_abs_error", None)
    rmse_limit = _cfg_get(raw, "benchmark.compare.rmse", None)
    require_argmax_match = bool(_cfg_get(raw, "benchmark.compare.require_argmax_match", False))
    min_cosine_similarity = _cfg_get(raw, "benchmark.compare.min_cosine_similarity", None)
    fail_on_mismatch = bool(_cfg_get(raw, "benchmark.fail_on_mismatch", False))

    intermediate_enabled = bool(_cfg_get(raw, "benchmark.intermediate.enabled", False))
    fail_on_layer_mismatch = bool(_cfg_get(raw, "benchmark.intermediate.fail_on_layer_mismatch", False))
    stop_on_first_bad_layer = bool(_cfg_get(raw, "benchmark.intermediate.stop_on_first_bad_layer", False))

    ref = run_onnx_reference(
        model_path=model_path,
        out_dir=bench_dir,
        seed=seed,
    )

    _write_input_bin_from_npy(ref.input_npy, out_dir)

    old_clean = raw.get("project", {}).get("clean", True)
    raw["project"]["clean"] = False
    try:
        compile_result = compiler.compile()
    finally:
        raw["project"]["clean"] = old_clean

    if not compile_result.hls_ran:
        raise RuntimeError("Benchmark requested but HLS was not run.")
    if compile_result.hls_ok is not True:
        raise RuntimeError(
            f"Benchmark requested but HLS run failed. "
            f"See {compile_result.hls_stdout_log} and {compile_result.hls_stderr_log}"
        )

    output_bin = _find_output_bin(out_dir)
    ref_y = np.load(ref.output_npy).astype(np.float32)
    hls_y = _load_hls_output_bin(output_bin, ref_y.shape)

    hls_output_npy = bench_dir / "hls_output.npy"
    np.save(hls_output_npy, hls_y)

    metrics = compare_outputs(
        ref_y,
        hls_y,
        atol=atol,
        rtol=rtol,
        max_abs_error_limit=max_abs_error_limit,
        mean_abs_error_limit=mean_abs_error_limit,
        rmse_limit=rmse_limit,
        require_argmax_match=require_argmax_match,
        min_cosine_similarity=min_cosine_similarity,
    )
    write_metrics(metrics, bench_dir)

    intermediate_summary = None
    if intermediate_enabled:
        ref_input = np.load(ref.input_npy).astype(np.float32)
        ref_layers_dir = bench_dir / "reference_layers"

        run_onnx_intermediate_reference(
            model_path=model_path,
            input_array=ref_input,
            graph=compile_result.graph,
            out_dir=ref_layers_dir,
        )

        csim_build_dir = out_dir / "hls" / "fpgai_hls_proj" / "sol1" / "csim" / "build"
        intermediate_out_dir = bench_dir / "intermediate"

        intermediate_summary = compare_intermediate_layers(
            graph=compile_result.graph,
            reference_dir=ref_layers_dir,
            csim_build_dir=csim_build_dir,
            out_dir=intermediate_out_dir,
            atol=atol,
            rtol=rtol,
            max_abs_error_limit=max_abs_error_limit,
            mean_abs_error_limit=mean_abs_error_limit,
            rmse_limit=rmse_limit,
            require_argmax_match=False,
            min_cosine_similarity=min_cosine_similarity,
        )

        if fail_on_layer_mismatch and intermediate_summary["first_bad_layer"] is not None:
            raise RuntimeError(
                f"Intermediate benchmark failed first at layer {intermediate_summary['first_bad_layer']}. "
                f"See {intermediate_out_dir / 'intermediate_summary.txt'}"
            )

        if stop_on_first_bad_layer and intermediate_summary["first_bad_layer"] is not None:
            raise RuntimeError(
                f"Stopped on first bad layer: {intermediate_summary['first_bad_layer']}. "
                f"See {intermediate_out_dir / 'intermediate_summary.txt'}"
            )

    manifest_patch = {
        "benchmark": {
            "passed": metrics.passed,
            "metrics_json": str((bench_dir / "metrics.json").resolve()),
            "summary_txt": str((bench_dir / "summary.txt").resolve()),
            "reference_output_npy": str(ref.output_npy.resolve()),
            "hls_output_npy": str((bench_dir / "hls_output.npy").resolve()),
            "hls_stdout_log": compile_result.hls_stdout_log,
            "hls_stderr_log": compile_result.hls_stderr_log,
            "hls_csynth_report": compile_result.hls_csynth_report,
            "intermediate_enabled": intermediate_enabled,
            "intermediate_first_bad_layer": (
                None if intermediate_summary is None else intermediate_summary["first_bad_layer"]
            ),
            "quant_metrics_json": (
                None if compile_result.quant_metrics_json is None else str(compile_result.quant_metrics_json.resolve())
            ),
            "quant_summary_txt": (
                None if compile_result.quant_summary_txt is None else str(compile_result.quant_summary_txt.resolve())
            ),
        }
    }
    (bench_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest_patch, indent=2),
        encoding="utf-8",
    )

    if fail_on_mismatch and not metrics.passed:
        raise RuntimeError(
            f"Benchmark failed. "
            f"See {bench_dir / 'summary.txt'} and {bench_dir / 'metrics.json'}"
        )

    return BenchmarkResult(
        build_dir=out_dir,
        bench_dir=bench_dir,
        passed=metrics.passed,
        metrics_json=bench_dir / "metrics.json",
        summary_txt=bench_dir / "summary.txt",
        reference_output_npy=ref.output_npy,
        hls_output_npy=hls_output_npy,
        quant_metrics_json=compile_result.quant_metrics_json,
        quant_summary_txt=compile_result.quant_summary_txt,
    )