from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil

import numpy as np

from fpgai.benchmark.reference import run_onnx_reference
from fpgai.benchmark.compare import compare_outputs, write_metrics
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

    # Start fresh for benchmark artifacts only if build dir already exists.
    if bench_dir.exists():
        shutil.rmtree(bench_dir)

    # Create reference first.
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

    # 1) ONNX reference
    ref = run_onnx_reference(
        model_path=model_path,
        out_dir=bench_dir,
        seed=seed,
    )

    # 2) Write exact same input as input.bin BEFORE compile/HLS run
    _write_input_bin_from_npy(ref.input_npy, out_dir)

    # 3) Prevent compile() from deleting benchmark artifacts / input.bin
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
            f"Benchmark requested but HLS run failed. See {compile_result.hls_stdout_log} and {compile_result.hls_stderr_log}"
        )

    # 4) Load HLS output produced from the same input
    output_bin = _find_output_bin(out_dir)

    ref_y = np.load(ref.output_npy).astype(np.float32)
    hls_y = _load_hls_output_bin(output_bin, ref_y.shape)

    hls_output_npy = bench_dir / "hls_output.npy"
    np.save(hls_output_npy, hls_y)

    # 5) Compare
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

    manifest_patch = {
        "benchmark": {
            "passed": metrics.passed,
            "metrics_json": str((bench_dir / "metrics.json").resolve()),
            "summary_txt": str((bench_dir / "summary.txt").resolve()),
            "reference_output_npy": str(ref.output_npy.resolve()),
            "hls_output_npy": str(hls_output_npy.resolve()),
            "hls_stdout_log": compile_result.hls_stdout_log,
            "hls_stderr_log": compile_result.hls_stderr_log,
            "hls_csynth_report": compile_result.hls_csynth_report,
        }
    }
    (bench_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest_patch, indent=2),
        encoding="utf-8",
    )

    if fail_on_mismatch and not metrics.passed:
        raise RuntimeError(
            f"Benchmark failed. See {bench_dir / 'summary.txt'} and {bench_dir / 'metrics.json'}"
        )

    return BenchmarkResult(
        build_dir=out_dir,
        bench_dir=bench_dir,
        passed=metrics.passed,
        metrics_json=bench_dir / "metrics.json",
        summary_txt=bench_dir / "summary.txt",
        reference_output_npy=ref.output_npy,
        hls_output_npy=hls_output_npy,
    )