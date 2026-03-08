from __future__ import annotations

from pathlib import Path

from fpgai.benchmark.pipeline import run_compile_correctness_benchmark


def test_compile_correctness_mnist():
    cfg = Path("fpgai.yml")
    result = run_compile_correctness_benchmark(
        config_path=cfg,
        atol=5e-2,
        rtol=5e-2,
        seed=0,
    )
    assert result.passed, f"Benchmark failed. See {result.summary_txt}"