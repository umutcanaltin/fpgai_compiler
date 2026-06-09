from __future__ import annotations

import argparse
import sys

from fpgai.benchmark.pipeline import run_compile_correctness_benchmark
from fpgai.config.loader import ConfigError, load_config, print_summary
from fpgai.engine.compiler import Compiler


def _benchmark_enabled(cfg) -> bool:
    raw = getattr(cfg, "raw", {}) or {}
    bench = raw.get("benchmark", {}) or {}
    return bool(bench.get("enabled", False))


def _should_run_inference_benchmark(cfg) -> bool:
    return cfg.pipeline.mode == "inference" and _benchmark_enabled(cfg)


def run_from_config(config_path: str) -> int:
    try:
        cfg = load_config(config_path)
        print_summary(cfg)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if _should_run_inference_benchmark(cfg):
            print("\n=============== FPGAI Benchmark ===============")
            bench = run_compile_correctness_benchmark(config_path=config_path)

            print(f"Benchmark passed : {bench.passed}")
            print(f"Metrics JSON     : {bench.metrics_json}")
            print(f"Summary TXT      : {bench.summary_txt}")

            if bench.quant_metrics_json is not None:
                print(f"Quant JSON       : {bench.quant_metrics_json}")
            if bench.quant_summary_txt is not None:
                print(f"Quant Summary    : {bench.quant_summary_txt}")
            if getattr(bench, "precision_sweep_results_json", None) is not None:
                print(f"Sweep JSON       : {bench.precision_sweep_results_json}")
            if getattr(bench, "precision_sweep_summary_txt", None) is not None:
                print(f"Sweep Summary    : {bench.precision_sweep_summary_txt}")

            print("==============================================")

            if bench.passed:
                print("[OK] Benchmark completed successfully.")
            else:
                print("[WARN] Benchmark completed but did not pass thresholds.")

            return 0

        compiler = Compiler(cfg)
        result = compiler.compile()
        print(result.summary())
        print(f"[OK] Wrote artifacts to: {result.out_dir}")
        return 0

    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("FPGAI compiler")
    subparsers = parser.add_subparsers(dest="command")

    compile_parser = subparsers.add_parser("compile", help="Compile using a YAML config")
    compile_parser.add_argument("--config", required=True, help="Path to fpgai.yml")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run compile/correctness benchmark using a YAML config",
    )
    benchmark_parser.add_argument("--config", required=True, help="Path to fpgai.yml")

    parser.add_argument(
        "--config",
        help="Backward-compatible config argument. Equivalent to: fpgai compile --config ...",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        raise SystemExit(run_from_config(args.config))

    if args.command in {"compile", "benchmark"}:
        raise SystemExit(run_from_config(args.config))

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()