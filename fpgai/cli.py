from __future__ import annotations

import argparse
import sys

from fpgai.benchmark.pipeline import run_compile_correctness_benchmark
from fpgai.config.loader import ConfigError, load_config, print_summary
from fpgai.engine.compiler import Compiler


def _benchmark_enabled(cfg) -> bool:
    raw = getattr(cfg, "raw", {}) or {}
    benchmark = raw.get("benchmark", {}) or {}
    return bool(benchmark.get("enabled", False))


def _should_run_inference_benchmark(cfg) -> bool:
    return cfg.pipeline.mode == "inference" and _benchmark_enabled(cfg)


def run_from_config(config_path: str, *, action: str = "auto") -> int:
    try:
        cfg = load_config(config_path)
        print_summary(cfg)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if action not in {"auto", "compile", "benchmark"}:
        print(f"[ERROR] Unsupported CLI action: {action}", file=sys.stderr)
        return 2

    try:
        should_benchmark = action == "benchmark" or (
            action == "auto" and _should_run_inference_benchmark(cfg)
        )

        if should_benchmark:
            if cfg.pipeline.mode != "inference":
                print(
                    "[ERROR] Correctness benchmarking currently supports "
                    "pipeline.mode=inference only.",
                    file=sys.stderr,
                )
                return 2

            print("\n=============== FPGAI Benchmark ===============")
            result = run_compile_correctness_benchmark(
                config_path=config_path,
            )

            print(f"Benchmark passed : {result.passed}")
            print(f"Metrics JSON     : {result.metrics_json}")
            print(f"Summary TXT      : {result.summary_txt}")

            if result.quant_metrics_json is not None:
                print(f"Quant JSON       : {result.quant_metrics_json}")

            if result.quant_summary_txt is not None:
                print(f"Quant Summary    : {result.quant_summary_txt}")

            if result.precision_sweep_results_json is not None:
                print(
                    "Sweep JSON       : "
                    f"{result.precision_sweep_results_json}"
                )

            if result.precision_sweep_summary_txt is not None:
                print(
                    "Sweep Summary    : "
                    f"{result.precision_sweep_summary_txt}"
                )

            print("==============================================")

            if result.passed:
                print("[OK] Benchmark completed successfully.")
                return 0

            print("[WARN] Benchmark completed but did not pass thresholds.")
            return 1

        compiler = Compiler(cfg)
        result = compiler.compile()

        print(result.summary())
        print(f"[OK] Wrote artifacts to: {result.out_dir}")
        return 0

    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpgai",
        description="Compile ONNX models into FPGAI HLS projects.",
    )

    subparsers = parser.add_subparsers(dest="command")

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile a model without automatically running the benchmark.",
    )
    compile_parser.add_argument(
        "--config",
        required=True,
        help="Path to the FPGAI YAML configuration.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the inference correctness benchmark.",
    )
    benchmark_parser.add_argument(
        "--config",
        required=True,
        help="Path to the FPGAI YAML configuration.",
    )

    parser.add_argument(
        "--config",
        help=(
            "Backward-compatible configuration argument. "
            "This follows benchmark.enabled in the YAML."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        raise SystemExit(
            run_from_config(
                args.config,
                action="auto",
            )
        )

    if args.command == "compile":
        raise SystemExit(
            run_from_config(
                args.config,
                action="compile",
            )
        )

    if args.command == "benchmark":
        raise SystemExit(
            run_from_config(
                args.config,
                action="benchmark",
            )
        )

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()