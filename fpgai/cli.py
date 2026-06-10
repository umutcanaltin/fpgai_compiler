from __future__ import annotations

import argparse
import sys

from fpgai.analysis.model_inspection import inspect_config
from fpgai.benchmark.pipeline import (
    run_compile_correctness_benchmark,
)
from fpgai.config.loader import (
    ConfigError,
    load_config,
    print_summary,
)
from fpgai.engine.compiler import Compiler


def _benchmark_enabled(cfg) -> bool:
    raw = getattr(
        cfg,
        "raw",
        {},
    ) or {}

    benchmark = raw.get(
        "benchmark",
        {},
    ) or {}

    return bool(
        benchmark.get(
            "enabled",
            False,
        )
    )


def _should_run_inference_benchmark(cfg) -> bool:
    return (
        cfg.pipeline.mode == "inference"
        and _benchmark_enabled(cfg)
    )


def run_from_config(
    config_path: str,
    *,
    action: str = "auto",
) -> int:
    try:
        cfg = load_config(config_path)
        print_summary(cfg)
    except ConfigError as exc:
        print(
            str(exc),
            file=sys.stderr,
        )
        return 2

    try:
        if action not in {
            "auto",
            "compile",
            "benchmark",
        }:
            print(
                f"[ERROR] Unsupported CLI action: {action}",
                file=sys.stderr,
            )
            return 2

        should_benchmark = (
            action == "benchmark"
            or (
                action == "auto"
                and _should_run_inference_benchmark(cfg)
            )
        )

        if should_benchmark:
            if cfg.pipeline.mode != "inference":
                print(
                    "[ERROR] Correctness benchmarking currently supports "
                    "pipeline.mode=inference only.",
                    file=sys.stderr,
                )
                return 2

            print(
                "\n=============== FPGAI Benchmark ==============="
            )

            benchmark = run_compile_correctness_benchmark(
                config_path=config_path,
            )

            print(
                f"Benchmark passed : {benchmark.passed}"
            )
            print(
                f"Metrics JSON     : {benchmark.metrics_json}"
            )
            print(
                f"Summary TXT      : {benchmark.summary_txt}"
            )

            if benchmark.quant_metrics_json is not None:
                print(
                    f"Quant JSON       : "
                    f"{benchmark.quant_metrics_json}"
                )

            if benchmark.quant_summary_txt is not None:
                print(
                    f"Quant Summary    : "
                    f"{benchmark.quant_summary_txt}"
                )

            precision_results = getattr(
                benchmark,
                "precision_sweep_results_json",
                None,
            )

            if precision_results is not None:
                print(
                    f"Sweep JSON       : {precision_results}"
                )

            precision_summary = getattr(
                benchmark,
                "precision_sweep_summary_txt",
                None,
            )

            if precision_summary is not None:
                print(
                    f"Sweep Summary    : {precision_summary}"
                )

            print(
                "=============================================="
            )

            if benchmark.passed:
                print(
                    "[OK] Benchmark completed successfully."
                )
            else:
                print(
                    "[WARN] Benchmark completed but did not "
                    "pass thresholds."
                )

            return 0

        compiler = Compiler(cfg)
        result = compiler.compile()

        print(result.summary())
        print(
            f"[OK] Wrote artifacts to: {result.out_dir}"
        )

        return 0

    except RuntimeError as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )
        return 1


def inspect_from_config(
    config_path: str,
    *,
    json_output: str | None = None,
) -> int:
    try:
        cfg = load_config(config_path)
        report = inspect_config(cfg)

        print(report.summary())

        if json_output is not None:
            output_path = report.write_json(
                json_output
            )
            print(
                f"[OK] Wrote inspection JSON to: {output_path}"
            )

        if report.compilation_ready:
            return 0

        return 1

    except ConfigError as exc:
        print(
            str(exc),
            file=sys.stderr,
        )
        return 2
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "FPGAI compiler"
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile using a YAML config",
    )
    compile_parser.add_argument(
        "--config",
        required=True,
        help="Path to fpgai.yml",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help=(
            "Run compile/correctness benchmark "
            "using a YAML config"
        ),
    )
    benchmark_parser.add_argument(
        "--config",
        required=True,
        help="Path to fpgai.yml",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help=(
            "Inspect model tensors, parameters, "
            "and backend operator support"
        ),
    )
    inspect_parser.add_argument(
        "--config",
        required=True,
        help="Path to fpgai.yml",
    )
    inspect_parser.add_argument(
        "--json-output",
        help=(
            "Optional path for the machine-readable "
            "inspection report"
        ),
    )

    parser.add_argument(
        "--config",
        help=(
            "Backward-compatible config argument. "
            "Equivalent to: fpgai compile --config ..."
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

    if args.command == "inspect":
        raise SystemExit(
            inspect_from_config(
                args.config,
                json_output=args.json_output,
            )
        )

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()