from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

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
from fpgai.experiments.sweep_runner import run_sweep_config

T = TypeVar("T")


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


def _default_cli_log_dir() -> Path:
    return Path("build") / "cli_logs"


def _new_log_paths(prefix: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = _default_cli_log_dir()
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return (
        log_dir / f"{stamp}_{prefix}_stdout.log",
        log_dir / f"{stamp}_{prefix}_stderr.log",
    )


def _run_captured(
    fn: Callable[[], T],
    *,
    prefix: str,
) -> tuple[T, Path, Path]:
    stdout_path, stderr_path = _new_log_paths(prefix)

    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_f:
        with contextlib.redirect_stdout(stdout_f), contextlib.redirect_stderr(
            stderr_f
        ):
            result = fn()

    return result, stdout_path, stderr_path


def _print_log_paths(
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    print("---------------------------------------------------")
    print(f"Captured stdout log : {stdout_path}")
    print(f"Captured stderr log : {stderr_path}")


def _print_compile_result(
    result,
    *,
    quiet: bool,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> None:
    if quiet:
        print("=============== FPGAI Compile Result ===============")
        print(f"Out dir              : {result.out_dir}")
        print(f"HLS ran              : {getattr(result, 'hls_ran', None)}")
        print(f"HLS ok               : {getattr(result, 'hls_ok', None)}")
        print(f"HLS returncode       : {getattr(result, 'hls_returncode', None)}")
    else:
        print(result.summary())

    if stdout_path is not None and stderr_path is not None:
        _print_log_paths(
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    print(
        f"[OK] Wrote artifacts to: {result.out_dir}"
    )


def _print_benchmark_result(
    benchmark,
    *,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> None:
    print("\n=============== FPGAI Benchmark ===============")
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

    if stdout_path is not None and stderr_path is not None:
        _print_log_paths(
            stdout_path=stdout_path,
            stderr_path=stderr_path,
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


def run_from_config(
    config_path: str,
    *,
    action: str = "auto",
    verbose: bool = False,
    quiet: bool = False,
) -> int:
    if verbose and quiet:
        print(
            "[ERROR] --verbose and --quiet cannot be used together.",
            file=sys.stderr,
        )
        return 2

    try:
        cfg = load_config(config_path)
        if verbose:
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

            def _run_benchmark():
                return run_compile_correctness_benchmark(
                    config_path=config_path,
                )

            if verbose:
                benchmark = _run_benchmark()
                _print_benchmark_result(benchmark)
            else:
                benchmark, stdout_path, stderr_path = _run_captured(
                    _run_benchmark,
                    prefix="benchmark",
                )
                _print_benchmark_result(
                    benchmark,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )

            return 0

        def _run_compile():
            compiler = Compiler(cfg)
            return compiler.compile()

        if verbose:
            result = _run_compile()
            _print_compile_result(
                result,
                quiet=quiet,
            )
        else:
            result, stdout_path, stderr_path = _run_captured(
                _run_compile,
                prefix="compile",
            )
            _print_compile_result(
                result,
                quiet=quiet,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
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


def _load_yaml_document(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected top-level YAML mapping in {config_path}"
        )
    return data


def _write_json_report(report: dict[str, Any], json_output: str | None) -> Path | None:
    if json_output is None:
        return None
    output_path = Path(json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.keys())
    return [value]


def inspect_sweep_config(
    config_path: str,
    *,
    json_output: str | None = None,
) -> int:
    try:
        data = _load_yaml_document(config_path)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    known_keys = {
        "name",
        "version",
        "description",
        "defaults",
        "parameters",
        "command_template",
        "design_name_template",
        "point_name_template",
        "materialize_configs",
        "metadata",
    }
    top_keys = sorted(data.keys())
    unknown_keys = sorted(k for k in top_keys if k not in known_keys)
    parameters = data.get("parameters")
    defaults = data.get("defaults")
    errors: list[str] = []

    if not isinstance(defaults, dict):
        errors.append("defaults: expected mapping")
    if not isinstance(parameters, (dict, list)) or len(parameters) == 0:
        errors.append("parameters: expected non-empty mapping or list")
    if "command_template" not in data:
        errors.append("command_template: missing")

    points = _as_list(parameters)
    report = {
        "kind": "sweep",
        "config": config_path,
        "valid": not errors,
        "name": data.get("name"),
        "top_level_keys": top_keys,
        "unknown_keys": unknown_keys,
        "parameter_count": len(points),
        "has_defaults": isinstance(defaults, dict),
        "has_command_template": "command_template" in data,
        "errors": errors,
    }

    print("=============== FPGAI Sweep Config Inspection ===============")
    print(f"Config                : {config_path}")
    print(f"Name                  : {report['name']}")
    print(f"Valid                 : {report['valid']}")
    print(f"Parameters            : {report['parameter_count']}")
    print(f"Has defaults          : {report['has_defaults']}")
    print(f"Has command template  : {report['has_command_template']}")
    if unknown_keys:
        print(f"Unknown keys          : {unknown_keys}")
    if errors:
        print("--------------------------------------------------------------")
        for error in errors:
            print(f" - {error}")
    print("==============================================================")

    output_path = _write_json_report(report, json_output)
    if output_path is not None:
        print(f"[OK] Wrote sweep inspection JSON to: {output_path}")

    return 0 if not errors else 1


def inspect_experiment_config(
    config_path: str,
    *,
    json_output: str | None = None,
) -> int:
    try:
        data = _load_yaml_document(config_path)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    top_keys = sorted(data.keys())
    required = ["paper", "inputs", "claim_levels", "limitations"]
    errors = [f"{key}: missing" for key in required if key not in data]

    paper = data.get("paper") or {}
    inputs = data.get("inputs") or {}
    claim_levels = data.get("claim_levels") or {}
    limitations = data.get("limitations") or []

    if "paper" in data and not isinstance(paper, dict):
        errors.append("paper: expected mapping")
    if "inputs" in data and not isinstance(inputs, dict):
        errors.append("inputs: expected mapping")
    if "claim_levels" in data and not isinstance(claim_levels, dict):
        errors.append("claim_levels: expected mapping")
    if "limitations" in data and not isinstance(limitations, (list, dict)):
        errors.append("limitations: expected list or mapping")

    report = {
        "kind": "paper_experiment",
        "config": config_path,
        "valid": not errors,
        "paper_title": paper.get("title") if isinstance(paper, dict) else None,
        "top_level_keys": top_keys,
        "input_count": len(inputs) if isinstance(inputs, dict) else 0,
        "claim_level_count": len(claim_levels) if isinstance(claim_levels, dict) else 0,
        "limitation_count": len(limitations) if isinstance(limitations, (list, dict)) else 0,
        "errors": errors,
    }

    print("============== FPGAI Experiment Config Inspection ==============")
    print(f"Config                : {config_path}")
    print(f"Paper title           : {report['paper_title']}")
    print(f"Valid                 : {report['valid']}")
    print(f"Inputs                : {report['input_count']}")
    print(f"Claim levels          : {report['claim_level_count']}")
    print(f"Limitations           : {report['limitation_count']}")
    if errors:
        print("-----------------------------------------------------------------")
        for error in errors:
            print(f" - {error}")
    print("================================================================")

    output_path = _write_json_report(report, json_output)
    if output_path is not None:
        print(f"[OK] Wrote experiment inspection JSON to: {output_path}")

    return 0 if not errors else 1



def run_sweep_from_config(
    config_path: str,
    *,
    out_dir: str | None = None,
    max_design_points: int | None = None,
    timeout_sec: int | None = None,
    dry_run: bool = False,
    repo_root: str = ".",
) -> int:
    """Run a sweep config through the package-level sweep runner.

    This is the public CLI path for experiment matrices. It intentionally
    keeps the command surface inside ``fpgai`` instead of requiring users to
    call ``scripts/run_fpgai_experiments.py`` directly.
    """

    try:
        repo_root_path = Path(repo_root).resolve()
        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            config_path_obj = repo_root_path / config_path_obj

        if out_dir is None:
            data = _load_yaml_document(str(config_path_obj))
            sweep_name = str(data.get("name") or config_path_obj.stem)
            out_path = repo_root_path / "experiments" / sweep_name
        else:
            out_path = Path(out_dir)
            if not out_path.is_absolute():
                out_path = repo_root_path / out_path

        payload = run_sweep_config(
            config_path_obj,
            experiment_dir=out_path,
            limit=max_design_points,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )

        results = payload.get("results", [])
        if not isinstance(results, list):
            results = []

        failed = int(payload.get("failed_count", 0) or 0)
        if failed == 0 and results:
            failed = sum(
                1 for row in results if isinstance(row, dict) and row.get("status") == "failed"
            )

        total = int(payload.get("result_count", len(results)) or 0)
        experiment_dir = Path(str(payload.get("experiment_dir", out_path)))
        results_path = experiment_dir / "results.json"

        print("=============== FPGAI Sweep Run ===============")
        print(f"Config              : {config_path_obj}")
        print(f"Out dir             : {experiment_dir}")
        print(f"Design points       : {total}")
        print(f"Failed              : {failed}")
        print(f"Dry run             : {bool(dry_run)}")
        print(f"Results JSON        : {results_path}")
        print("================================================")

        if failed:
            print(f"[WARN] Sweep completed with {failed} failed record(s).")
        else:
            print("[OK] Sweep completed successfully.")

        return 0

    except FileNotFoundError as exc:
        print(f"[ERROR] File not found: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
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
        help="Path to single FPGAI compile YAML",
    )
    compile_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream compiler/tool output to the terminal.",
    )
    compile_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only a minimal compile summary.",
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
        help="Path to single FPGAI compile YAML",
    )
    benchmark_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream benchmark/tool output to the terminal.",
    )
    benchmark_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Accepted for consistency; benchmark already prints a concise summary.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help=(
            "Inspect a single compile config, model tensors, "
            "and backend operator support"
        ),
    )
    inspect_parser.add_argument(
        "--config",
        required=True,
        help="Path to single FPGAI compile YAML",
    )
    inspect_parser.add_argument(
        "--json-output",
        help=(
            "Optional path for the machine-readable "
            "inspection report"
        ),
    )

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Inspect or run FPGAI sweep/matrix configs",
    )
    sweep_subparsers = sweep_parser.add_subparsers(
        dest="sweep_command"
    )
    sweep_inspect_parser = sweep_subparsers.add_parser(
        "inspect",
        help="Inspect a sweep YAML without running experiments",
    )
    sweep_inspect_parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/sweeps/*.yml",
    )
    sweep_inspect_parser.add_argument(
        "--json-output",
        help="Optional path for sweep inspection JSON",
    )

    sweep_run_parser = sweep_subparsers.add_parser(
        "run",
        help="Run a sweep/matrix YAML and write experiment results",
    )
    sweep_run_parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/sweeps/*.yml",
    )
    sweep_run_parser.add_argument(
        "--out",
        required=True,
        help="Output experiment directory",
    )
    sweep_run_parser.add_argument(
        "--max-design-points",
        type=int,
        default=None,
        help="Optional limit on generated design points",
    )
    sweep_run_parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Optional per-design timeout in seconds",
    )
    sweep_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize records/logs without executing commands",
    )
    sweep_run_parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root for resolving relative sweep paths",
    )

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Inspect or run paper experiment configs",
    )
    experiment_subparsers = experiment_parser.add_subparsers(
        dest="experiment_command"
    )
    experiment_inspect_parser = experiment_subparsers.add_parser(
        "inspect",
        help="Inspect a paper experiment YAML without running experiments",
    )
    experiment_inspect_parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/experiments/*.yml",
    )
    experiment_inspect_parser.add_argument(
        "--json-output",
        help="Optional path for experiment inspection JSON",
    )

    parser.add_argument(
        "--config",
        help=(
            "Backward-compatible config argument. "
            "Equivalent to: fpgai compile --config ..."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output for backward-compatible --config mode.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet output for backward-compatible --config mode.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compile":
        raise SystemExit(
            run_from_config(
                args.config,
                action="compile",
                verbose=args.verbose,
                quiet=args.quiet,
            )
        )

    if args.command == "benchmark":
        raise SystemExit(
            run_from_config(
                args.config,
                action="benchmark",
                verbose=args.verbose,
                quiet=args.quiet,
            )
        )

    if args.command == "inspect":
        raise SystemExit(
            inspect_from_config(
                args.config,
                json_output=args.json_output,
            )
        )

    if args.command == "sweep":
        if args.sweep_command == "inspect":
            raise SystemExit(
                inspect_sweep_config(
                    args.config,
                    json_output=args.json_output,
                )
            )
        if args.sweep_command == "run":
            raise SystemExit(
                run_sweep_from_config(
                    args.config,
                    out_dir=args.out,
                    max_design_points=args.max_design_points,
                    timeout_sec=args.timeout_sec,
                    dry_run=args.dry_run,
                    repo_root=args.repo_root,
                )
            )
        parser.error("sweep requires a subcommand, e.g. 'inspect' or 'run'")

    if args.command == "experiment":
        if args.experiment_command == "inspect":
            raise SystemExit(
                inspect_experiment_config(
                    args.config,
                    json_output=args.json_output,
                )
            )
        parser.error("experiment requires a subcommand, e.g. 'inspect'")

    if args.config:
        raise SystemExit(
            run_from_config(
                args.config,
                action="auto",
                verbose=args.verbose,
                quiet=args.quiet,
            )
        )

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
