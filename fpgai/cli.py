from __future__ import annotations

from fpgai.reporting.artifacts import build_report
from fpgai.validation.results import validate_results
import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

from fpgai.analysis.model_inspection import inspect_config, write_model_inspection_report
from fpgai.analysis.performance_estimator import estimate_performance
from fpgai.analysis.resource_estimator import estimate_resources_from_descriptors
from fpgai.engine.planner import make_compile_plan
from fpgai.engine.analysis import analyze_graph
from fpgai.ir.passes import assign_stable_names
from fpgai.config.loader import (
    ConfigError,
    load_config,
    print_summary,
)
from fpgai.engine.compiler import Compiler
from fpgai.experiments.sweep_runner import run_sweep_config
from fpgai.experiments.benchmark_runner import run_experiment_from_config

T = TypeVar("T")


def run_compile_correctness_benchmark(*, config_path: str):
    """Lazy optional-dependency boundary for inference benchmarking."""
    from fpgai.benchmark.pipeline import run_compile_correctness_benchmark as _run
    return _run(config_path=config_path)


def run_compile_training_benchmark(*, config_path: str):
    """Lazy optional-dependency boundary for training benchmarking."""
    from fpgai.benchmark.pipeline import run_compile_training_benchmark as _run
    return _run(config_path=config_path)


def validate_correctness(config_path: str):
    """Lazy optional-dependency boundary for correctness validation."""
    from fpgai.validation.correctness import validate_correctness as _validate
    return _validate(config_path)


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

        manifest_sections = []
        for helper_name in [
            "_manifest_summary_lines",
            "_prediction_artifact_summary_lines",
            "_design_space_summary_lines",
            "_hls_artifacts_summary_lines",
            "_vivado_bridge_summary_lines",
            "_runtime_package_summary_lines",
            "_pipeline_stage_summary_lines",
        ]:
            helper = getattr(result, helper_name, None)
            if helper is None:
                continue
            section = helper()
            if section:
                manifest_sections.append(section)

        for section in manifest_sections:
            print("---------------------------------------------------")
            for line in section:
                print(line)
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
            if cfg.pipeline.mode == "inference":
                def _run_benchmark():
                    return run_compile_correctness_benchmark(
                        config_path=config_path,
                    )
            elif cfg.pipeline.mode == "training_on_device":
                def _run_benchmark():
                    return run_compile_training_benchmark(
                        config_path=config_path,
                    )
            else:
                print(
                    "[ERROR] Benchmarking supports pipeline.mode=inference "
                    "and pipeline.mode=training_on_device.",
                    file=sys.stderr,
                )
                return 2

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


def _inspection_predictions_from_config(cfg) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build honest pre-HLS resource/timing predictions for inspect artifacts."""
    raw_cfg = getattr(cfg, "raw", {}) or {}
    model_path = (
        raw_cfg.get("model", {}).get("path")
        or raw_cfg.get("model_path")
        or getattr(getattr(cfg, "model", None), "path", None)
    )

    if model_path is None:
        raise ValueError("Cannot build predictions without a model path")

    from fpgai.frontend.onnx import import_onnx
    graph = import_onnx(
        model_path,
        canonicalize=True,
        infer_shapes=True,
    )
    graph = assign_stable_names(graph)

    descriptors = analyze_graph(graph)
    compile_plan = make_compile_plan(
        cfg,
        descriptors,
    )

    resource_prediction = estimate_resources_from_descriptors(
        descriptors,
        raw_cfg,
        compile_plan=compile_plan,
    )
    timing_prediction = estimate_performance(
        resource_estimate=resource_prediction,
        raw_cfg=raw_cfg,
    )

    resource_prediction = dict(resource_prediction)
    timing_prediction = dict(timing_prediction)

    resource_prediction["prediction_kind"] = "pre_hls_resource_estimate"
    resource_prediction["prediction_status"] = "estimate"
    resource_prediction["model_path"] = str(model_path)
    resource_prediction["descriptor_count"] = len(descriptors)
    resource_prediction["architecture_signature"] = getattr(
        compile_plan,
        "architecture_signature",
        None,
    )

    timing_prediction["prediction_kind"] = "pre_hls_timing_estimate"
    timing_prediction["prediction_status"] = "estimate"
    timing_prediction["model_path"] = str(model_path)
    timing_prediction["descriptor_count"] = len(descriptors)
    timing_prediction["architecture_signature"] = getattr(
        compile_plan,
        "architecture_signature",
        None,
    )

    return resource_prediction, timing_prediction


def inspect_from_config(
    config_path: str,
    *,
    json_output: str | None = None,
    out: str | None = None,
) -> int:
    try:
        cfg = load_config(config_path)
        report = inspect_config(cfg)

        wrote_anything = False

        if json_output is not None:
            output_path = report.write_json(
                json_output
            )
            print(
                f"[OK] Wrote inspection JSON to: {output_path}"
            )
            wrote_anything = True

        if out is not None:
            resource_prediction, timing_prediction = (
                _inspection_predictions_from_config(cfg)
            )
            paths = write_model_inspection_report(
                report,
                out,
                resource_prediction=resource_prediction,
                timing_prediction=timing_prediction,
            )
            print(f"[OK] Model profile JSON: {paths['model_profile_json']}")
            print(f"[OK] Resource prediction JSON: {paths['resource_prediction_json']}")
            print(f"[OK] Timing prediction JSON: {paths['timing_prediction_json']}")
            print(f"[OK] Prediction summary: {paths['prediction_summary_md']}")
            if paths.get("model_gap_audit_json"):
                print(f"[OK] Model gap audit JSON: {paths['model_gap_audit_json']}")
            wrote_anything = True

        if not wrote_anything:
            print(report.summary())

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
        "design_points",
        "command_template",
        "design_name_template",
        "point_name_template",
        "materialize_configs",
        "metadata",
        "vivado",
    }
    top_keys = sorted(data.keys())
    unknown_keys = sorted(k for k in top_keys if k not in known_keys)
    parameters = data.get("parameters")
    design_points = data.get("design_points")
    defaults = data.get("defaults")
    errors: list[str] = []

    vivado_cfg = data.get("vivado")
    has_parameters = isinstance(parameters, (dict, list)) and len(parameters) > 0
    has_design_points = isinstance(design_points, list) and len(design_points) > 0
    has_vivado_marker = isinstance(vivado_cfg, dict) and bool(vivado_cfg.get("enabled", False))

    if not isinstance(defaults, dict) and not has_vivado_marker:
        errors.append("defaults: expected mapping")
    if not has_parameters and not has_design_points and not has_vivado_marker:
        errors.append("parameters/design_points: expected non-empty mapping/list, non-empty design_points list, or enabled vivado marker")
    if has_parameters and "command_template" not in data:
        errors.append("command_template: missing for parameter sweep")

    points = _as_list(parameters) if has_parameters else []
    report = {
        "kind": "sweep",
        "config": config_path,
        "valid": not errors,
        "name": data.get("name"),
        "top_level_keys": top_keys,
        "unknown_keys": unknown_keys,
        "parameter_count": len(points),
        "design_point_count": len(design_points) if isinstance(design_points, list) else 0,
        "has_defaults": isinstance(defaults, dict),
        "has_command_template": "command_template" in data,
        "has_vivado_marker": has_vivado_marker,
        "errors": errors,
    }

    print("=============== FPGAI Sweep Config Inspection ===============")
    print(f"Config                : {config_path}")
    print(f"Name                  : {report['name']}")
    print(f"Valid                 : {report['valid']}")
    print(f"Parameters            : {report['parameter_count']}")
    print(f"Design points         : {report['design_point_count']}")
    print(f"Has defaults          : {report['has_defaults']}")
    print(f"Has command template  : {report['has_command_template']}")
    print(f"Has Vivado marker     : {report['has_vivado_marker']}")
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
    required = ["benchmark", "inputs", "limitations"]
    errors = [f"{key}: missing" for key in required if key not in data]
    if "validation_levels" not in data and "claim_levels" not in data:
        errors.append("validation_levels: missing")

    benchmark = data.get("benchmark") or {}
    inputs = data.get("inputs") or {}
    validation_levels = data.get("validation_levels") or data.get("claim_levels") or {}
    limitations = data.get("limitations") or []

    if "benchmark" in data and not isinstance(benchmark, dict):
        errors.append("benchmark: expected mapping")
    if "inputs" in data and not isinstance(inputs, dict):
        errors.append("inputs: expected mapping")
    if ("validation_levels" in data or "claim_levels" in data) and not isinstance(validation_levels, dict):
        errors.append("validation_levels: expected mapping")
    if "limitations" in data and not isinstance(limitations, (list, dict)):
        errors.append("limitations: expected list or mapping")

    report = {
        "kind": "benchmark",
        "config": config_path,
        "valid": not errors,
        "benchmark_title": benchmark.get("title") if isinstance(benchmark, dict) else None,
        "top_level_keys": top_keys,
        "input_count": len(inputs) if isinstance(inputs, dict) else 0,
        "validation_level_count": len(validation_levels) if isinstance(validation_levels, dict) else 0,
        "limitation_count": len(limitations) if isinstance(limitations, (list, dict)) else 0,
        "errors": errors,
    }

    print("============== FPGAI Experiment Config Inspection ==============")
    print(f"Config                : {config_path}")
    print(f"Benchmark title           : {report['benchmark_title']}")
    print(f"Valid                 : {report['valid']}")
    print(f"Inputs                : {report['input_count']}")
    print(f"Validation levels          : {report['validation_level_count']}")
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



def _tail_text_file(path_value: object, *, max_lines: int = 8) -> list[str]:
    """Return a compact non-empty tail from a sweep log path."""

    if not path_value:
        return []
    try:
        path = Path(str(path_value))
        if not path.exists() or not path.is_file():
            return []
        lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
        lines = [line for line in lines if line.strip()]
        return lines[-max_lines:]
    except OSError:
        return []


def _print_sweep_failures(results: list[dict[str, object]]) -> None:
    failed_rows = [row for row in results if row.get("status") == "failed"]
    if not failed_rows:
        return

    print("---------------- Sweep failure details ----------------")
    for row in failed_rows:
        name = row.get("design_name") or "<unnamed>"
        print(f"Design              : {name}")
        print(f"Return code         : {row.get('returncode')}")
        print(f"Reason              : {row.get('error') or 'command failed'}")
        print(f"Generated config    : {row.get('config_path')}")
        print(f"Stdout log          : {row.get('stdout_log')}")
        print(f"Stderr log          : {row.get('stderr_log')}")
        tail = _tail_text_file(row.get("stderr_log"))
        if not tail:
            tail = _tail_text_file(row.get("stdout_log"))
        if tail:
            print("Log tail:")
            for line in tail:
                print(f"  {line}")
        print("--------------------------------------------------------")


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
    call internal experiment runner scripts directly.
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
            _print_sweep_failures([row for row in results if isinstance(row, dict)])
            print(f"[ERROR] Sweep completed with {failed} failed record(s).")
            return 1

        print("[OK] Sweep completed successfully.")
        return 0

    except FileNotFoundError as exc:
        print(f"[ERROR] File not found: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1



def _handle_report_build(args) -> int:
    try:
        result = build_report(args.input, args.out)
    except Exception as exc:
        print(f"[ERROR] Failed to build report: {exc}")
        return 2

    print("=============== FPGAI Report Build ===============")
    print(f"Input              : {result.input_dir}")
    print(f"Out dir            : {result.out_dir}")
    print(f"Result records     : {result.result_count}")
    print(f"Passed records     : {result.passed_count}")
    print(f"Failed records     : {result.failed_count}")
    print(f"Skipped records    : {result.skipped_count}")
    print(f"Summary            : {result.summary_md}")
    print(f"Results table      : {result.results_table_csv}")
    print(f"Claim traceability : {result.claim_traceability_md}")
    print("===================================================")
    return 0



def _run_existing_reporting_main(module_name: str, argv: list[str]) -> int:
    """Run an existing fpgai.reporting module main() through the public CLI.

    This keeps report logic in the existing reporting modules and only wires it
    into `fpgai report ...`.
    """
    import importlib

    old_argv = sys.argv[:]
    sys.argv = [f"fpgai report {module_name.rsplit('.', 1)[-1]}", *argv]

    try:
        module = importlib.import_module(module_name)
        result = module.main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Failed to run {module_name}: {exc}", file=sys.stderr)
        return 2
    finally:
        sys.argv = old_argv

    return int(result or 0)



def _handle_report_benchmark_artifacts(args) -> int:
    return _run_existing_reporting_main(
        "fpgai.reporting.generate_benchmark_artifacts",
        ["--csv", args.csv, "--outdir", args.out],
    )



def _handle_report_frontier(args) -> int:
    argv = ["--csv", args.csv, "--outdir", args.out]
    if getattr(args, "require_pass", False):
        argv.append("--require-pass")
    return _run_existing_reporting_main(
        "fpgai.reporting.design_frontier",
        argv,
    )



def _handle_report_estimator(args) -> int:
    argv = [
        "--csv",
        args.csv,
        "--out-dir",
        args.out,
        "--inference-filter",
        args.inference_filter,
        "--training-filter",
        args.training_filter,
    ]
    return _run_existing_reporting_main(
        "fpgai.reporting.estimator_tables",
        argv,
    )



def _handle_validate_correctness(args) -> int:
    try:
        result = validate_correctness(args.config)
    except Exception as exc:
        print(f"[ERROR] Correctness validation failed: {exc}")
        return 2

    print("============== FPGAI Correctness Validation ==============")
    print(f"Config         : {result.config_path}")
    print(f"Pipeline mode  : {result.pipeline_mode}")
    print(f"Requested      : {result.requested}")
    print(f"Executed       : {result.executed}")
    print(f"Passed         : {result.passed}")

    if result.reason:
        print(f"Reason         : {result.reason}")
    if result.build_dir is not None:
        print(f"Build dir      : {result.build_dir}")
    if result.bench_dir is not None:
        print(f"Bench dir      : {result.bench_dir}")
    if result.metrics_json is not None:
        print(f"Metrics JSON   : {result.metrics_json}")
    if result.summary_txt is not None:
        print(f"Summary TXT    : {result.summary_txt}")

    print("===========================================================")
    return 0 if result.passed else 1


def _handle_validate_results(args) -> int:
    result = validate_results(args.input)

    print("============== FPGAI Results Validation ==============")
    print(f"Input          : {result.input_dir}")
    print(f"Passed         : {result.passed}")
    print(f"Child items    : {result.child_count}")
    print(f"Failed children: {result.failed_child_count}")
    print(f"Errors         : {result.error_count}")
    print(f"Warnings       : {result.warning_count}")

    for issue in result.issues:
        print(f"[{issue.level.upper()}] {issue.path}: {issue.message}")

    print("=======================================================")
    return 0 if result.passed else 1


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
    inspect_parser.add_argument(
        "--out",
        default=None,
        help="Optional directory for model inspection/prediction artifacts.",
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
        help="Inspect or run benchmark configs",
    )
    experiment_subparsers = experiment_parser.add_subparsers(
        dest="experiment_command"
    )
    experiment_inspect_parser = experiment_subparsers.add_parser(
        "inspect",
        help="Inspect a benchmark YAML without running experiments",
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

    experiment_run_parser = experiment_subparsers.add_parser(
        "run",
        help="Run a benchmark YAML",
    )
    experiment_run_parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/experiments/*.yml",
    )
    experiment_run_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for benchmark results",
    )
    experiment_run_parser.add_argument(
        "--max-design-points",
        type=int,
        default=None,
        help="Optional cap per sweep for smoke runs",
    )
    experiment_run_parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Optional timeout per design point",
    )
    experiment_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve experiment items without running sweeps",
    )
    experiment_run_parser.add_argument(
        "--repo-root",
        default=".",
        help="Optional repository root override",
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


    report_parser = subparsers.add_parser(
        "report",
        help="Generate reports from FPGAI experiment outputs",
    )
    report_subparsers = report_parser.add_subparsers(dest="report_command")

    report_build_parser = report_subparsers.add_parser(
        "build",
        help="Build summary tables and claim traceability from experiment outputs",
    )
    report_build_parser.add_argument(
        "--input",
        required=True,
        help="Input experiment output directory, for example benchmark_runs/publication",
    )
    report_build_parser.add_argument(
        "--out",
        required=True,
        help="Output report directory, for example reports/publication",
    )

    report_benchmark_artifacts_parser = report_subparsers.add_parser(
        "benchmark-artifacts",
        help="Generate benchmark tables and figures from an existing sweep CSV",
    )
    report_benchmark_artifacts_parser.add_argument(
        "--csv",
        required=True,
        help="Input policy_sweep_results.csv or compatible FPGAI result CSV",
    )
    report_benchmark_artifacts_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for generated benchmark artifacts",
    )

    report_frontier_parser = report_subparsers.add_parser(
        "frontier",
        help="Generate DSP-vs-latency Pareto frontier artifacts from an existing sweep CSV",
    )
    report_frontier_parser.add_argument(
        "--csv",
        required=True,
        help="Input policy_sweep_results.csv or compatible FPGAI result CSV",
    )
    report_frontier_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for generated frontier artifacts",
    )
    report_frontier_parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Keep only rows with benchmark_passed=True when available",
    )

    report_estimator_parser = report_subparsers.add_parser(
        "estimator",
        help="Generate estimator-vs-real resource and latency tables from an existing CSV",
    )
    report_estimator_parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV containing predicted and actual resource/latency columns",
    )
    report_estimator_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for estimator comparison tables",
    )
    report_estimator_parser.add_argument(
        "--inference-filter",
        default="inference",
        help="Substring used to identify inference rows in the CSV mode column",
    )
    report_estimator_parser.add_argument(
        "--training-filter",
        default="training",
        help="Substring used to identify training rows in the CSV mode column",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate FPGAI experiment outputs",
    )
    validate_subparsers = validate_parser.add_subparsers(dest="validate_command")

    validate_results_parser = validate_subparsers.add_parser(
        "results",
        help="Validate experiment result files and detect false passes",
    )
    validate_results_parser.add_argument(
        "--input",
        required=True,
        help="Input experiment output directory, for example benchmark_runs/publication",
    )


    validate_correctness_parser = validate_subparsers.add_parser(
        "correctness",
        help="Run inference correctness validation for a compile config",
    )
    validate_correctness_parser.add_argument(
        "--config",
        required=True,
        help="Compile config to validate, for example configs/examples/inference_compile.yml",
    )


    ecosystem_parser = subparsers.add_parser(
        "ecosystem",
        help="Create, validate, discover, and inspect FPGAI Ecosystem contributions",
    )
    ecosystem_subparsers = ecosystem_parser.add_subparsers(dest="ecosystem_command")
    ecosystem_init = ecosystem_subparsers.add_parser("init", help="Create an FPGAI Ecosystem contribution scaffold")
    ecosystem_init.add_argument("--type", dest="asset_type", required=True, help="model, operator/layer, implementation, hls, vhdl, board, backend, dataset, optimizer, loss, reporter, ...")
    ecosystem_init.add_argument("--id", dest="package_id", required=True, help="Namespace-qualified ecosystem package ID")
    ecosystem_init.add_argument("--out", required=True, help="Output contribution directory")
    ecosystem_init.add_argument("--name", default=None)
    ecosystem_init.add_argument("--language", choices=["hls_cpp", "vhdl", "verilog", "systemverilog"], default=None)
    ecosystem_init.add_argument("--force", action="store_true")
    ecosystem_validate = ecosystem_subparsers.add_parser("validate", help="Validate an ecosystem package manifest without executing contributor code")
    ecosystem_validate.add_argument("path", help="Contribution root containing fpgai.yaml")
    ecosystem_validate.add_argument("--json", action="store_true", dest="json_output")
    ecosystem_inventory = ecosystem_subparsers.add_parser("inventory", help="Show the built-in FPGAI Ecosystem registry inventory")
    ecosystem_inventory.add_argument("--json", action="store_true", dest="json_output")
    ecosystem_discover = ecosystem_subparsers.add_parser("discover", help="Discover built-in and project-local FPGAI Ecosystem packages")
    ecosystem_discover.add_argument("--project-root", default=".", help="Project root containing packages/")
    ecosystem_discover.add_argument("--directory", action="append", default=[], help="Additional ecosystem package directory; may be repeated")
    ecosystem_discover.add_argument("--max-depth", type=int, default=2)
    ecosystem_discover.add_argument("--strict", action="store_true")
    ecosystem_discover.add_argument("--json", action="store_true", dest="json_output")
    ecosystem_subparsers.add_parser("types", help="List supported FPGAI Ecosystem contribution asset types")

    frontend_parser = subparsers.add_parser(
        "frontend",
        help="Inspect or import registered model-source frontends",
    )
    frontend_subparsers = frontend_parser.add_subparsers(dest="frontend_command")
    frontend_subparsers.add_parser("list", help="List registered source frontends")
    frontend_subparsers.add_parser("routes", help="Show framework-to-FPGAI ingress routes and legalization status")
    frontend_import_parser = frontend_subparsers.add_parser(
        "import", help="Import ONNX/MLIR/StableHLO to FPGAI IR without compiling"
    )
    frontend_import_parser.add_argument("--input", required=True, help="Input model/IR path")
    frontend_import_parser.add_argument("--format", dest="format_hint", default=None, help="onnx, mlir, stablehlo, or external registered frontend")
    frontend_import_parser.add_argument("--framework", default=None, help="Optional source framework provenance, e.g. jax or pytorch")
    frontend_import_parser.add_argument("--pipeline-mode", default="inference")
    frontend_import_parser.add_argument("--target-board", default=None)
    frontend_import_parser.add_argument("--out", required=True, help="Output directory")

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


    if args.command == "report":
        if getattr(args, "report_command", None) == "build":
            return _handle_report_build(args)
        if getattr(args, "report_command", None) == "benchmark-artifacts":
            raise SystemExit(_handle_report_benchmark_artifacts(args))
        if getattr(args, "report_command", None) == "frontier":
            raise SystemExit(_handle_report_frontier(args))
        if getattr(args, "report_command", None) == "estimator":
            raise SystemExit(_handle_report_estimator(args))
        parser.error("report requires a subcommand")
        return 2

    if args.command == "validate":
        if getattr(args, "validate_command", None) == "results":
            return _handle_validate_results(args)
        if getattr(args, "validate_command", None) == "correctness":
            return _handle_validate_correctness(args)
        parser.error("validate requires a subcommand")
        return 2

    if args.command == "ecosystem":
        ecosystem_command = getattr(args, "ecosystem_command", None)
        if ecosystem_command == "init":
            from fpgai.ecosystem import scaffold_contribution
            root = scaffold_contribution(
                args.out, asset_type=args.asset_type, package_id=args.package_id,
                name=args.name, language=args.language, force=args.force,
            )
            print(str(root))
            return
        if ecosystem_command == "validate":
            from fpgai.contracts.package_validation import validate_package_manifest
            result = validate_package_manifest(args.path)
            if args.json_output:
                print(result.to_json())
            else:
                print(f"{result.status}: {result.package_id or args.path}")
                for issue in (*result.errors, *result.warnings):
                    print(f"{issue.severity}: {issue.code} {issue.field}: {issue.message}")
            raise SystemExit(0 if result.ok else 1)
        if ecosystem_command == "inventory":
            from fpgai.registries.builtin_catalogue import build_builtin_catalogue
            from fpgai.registries.registry_inventory import inventory_payload
            payload = inventory_payload(build_builtin_catalogue())
            if args.json_output:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                for item in payload.get("entries", []):
                    print(f"{item.get('asset_type',''):16} {item.get('package_id',''):44} {item.get('version','')}")
            return
        if ecosystem_command == "discover":
            from fpgai.discovery import DiscoveryRequest, discover_packages
            result = discover_packages(DiscoveryRequest(
                project_root=args.project_root,
                configured_directories=tuple(args.directory or ()),
                include_builtin=True,
                max_depth=args.max_depth,
                strict=args.strict,
            ))
            payload = result.to_dict()
            if args.json_output:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                summary = payload["summary"]
                print(f"accepted={summary['accepted']} quarantined={summary['quarantined']} conflicts={summary['conflicts']}")
                for item in payload["packages"]:
                    print(f"{item['status']:12} {item['asset_type']:16} {item['package_id']}@{item['version']}")
            raise SystemExit(0 if result.ok else 1)
        if ecosystem_command == "types":
            from fpgai.ecosystem import supported_contribution_types
            for asset_type in supported_contribution_types():
                print(asset_type)
            return
        parser.error("ecosystem requires 'init', 'validate', 'inventory', 'discover', or 'types'")

    if args.command == "frontend":
        from fpgai.frontend import frontend_registry, import_model_source
        if getattr(args, "frontend_command", None) == "list":
            for name, spec in sorted(frontend_registry().items()):
                print(f"{name:16} {spec.provider:16} {spec.description}")
            return
        if getattr(args, "frontend_command", None) == "routes":
            from fpgai.frontend import source_framework_route
            for framework in ("jax", "tensorflow", "pytorch", "onnx"):
                route = source_framework_route(framework)
                accepted = ",".join(route.get("accepted_formats", ())) or "-"
                print(f"{framework:12} {route.get('maturity','-'):32} {accepted:24} {route.get('preferred_dialect','-')}")
            return
        if getattr(args, "frontend_command", None) == "import":
            from fpgai.frontend.mlir import canonical_ir_equivalence_manifest, mlir_bridge_manifest, write_fpgai_mlir
            from fpgai.layers.composites import expand_composite_layers
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            graph = import_model_source(
                args.input,
                format_hint=args.format_hint,
                source_framework=args.framework,
                pipeline_mode=args.pipeline_mode,
                target_board=args.target_board,
            )
            graph = expand_composite_layers(graph)
            manifest = mlir_bridge_manifest(graph)
            ir_path = out_dir / "fpgai_ir.json"
            ir_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            mlir_path = write_fpgai_mlir(graph, out_dir / "fpgai_bridge.mlir")
            result = {
                "schema": "fpgai.frontend-import-result/v1",
                "status": "passed",
                "source": str(args.input),
                "format": graph.metadata.get("source", {}).get("format"),
                "framework": args.framework,
                "operator_count": len(graph.ops),
                "tensor_count": len(graph.tensors),
                "ingress_route": graph.metadata.get("source", {}).get("ingress_route"),
                "canonical_structure_fingerprint_sha256": canonical_ir_equivalence_manifest(graph)["fingerprint_sha256"],
                "canonical_parameter_fingerprint_sha256": canonical_ir_equivalence_manifest(graph, include_parameter_values=True)["fingerprint_sha256"],
                "fpgai_ir": str(ir_path),
                "fpgai_mlir": str(mlir_path),
            }
            result_path = out_dir / "frontend_import_result.json"
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        parser.error("frontend requires 'list', 'routes', or 'import'")

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
                out=args.out,
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
        if args.experiment_command == "run":
            raise SystemExit(
                run_experiment_from_config(
                    args.config,
                    out_dir=args.out,
                    run_sweep_callable=run_sweep_from_config,
                    max_design_points=args.max_design_points,
                    timeout_sec=args.timeout_sec,
                    dry_run=args.dry_run,
                    repo_root=args.repo_root,
                )
            )
        parser.error("experiment requires a subcommand, e.g. 'inspect' or 'run'")

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
