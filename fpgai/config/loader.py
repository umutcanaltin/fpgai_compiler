from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from fpgai.config.access import get_path


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    message: str


class ConfigError(Exception):
    def __init__(self, issues: List[ConfigIssue]):
        self.issues = issues
        super().__init__(self.__str__())

    def __str__(self) -> str:
        lines = ["FPGAI config validation failed:"]
        for issue in self.issues:
            lines.append(f" - {issue.path}: {issue.message}")
        return "\n".join(lines)


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"Expected mapping node, got {node.id}",
                node.start_mark,
            )

        self.flatten_mapping(node)
        mapping: Dict[str, Any] = {}

        for key_node, value_node in node.value:
            key = self.construct_object(
                key_node,
                deep=deep,
            )

            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc

            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )

            mapping[key] = self.construct_object(
                value_node,
                deep=deep,
            )

        return mapping


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as config_file:
            raw = yaml.load(
                config_file,
                Loader=_UniqueKeyLoader,
            ) or {}
    except OSError as exc:
        raise ConfigError(
            [
                ConfigIssue(
                    "config",
                    f"Could not read file: {exc}",
                )
            ]
        ) from exc
    except yaml.YAMLError as exc:
        mark = getattr(
            exc,
            "problem_mark",
            None,
        )
        location = ""

        if mark is not None:
            location = (
                f" at line {mark.line + 1}, "
                f"column {mark.column + 1}"
            )

        problem = (
            getattr(exc, "problem", None)
            or str(exc)
        )

        raise ConfigError(
            [
                ConfigIssue(
                    "config",
                    f"Invalid YAML{location}: {problem}",
                )
            ]
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            [
                ConfigIssue(
                    "root",
                    "Top-level YAML must be a mapping/dict",
                )
            ]
        )

    return raw


_deep_get = get_path


@dataclass(frozen=True)
class ModelCfg:
    path: str


@dataclass(frozen=True)
class PipelineCfg:
    mode: str


@dataclass(frozen=True)
class OperatorsCfg:
    supported: List[str]


@dataclass(frozen=True)
class FPGAIConfig:
    version: int
    model: ModelCfg
    pipeline: PipelineCfg
    operators: OperatorsCfg
    raw: Dict[str, Any]


PIPELINE_MODES_V1 = {
    "inference",
    "training_on_device",
}

PARALLEL_POLICIES_V1 = {
    "Fit-First",
    "DSP-Saver",
    "BRAM-Saver",
    "Memory-First",
    "Balanced",
    "Throughput-First",
    "Latency-First",
}

TOP_LEVEL_SECTIONS_V1 = {
    "version",
    "project",
    "pipeline",
    "model",
    "targets",
    "operators",
    "numerics",
    "analysis",
    "data_movement",
    "memory",
    "weights",
    "optimization",
    "training",
    "backends",
    "toolchain",
    "benchmark",
    "build",
    "codegen",
    "runtime",
    "communication",
    "debug",
    "metadata",
}

DEFAULT_NUMERIC_ROLES = {
    "activation",
    "weight",
    "bias",
    "accum",
}

TRAINING_NUMERIC_ROLE_ALIASES = {
    "grad": "grad",
    "grad_accum": "grad_accum",
    "master_weight": "master_weight",
    "optimizer_state": "optimizer_state",
    "gradient": "grad",
    "gradient_accum": "grad_accum",
    "weight_master": "master_weight",
}


def _is_valid_precision_spec(
    value: Any,
) -> bool:
    if not isinstance(value, dict):
        return False

    if value.get("type") != "ap_fixed":
        return False

    total_bits = value.get("total_bits")
    int_bits = value.get("int_bits")

    # bool is a subclass of int, so use exact type checks.
    if (
        type(total_bits) is not int
        or type(int_bits) is not int
    ):
        return False

    if total_bits <= 0 or int_bits <= 0:
        return False

    if int_bits > total_bits:
        return False

    return True


def _precision_error_message() -> str:
    return (
        "Expected {type: ap_fixed, total_bits: int, "
        "int_bits: int} with positive values and "
        "int_bits <= total_bits"
    )


def _ap_str(node: Any) -> str:
    if (
        isinstance(node, dict)
        and node.get("type") == "ap_fixed"
    ):
        return (
            f"ap_fixed<"
            f"{int(node.get('total_bits', 16))},"
            f"{int(node.get('int_bits', 6))}>"
        )

    return "float"


def _validate_default_numerics(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    defaults = _deep_get(
        raw,
        "numerics.defaults",
        None,
    )

    if defaults is None:
        return

    if not isinstance(defaults, dict):
        issues.append(
            ConfigIssue(
                "numerics.defaults",
                "Expected a mapping",
            )
        )
        return

    unknown_roles = (
        set(defaults)
        - DEFAULT_NUMERIC_ROLES
    )

    for role in sorted(unknown_roles):
        issues.append(
            ConfigIssue(
                f"numerics.defaults.{role}",
                f"Unknown numeric role {role!r}",
            )
        )

    for role in DEFAULT_NUMERIC_ROLES:
        spec = defaults.get(role)

        if (
            spec is not None
            and not _is_valid_precision_spec(spec)
        ):
            issues.append(
                ConfigIssue(
                    f"numerics.defaults.{role}",
                    _precision_error_message(),
                )
            )


def _validate_layer_match(
    match: Any,
    path: str,
    issues: List[ConfigIssue],
) -> None:
    if not isinstance(match, dict):
        issues.append(
            ConfigIssue(
                path,
                "Missing or invalid match mapping",
            )
        )
        return

    allowed_keys = {
        "name",
        "op_type",
        "index",
    }

    unknown_keys = set(match) - allowed_keys

    for key in sorted(unknown_keys):
        issues.append(
            ConfigIssue(
                f"{path}.{key}",
                f"Unknown match field {key!r}",
            )
        )

    if not any(
        key in match
        for key in allowed_keys
    ):
        issues.append(
            ConfigIssue(
                path,
                "At least one of name, op_type, or "
                "index must be provided",
            )
        )

    if (
        "name" in match
        and (
            not isinstance(match["name"], str)
            or not match["name"].strip()
        )
    ):
        issues.append(
            ConfigIssue(
                f"{path}.name",
                "Expected a non-empty string",
            )
        )

    if (
        "op_type" in match
        and (
            not isinstance(match["op_type"], str)
            or not match["op_type"].strip()
        )
    ):
        issues.append(
            ConfigIssue(
                f"{path}.op_type",
                "Expected a non-empty string",
            )
        )

    if (
        "index" in match
        and type(match["index"]) is not int
    ):
        issues.append(
            ConfigIssue(
                f"{path}.index",
                "Expected an integer",
            )
        )

    if (
        "index" in match
        and type(match["index"]) is int
        and match["index"] < 0
    ):
        issues.append(
            ConfigIssue(
                f"{path}.index",
                "Expected a non-negative integer",
            )
        )


def _validate_layerwise_numerics(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    layer_rules = _deep_get(
        raw,
        "numerics.layers",
        [],
    )

    if layer_rules is None:
        return

    if not isinstance(layer_rules, list):
        issues.append(
            ConfigIssue(
                "numerics.layers",
                "Expected a list of layer precision rules",
            )
        )
        return

    allowed_rule_keys = (
        DEFAULT_NUMERIC_ROLES
        | {"match"}
    )

    for index, rule in enumerate(layer_rules):
        path = f"numerics.layers[{index}]"

        if not isinstance(rule, dict):
            issues.append(
                ConfigIssue(
                    path,
                    "Each entry must be a mapping",
                )
            )
            continue

        unknown_keys = (
            set(rule)
            - allowed_rule_keys
        )

        for key in sorted(unknown_keys):
            issues.append(
                ConfigIssue(
                    f"{path}.{key}",
                    f"Unknown layer precision field {key!r}",
                )
            )

        _validate_layer_match(
            rule.get("match"),
            f"{path}.match",
            issues,
        )

        for role in DEFAULT_NUMERIC_ROLES:
            if (
                role in rule
                and not _is_valid_precision_spec(
                    rule[role]
                )
            ):
                issues.append(
                    ConfigIssue(
                        f"{path}.{role}",
                        _precision_error_message(),
                    )
                )


def _validate_and_normalize_training_numerics(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    numerics = raw.setdefault(
        "numerics",
        {},
    )

    if not isinstance(numerics, dict):
        issues.append(
            ConfigIssue(
                "numerics",
                "Expected a mapping",
            )
        )
        return

    training = numerics.get(
        "training",
        {},
    )

    if training is None:
        numerics["training"] = {}
        return

    if not isinstance(training, dict):
        issues.append(
            ConfigIssue(
                "numerics.training",
                "Expected a mapping",
            )
        )
        return

    normalized: Dict[str, Any] = {}

    for role, spec in training.items():
        if role not in TRAINING_NUMERIC_ROLE_ALIASES:
            issues.append(
                ConfigIssue(
                    f"numerics.training.{role}",
                    f"Unknown training numeric role {role!r}",
                )
            )
            continue

        canonical_role = (
            TRAINING_NUMERIC_ROLE_ALIASES[role]
        )

        if not _is_valid_precision_spec(spec):
            issues.append(
                ConfigIssue(
                    f"numerics.training.{role}",
                    _precision_error_message(),
                )
            )
            continue

        if canonical_role in normalized:
            issues.append(
                ConfigIssue(
                    f"numerics.training.{role}",
                    f"Duplicate alias for training numeric "
                    f"role {canonical_role!r}",
                )
            )
            continue

        normalized[canonical_role] = spec

    numerics["training"] = normalized


def _validate_quantization_report(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    report = _deep_get(
        raw,
        "analysis.quantization_report",
        None,
    )

    if report is None:
        return

    if not isinstance(report, dict):
        issues.append(
            ConfigIssue(
                "analysis.quantization_report",
                "Expected a mapping",
            )
        )
        return

    enabled = report.get(
        "enabled",
        False,
    )

    if not isinstance(enabled, bool):
        issues.append(
            ConfigIssue(
                "analysis.quantization_report.enabled",
                "Expected a boolean",
            )
        )

    seed = report.get(
        "seed",
        0,
    )

    if type(seed) is not int:
        issues.append(
            ConfigIssue(
                "analysis.quantization_report.seed",
                "Expected an integer",
            )
        )

    input_npy = report.get(
        "input_npy",
        None,
    )

    if (
        input_npy is not None
        and not isinstance(input_npy, str)
    ):
        issues.append(
            ConfigIssue(
                "analysis.quantization_report.input_npy",
                "Expected a filesystem path string",
            )
        )


def _validate_precision_sweep_candidate(
    candidate: Any,
    index: int,
    *,
    default_override_mode: str,
    seen_names: set[str],
    issues: List[ConfigIssue],
) -> None:
    path = (
        f"analysis.precision_sweep."
        f"candidates[{index}]"
    )

    if not isinstance(candidate, dict):
        issues.append(
            ConfigIssue(
                path,
                "Each candidate must be a mapping",
            )
        )
        return

    name = candidate.get(
        "name",
        f"candidate_{index}",
    )

    if (
        not isinstance(name, str)
        or not name.strip()
    ):
        issues.append(
            ConfigIssue(
                f"{path}.name",
                "Expected a non-empty string",
            )
        )
        name = f"candidate_{index}"
    else:
        name = name.strip()

    if name in seen_names:
        issues.append(
            ConfigIssue(
                f"{path}.name",
                "Duplicate precision sweep candidate "
                f"name: {name}",
            )
        )

    seen_names.add(name)

    override_mode = candidate.get(
        "layer_overrides",
        default_override_mode,
    )

    if override_mode not in {
        "clear",
        "preserve",
    }:
        issues.append(
            ConfigIssue(
                f"{path}.layer_overrides",
                "Must be one of ['clear', 'preserve']",
            )
        )

    defaults = candidate.get(
        "defaults",
        {},
    )

    if not isinstance(defaults, dict):
        issues.append(
            ConfigIssue(
                f"{path}.defaults",
                "Expected a mapping",
            )
        )
        return

    unknown_roles = (
        set(defaults)
        - DEFAULT_NUMERIC_ROLES
    )

    for role in sorted(unknown_roles):
        issues.append(
            ConfigIssue(
                f"{path}.defaults.{role}",
                f"Unknown numeric role {role!r}",
            )
        )

    for role in DEFAULT_NUMERIC_ROLES:
        spec = defaults.get(role)

        if spec is None:
            issues.append(
                ConfigIssue(
                    f"{path}.defaults.{role}",
                    "Missing precision specification",
                )
            )
        elif not _is_valid_precision_spec(spec):
            issues.append(
                ConfigIssue(
                    f"{path}.defaults.{role}",
                    _precision_error_message(),
                )
            )


def _validate_precision_sweep(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    sweep = _deep_get(
        raw,
        "analysis.precision_sweep",
        None,
    )

    if sweep is None:
        return

    if not isinstance(sweep, dict):
        issues.append(
            ConfigIssue(
                "analysis.precision_sweep",
                "Expected a mapping",
            )
        )
        return

    enabled = sweep.get(
        "enabled",
        False,
    )

    if not isinstance(enabled, bool):
        issues.append(
            ConfigIssue(
                "analysis.precision_sweep.enabled",
                "Expected a boolean",
            )
        )

    override_mode = sweep.get(
        "layer_overrides",
        "clear",
    )

    if override_mode not in {
        "clear",
        "preserve",
    }:
        issues.append(
            ConfigIssue(
                "analysis.precision_sweep.layer_overrides",
                "Must be one of ['clear', 'preserve']",
            )
        )

    require_match = sweep.get(
        "require_prediction_match",
        True,
    )

    if not isinstance(require_match, bool):
        issues.append(
            ConfigIssue(
                "analysis.precision_sweep."
                "require_prediction_match",
                "Expected a boolean",
            )
        )

    minimum_cosine = sweep.get(
        "minimum_cosine",
        0.99,
    )

    if (
        type(minimum_cosine) not in {
            int,
            float,
        }
        or not 0.0 <= float(minimum_cosine) <= 1.0
    ):
        issues.append(
            ConfigIssue(
                "analysis.precision_sweep.minimum_cosine",
                "Expected a number between 0 and 1",
            )
        )

    candidates = sweep.get(
        "candidates",
        [],
    )

    if not isinstance(candidates, list):
        issues.append(
            ConfigIssue(
                "analysis.precision_sweep.candidates",
                "Expected a list",
            )
        )
        return

    if enabled and not candidates:
        issues.append(
            ConfigIssue(
                "analysis.precision_sweep.candidates",
                "Must be a non-empty list when enabled",
            )
        )

    seen_names: set[str] = set()

    for index, candidate in enumerate(candidates):
        _validate_precision_sweep_candidate(
            candidate,
            index,
            default_override_mode=override_mode,
            seen_names=seen_names,
            issues=issues,
        )


def _validate_analysis_cfg(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    analysis = raw.get(
        "analysis",
        {},
    )

    if analysis is None:
        return

    if not isinstance(analysis, dict):
        issues.append(
            ConfigIssue(
                "analysis",
                "Expected a mapping",
            )
        )
        return

    _validate_quantization_report(
        raw,
        issues,
    )
    _validate_precision_sweep(
        raw,
        issues,
    )


def _validate_top_level_sections(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    for key in sorted(set(raw) - TOP_LEVEL_SECTIONS_V1):
        issues.append(
            ConfigIssue(
                key,
                f"Unknown top-level configuration section {key!r}",
            )
        )


def _validate_clock_config(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    clocks = _deep_get(raw, "targets.platform.clocks", None)

    if clocks is None and "targets" not in raw:
        return

    if not isinstance(clocks, list) or not clocks:
        issues.append(
            ConfigIssue(
                "targets.platform.clocks",
                "Expected a non-empty list of clock mappings",
            )
        )
        return

    for index, clock in enumerate(clocks):
        path = f"targets.platform.clocks[{index}]"
        if not isinstance(clock, dict):
            issues.append(ConfigIssue(path, "Expected a mapping"))
            continue

        for key in sorted(set(clock) - {"name", "target_mhz"}):
            issues.append(
                ConfigIssue(
                    f"{path}.{key}",
                    f"Unknown clock field {key!r}",
                )
            )

        name = clock.get("name")
        if not isinstance(name, str) or not name.strip():
            issues.append(
                ConfigIssue(f"{path}.name", "Expected a non-empty string")
            )

        if "target_mhz" in clock:
            target_mhz = clock.get("target_mhz")
            if (
                type(target_mhz) not in {int, float}
                or float(target_mhz) <= 0.0
            ):
                issues.append(
                    ConfigIssue(
                        f"{path}.target_mhz",
                        "Expected a positive number when provided",
                    )
                )


def _validate_fit_policy(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    allowed = {"report_only", "warn", "enforce"}

    for path in ("targets.platform.fit_policy", "hardware.fit_policy"):
        value = _deep_get(raw, path, None)
        if value is None:
            continue

        normalized = str(value).strip().lower()
        if normalized not in allowed:
            issues.append(
                ConfigIssue(
                    path,
                    "Invalid fit_policy. Expected one of: "
                    + ", ".join(sorted(allowed)),
                )
            )


def _validate_parallel_policy(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    requested_paths = (
        "optimization.parallel_policy",
        "analysis.design_space.policy_name",
    )
    requested = [
        (path, _deep_get(raw, path, None))
        for path in requested_paths
        if _deep_get(raw, path, None) is not None
    ]

    for path, value in requested:
        if value not in PARALLEL_POLICIES_V1:
            issues.append(
                ConfigIssue(
                    path,
                    f"Unknown policy {value!r}; expected one of "
                    f"{sorted(PARALLEL_POLICIES_V1)}",
                )
            )

    if len(requested) == 2 and requested[0][1] != requested[1][1]:
        issues.append(
            ConfigIssue(
                "optimization.parallel_policy",
                "Conflicts with analysis.design_space.policy_name",
            )
        )


def _validate_compiler_controls(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    top_name = _deep_get(raw, "pipeline.outputs.top_kernel_name", None)
    if top_name is not None and (
        not isinstance(top_name, str) or not top_name.strip()
    ):
        issues.append(
            ConfigIssue(
                "pipeline.outputs.top_kernel_name",
                "Expected a non-empty string",
            )
        )

    weights_mode = _deep_get(
        raw,
        "data_movement.ps_pl.weights.mode",
        None,
    )
    if weights_mode is not None and weights_mode not in {
        "embedded",
        "stream",
        "ddr",
        "dma_ddr",
    }:
        issues.append(
            ConfigIssue(
                "data_movement.ps_pl.weights.mode",
                "Must be one of ['ddr', 'dma_ddr', 'embedded', 'stream']",
            )
        )

    for path in (
        "backends.hls.enabled",
        "backends.host_cpp.enabled",
        "toolchain.vitis_hls.enabled",
        "optimization.capabilities.strict",
    ):
        value = _deep_get(raw, path, None)
        if value is not None and not isinstance(value, bool):
            issues.append(ConfigIssue(path, "Expected a boolean"))


def load_config(path: str) -> FPGAIConfig:
    if not os.path.exists(path):
        raise ConfigError(
            [
                ConfigIssue(
                    "config",
                    f"File not found: {path}",
                )
            ]
        )

    raw = _load_yaml(path)
    issues: List[ConfigIssue] = []

    _validate_top_level_sections(raw, issues)

    version = raw.get(
        "version",
        1,
    )

    if type(version) is not int:
        issues.append(
            ConfigIssue(
                "version",
                "Must be an integer",
            )
        )
        version = 1

    if version != 1:
        issues.append(
            ConfigIssue(
                "version",
                f"Unsupported version {version}; "
                "only version 1 is supported",
            )
        )

    model_path = _deep_get(
        raw,
        "model.path",
        None,
    )

    if (
        not isinstance(model_path, str)
        or not model_path.strip()
    ):
        issues.append(
            ConfigIssue(
                "model.path",
                "Missing or invalid model.path",
            )
        )
        model_path = ""
    else:
        model_path = model_path.strip()

        if not os.path.exists(model_path):
            issues.append(
                ConfigIssue(
                    "model.path",
                    f"File does not exist: {model_path}",
                )
            )

    mode = _deep_get(
        raw,
        "pipeline.mode",
        "inference",
    )

    if (
        not isinstance(mode, str)
        or mode not in PIPELINE_MODES_V1
    ):
        issues.append(
            ConfigIssue(
                "pipeline.mode",
                f"Must be one of "
                f"{sorted(PIPELINE_MODES_V1)}",
            )
        )
        mode = "inference"

    operators = _deep_get(
        raw,
        "operators.supported",
        None,
    )

    if (
        not isinstance(operators, list)
        or not operators
        or not all(
            isinstance(operator, str)
            and operator.strip()
            for operator in operators
        )
    ):
        issues.append(
            ConfigIssue(
                "operators.supported",
                "Expected a non-empty list of strings",
            )
        )
        operators = []

    operators = [
        operator.strip()
        for operator in operators
    ]

    if len(set(operators)) != len(operators):
        issues.append(
            ConfigIssue(
                "operators.supported",
                "Duplicate operator names are not allowed",
            )
        )

    _validate_default_numerics(
        raw,
        issues,
    )
    _validate_layerwise_numerics(
        raw,
        issues,
    )
    _validate_and_normalize_training_numerics(
        raw,
        issues,
    )
    _validate_analysis_cfg(
        raw,
        issues,
    )
    _validate_clock_config(
        raw,
        issues,
    )
    _validate_fit_policy(
        raw,
        issues,
    )
    _validate_parallel_policy(
        raw,
        issues,
    )
    _validate_compiler_controls(
        raw,
        issues,
    )

    if issues:
        raise ConfigError(issues)

    return FPGAIConfig(
        version=version,
        model=ModelCfg(
            path=model_path,
        ),
        pipeline=PipelineCfg(
            mode=mode,
        ),
        operators=OperatorsCfg(
            supported=operators,
        ),
        raw=raw,
    )


def print_summary(cfg: FPGAIConfig) -> None:
    raw = cfg.raw

    def get(
        path: str,
        default: Any = None,
    ) -> Any:
        return _deep_get(
            raw,
            path,
            default,
        )

    board = get(
        "targets.platform.board",
        "kv260",
    )
    part = get(
        "targets.platform.part",
        "xck26-sfvc784-2LV-c",
    )
    clock = get(
        "targets.platform.clocks.0.target_mhz",
        200,
    )

    activation = get(
        "numerics.defaults.activation",
        {},
    )
    weight = get(
        "numerics.defaults.weight",
        {},
    )
    bias = get(
        "numerics.defaults.bias",
        {},
    )
    accum = get(
        "numerics.defaults.accum",
        {},
    )

    layer_rules = get(
        "numerics.layers",
        [],
    ) or []
    training_rules = get(
        "numerics.training",
        {},
    ) or {}

    compression = bool(
        get(
            "data_movement.ps_pl."
            "compression.enabled",
            False,
        )
    )
    vitis = bool(
        get(
            "toolchain.vitis_hls.enabled",
            True,
        )
    )
    vivado = bool(
        get(
            "toolchain.vivado.enabled",
            True,
        )
    )
    verbose = bool(
        get(
            "debug.verbose",
            False,
        )
    )
    quant_enabled = bool(
        get(
            "analysis.quantization_report.enabled",
            False,
        )
    )
    sweep_enabled = bool(
        get(
            "analysis.precision_sweep.enabled",
            False,
        )
    )
    sweep_override_mode = get(
        "analysis.precision_sweep."
        "layer_overrides",
        "clear",
    )

    print(
        "\n================ FPGAI Config Summary "
        "================"
    )
    print(f"Config version        : {cfg.version}")
    print(f"Model path            : {cfg.model.path}")
    print(f"Pipeline mode         : {cfg.pipeline.mode}")
    print(
        "------------------------------------------------------"
    )
    print(f"Target board          : {board}")
    print(f"Target part           : {part}")
    print(f"Target clock (MHz)    : {clock}")
    print(
        "------------------------------------------------------"
    )
    print("Precision kind        : fixed")
    print(f" activation           : {_ap_str(activation)}")
    print(f" weight               : {_ap_str(weight)}")
    print(f" bias                 : {_ap_str(bias)}")
    print(f" accum                : {_ap_str(accum)}")
    print(f"Layerwise overrides   : {len(layer_rules)}")
    print(
        f"Training numerics     : "
        f"{sorted(training_rules.keys())}"
    )
    print(
        "------------------------------------------------------"
    )
    print("Operator allowlist    :")

    for operator in cfg.operators.supported:
        print(f" - {operator}")

    print(
        "------------------------------------------------------"
    )
    print(f"Compression enabled   : {compression}")
    print(f"Quant report enabled  : {quant_enabled}")
    print(f"Precision sweep       : {sweep_enabled}")
    print(f"Sweep layer overrides : {sweep_override_mode}")
    print(
        "------------------------------------------------------"
    )
    print(f"Toolchain.vitis_hls   : {vitis}")
    print(f"Toolchain.vivado      : {vivado}")
    print(
        "------------------------------------------------------"
    )
    print(f"Debug.verbose         : {verbose}")
    print(
        "======================================================\n"
    )
