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
    format: str | None = None
    framework: str | None = None


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
    "validation",
    "ecosystem",
    "implementations",
    "architecture",
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



def _validate_quantization_cfg(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    quant = _deep_get(raw, "numerics.quantization", None)
    if quant is None:
        return
    if not isinstance(quant, dict):
        issues.append(ConfigIssue("numerics.quantization", "Expected a mapping"))
        return

    allowed = {"mode", "weights", "activations", "calibration", "qat"}
    for key in sorted(set(quant) - allowed):
        issues.append(ConfigIssue(f"numerics.quantization.{key}", f"Unknown quantization field {key!r}"))

    mode = quant.get("mode", "none")
    if mode not in {"none", "ptq", "qat"}:
        issues.append(ConfigIssue("numerics.quantization.mode", "Must be one of ['none', 'ptq', 'qat']"))

    def validate_tensor_spec(name: str) -> None:
        spec = quant.get(name)
        if spec is None:
            return
        path = f"numerics.quantization.{name}"
        if not isinstance(spec, dict):
            issues.append(ConfigIssue(path, "Expected a mapping"))
            return
        allowed_spec = {"bits", "scheme", "granularity", "signed", "axis", "rounding", "saturation"}
        for key in sorted(set(spec) - allowed_spec):
            issues.append(ConfigIssue(f"{path}.{key}", f"Unknown quantization field {key!r}"))
        bits = spec.get("bits", 8)
        if type(bits) is not int or not 2 <= bits <= 32:
            issues.append(ConfigIssue(f"{path}.bits", "Expected an integer in [2, 32]"))
        if spec.get("scheme", "symmetric") not in {"symmetric", "asymmetric"}:
            issues.append(ConfigIssue(f"{path}.scheme", "Must be one of ['symmetric', 'asymmetric']"))
        granularity = spec.get("granularity", "per_tensor")
        if granularity not in {"per_tensor", "per_channel"}:
            issues.append(ConfigIssue(f"{path}.granularity", "Must be one of ['per_tensor', 'per_channel']"))
        axis = spec.get("axis")
        if granularity == "per_channel" and type(axis) is not int:
            issues.append(ConfigIssue(f"{path}.axis", "Per-channel quantization requires an integer axis"))
        if granularity == "per_tensor" and axis is not None:
            issues.append(ConfigIssue(f"{path}.axis", "Per-tensor quantization must not set axis"))
        if "signed" in spec and not isinstance(spec["signed"], bool):
            issues.append(ConfigIssue(f"{path}.signed", "Expected a boolean"))
        if spec.get("rounding", "nearest") not in {"nearest", "floor", "ceil"}:
            issues.append(ConfigIssue(f"{path}.rounding", "Must be one of ['nearest', 'floor', 'ceil']"))
        if spec.get("saturation", "saturate") not in {"saturate", "wrap"}:
            issues.append(ConfigIssue(f"{path}.saturation", "Must be one of ['saturate', 'wrap']"))

    validate_tensor_spec("weights")
    validate_tensor_spec("activations")

    calibration = quant.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, dict):
            issues.append(ConfigIssue("numerics.quantization.calibration", "Expected a mapping"))
        else:
            allowed_cal = {"method", "percentile", "samples", "dataset", "array_key"}
            for key in sorted(set(calibration) - allowed_cal):
                issues.append(ConfigIssue(f"numerics.quantization.calibration.{key}", f"Unknown calibration field {key!r}"))
            method = calibration.get("method", "min_max")
            if method not in {"min_max", "percentile"}:
                issues.append(ConfigIssue("numerics.quantization.calibration.method", "Must be one of ['min_max', 'percentile']"))
            percentile = calibration.get("percentile", 99.99)
            if method == "percentile" and (type(percentile) not in {int, float} or isinstance(percentile, bool) or not 0.0 < float(percentile) <= 100.0):
                issues.append(ConfigIssue("numerics.quantization.calibration.percentile", "Expected a number in (0, 100]"))
            samples = calibration.get("samples")
            if samples is not None and (type(samples) is not int or samples <= 0):
                issues.append(ConfigIssue("numerics.quantization.calibration.samples", "Expected a positive integer"))
            dataset = calibration.get("dataset")
            if dataset is not None and (not isinstance(dataset, str) or not dataset.strip()):
                issues.append(ConfigIssue("numerics.quantization.calibration.dataset", "Expected a non-empty .npy/.npz path string"))
            array_key = calibration.get("array_key")
            if array_key is not None and (not isinstance(array_key, str) or not array_key.strip()):
                issues.append(ConfigIssue("numerics.quantization.calibration.array_key", "Expected a non-empty string"))

    qat = quant.get("qat")
    if qat is not None:
        if not isinstance(qat, dict):
            issues.append(ConfigIssue("numerics.quantization.qat", "Expected a mapping"))
        else:
            allowed_qat = {"fake_quant", "straight_through_estimator", "freeze_after_updates"}
            for key in sorted(set(qat) - allowed_qat):
                issues.append(ConfigIssue(f"numerics.quantization.qat.{key}", f"Unknown QAT field {key!r}"))
            for key in ("fake_quant", "straight_through_estimator"):
                if key in qat and not isinstance(qat[key], bool):
                    issues.append(ConfigIssue(f"numerics.quantization.qat.{key}", "Expected a boolean"))
            freeze = qat.get("freeze_after_updates")
            if freeze is not None and (type(freeze) is not int or freeze < 0):
                issues.append(ConfigIssue("numerics.quantization.qat.freeze_after_updates", "Expected a non-negative integer"))

    if mode == "ptq" and calibration is None:
        issues.append(ConfigIssue("numerics.quantization.calibration", "PTQ mode requires a calibration mapping"))
    if mode == "qat" and qat is None:
        issues.append(ConfigIssue("numerics.quantization.qat", "QAT mode requires a qat mapping"))

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


def _validate_validation_cfg(
    raw: Dict[str, Any],
    issues: List[ConfigIssue],
) -> None:
    validation = raw.get("validation")
    if validation is None:
        return
    if not isinstance(validation, dict):
        issues.append(ConfigIssue("validation", "Expected a mapping"))
        return

    allowed_validation = {"task", "dataset", "decision_thresholds", "training_validation", "numeric"}
    for key in sorted(set(validation) - allowed_validation):
        issues.append(
            ConfigIssue(
                f"validation.{key}",
                f"Unknown validation field {key!r}",
            )
        )

    numeric = validation.get("numeric")
    if numeric is not None:
        if not isinstance(numeric, dict):
            issues.append(ConfigIssue("validation.numeric", "Expected a mapping"))
        else:
            for key in sorted(set(numeric) - {"probes"}):
                issues.append(ConfigIssue(f"validation.numeric.{key}", f"Unknown numeric validation field {key!r}"))
            probes = numeric.get("probes")
            if probes is not None:
                if not isinstance(probes, dict):
                    issues.append(ConfigIssue("validation.numeric.probes", "Expected a mapping"))
                else:
                    for key in sorted(set(probes) - {"enabled", "selectors", "stages"}):
                        issues.append(ConfigIssue(f"validation.numeric.probes.{key}", f"Unknown probe field {key!r}"))
                    stages = probes.get("stages", [])
                    from fpgai.validation.training_probes import SUPPORTED_PROBE_STAGES
                    if not isinstance(stages, list):
                        issues.append(ConfigIssue("validation.numeric.probes.stages", "Expected a list"))
                    else:
                        for i, stage in enumerate(stages):
                            if str(stage) not in SUPPORTED_PROBE_STAGES:
                                issues.append(ConfigIssue(f"validation.numeric.probes.stages.{i}", f"Unsupported probe stage {stage!r}"))
                    selectors = probes.get("selectors", [])
                    if not isinstance(selectors, list):
                        issues.append(ConfigIssue("validation.numeric.probes.selectors", "Expected a list"))
                    else:
                        for i, selector in enumerate(selectors):
                            if not isinstance(selector, dict):
                                issues.append(ConfigIssue(f"validation.numeric.probes.selectors.{i}", "Expected a mapping")); continue
                            for key in sorted(set(selector)-{"operator","parameter","tensor_index"}):
                                issues.append(ConfigIssue(f"validation.numeric.probes.selectors.{i}.{key}", f"Unknown selector field {key!r}"))
                            if not str(selector.get("operator", "")).strip():
                                issues.append(ConfigIssue(f"validation.numeric.probes.selectors.{i}.operator", "Operator is required"))
                            idx=selector.get("tensor_index")
                            if not isinstance(idx, list) or len(idx)!=2 or any(not isinstance(v,int) or v<0 for v in idx):
                                issues.append(ConfigIssue(f"validation.numeric.probes.selectors.{i}.tensor_index", "Dense weight probe requires two non-negative integer indices"))

    training_validation = validation.get("training_validation")
    if training_validation is not None:
        if not isinstance(training_validation, dict):
            issues.append(ConfigIssue("validation.training_validation", "Expected a mapping"))
        else:
            allowed_training_validation = {"dataset"}
            for key in sorted(set(training_validation) - allowed_training_validation):
                issues.append(
                    ConfigIssue(
                        f"validation.training_validation.{key}",
                        f"Unknown training validation field {key!r}",
                    )
                )
            if "dataset" in training_validation:
                nested_issues: List[ConfigIssue] = []
                _validate_validation_cfg(
                    {"validation": {"dataset": training_validation.get("dataset")}},
                    nested_issues,
                )
                for issue in nested_issues:
                    path = issue.path.replace(
                        "validation.dataset",
                        "validation.training_validation.dataset",
                        1,
                    )
                    issues.append(ConfigIssue(path, issue.message))

    task = validation.get("task")
    if task is not None and task not in {"classification", "regression"}:
        issues.append(
            ConfigIssue(
                "validation.task",
                "Must be one of ['classification', 'regression']",
            )
        )

    dataset = validation.get("dataset")
    if dataset is None:
        return
    if not isinstance(dataset, dict):
        issues.append(ConfigIssue("validation.dataset", "Expected a mapping"))
        return

    allowed_dataset = {
        "source", "path", "inputs", "inputs_key", "labels_key",
        "targets_key", "sample_shape", "dtype", "labels",
        "labels_path", "labels_dtype", "targets", "targets_path",
        "targets_dtype", "sample_selection", "name", "root", "split",
        "download", "preprocessing",
    }
    for key in sorted(set(dataset) - allowed_dataset):
        issues.append(
            ConfigIssue(
                f"validation.dataset.{key}",
                f"Unknown dataset field {key!r}",
            )
        )

    source = dataset.get("source")
    if source is not None:
        normalized = str(source).strip().lower().replace("-", "_")
        if normalized not in {"npy", "npz", "binary", "bin", "raw_binary", "torchvision"}:
            issues.append(
                ConfigIssue(
                    "validation.dataset.source",
                    "Must be one of ['bin', 'binary', 'npy', 'npz', 'raw_binary', 'torchvision']",
                )
            )

    normalized_source = str(source or "").strip().lower().replace("-", "_")
    if normalized_source == "torchvision":
        name = dataset.get("name")
        if name not in {"MNIST", "FashionMNIST"}:
            issues.append(ConfigIssue(
                "validation.dataset.name",
                "Must be one of ['FashionMNIST', 'MNIST']",
            ))
        split = dataset.get("split", "test")
        if split not in {"train", "test"}:
            issues.append(ConfigIssue(
                "validation.dataset.split",
                "Must be one of ['test', 'train']",
            ))
        download = dataset.get("download")
        if download is not None and type(download) is not bool:
            issues.append(ConfigIssue(
                "validation.dataset.download",
                "Expected a boolean",
            ))
        preprocessing = dataset.get("preprocessing")
        if preprocessing is not None:
            if not isinstance(preprocessing, dict):
                issues.append(ConfigIssue("validation.dataset.preprocessing", "Expected a mapping"))
            else:
                allowed_preprocessing = {"normalize", "flatten", "add_channel_dim", "mean", "std"}
                for key in sorted(set(preprocessing) - allowed_preprocessing):
                    issues.append(ConfigIssue(
                        f"validation.dataset.preprocessing.{key}",
                        f"Unknown preprocessing field {key!r}",
                    ))
                for key in ("normalize", "flatten", "add_channel_dim"):
                    value = preprocessing.get(key)
                    if value is not None and type(value) is not bool:
                        issues.append(ConfigIssue(
                            f"validation.dataset.preprocessing.{key}",
                            "Expected a boolean",
                        ))
                std = preprocessing.get("std")
                if std is not None and (not isinstance(std, (int, float)) or isinstance(std, bool) or float(std) == 0.0):
                    issues.append(ConfigIssue(
                        "validation.dataset.preprocessing.std",
                        "Expected a non-zero number",
                    ))
    else:
        dataset_path = dataset.get("path", dataset.get("inputs"))
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            issues.append(
                ConfigIssue(
                    "validation.dataset.path",
                    "Missing or invalid dataset path",
                )
            )

    selection = dataset.get("sample_selection")
    if selection is not None:
        if not isinstance(selection, dict):
            issues.append(
                ConfigIssue(
                    "validation.dataset.sample_selection",
                    "Expected a mapping",
                )
            )
        else:
            for key in sorted(set(selection) - {"offset", "count", "mode", "seed", "per_class_count"}):
                issues.append(
                    ConfigIssue(
                        f"validation.dataset.sample_selection.{key}",
                        f"Unknown sample selection field {key!r}",
                    )
                )
            mode = selection.get("mode")
            if mode is not None and str(mode).strip().lower().replace("-", "_") not in {"first", "random", "balanced", "balanced_per_class"}:
                issues.append(ConfigIssue(
                    "validation.dataset.sample_selection.mode",
                    "Must be one of ['balanced', 'balanced_per_class', 'first', 'random']",
                ))
            seed = selection.get("seed")
            if seed is not None and type(seed) is not int:
                issues.append(ConfigIssue(
                    "validation.dataset.sample_selection.seed",
                    "Expected an integer",
                ))
            per_class_count = selection.get("per_class_count")
            if per_class_count is not None and (type(per_class_count) is not int or per_class_count <= 0):
                issues.append(ConfigIssue(
                    "validation.dataset.sample_selection.per_class_count",
                    "Expected a positive integer",
                ))
            offset = selection.get("offset")
            count = selection.get("count")
            if offset is not None and (type(offset) is not int or offset < 0):
                issues.append(
                    ConfigIssue(
                        "validation.dataset.sample_selection.offset",
                        "Expected a non-negative integer",
                    )
                )
            if count is not None and (type(count) is not int or count <= 0):
                issues.append(
                    ConfigIssue(
                        "validation.dataset.sample_selection.count",
                        "Expected a positive integer",
                    )
                )

    thresholds = validation.get("decision_thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            issues.append(
                ConfigIssue(
                    "validation.decision_thresholds",
                    "Expected a mapping",
                )
            )
        else:
            allowed_thresholds = {
                "min_prediction_agreement",
                "max_accuracy_drop_pct",
            }
            for key in sorted(set(thresholds) - allowed_thresholds):
                issues.append(
                    ConfigIssue(
                        f"validation.decision_thresholds.{key}",
                        f"Unknown decision threshold field {key!r}",
                    )
                )
            agreement = thresholds.get("min_prediction_agreement")
            if agreement is not None and (
                not isinstance(agreement, (int, float))
                or isinstance(agreement, bool)
                or not 0.0 <= float(agreement) <= 1.0
            ):
                issues.append(
                    ConfigIssue(
                        "validation.decision_thresholds.min_prediction_agreement",
                        "Expected a number between 0 and 1",
                    )
                )
            drop = thresholds.get("max_accuracy_drop_pct")
            if drop is not None and (
                not isinstance(drop, (int, float))
                or isinstance(drop, bool)
                or float(drop) < 0.0
            ):
                issues.append(
                    ConfigIssue(
                        "validation.decision_thresholds.max_accuracy_drop_pct",
                        "Expected a non-negative number",
                    )
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
    aliases = {"block_over_limit": "enforce"}

    for path in (
        "targets.platform.fit_policy",
        "hardware.fit_policy",
        "build.fit_policy",
    ):
        value = _deep_get(raw, path, None)
        if value is None:
            continue

        normalized = str(value).strip().lower()
        if normalized in aliases:
            normalized = aliases[normalized]
        if normalized not in allowed:
            issues.append(
                ConfigIssue(
                    path,
                    "Invalid fit_policy. Expected one of: "
                    + ", ".join(sorted(allowed | set(aliases))),
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




def _validate_architecture_config(raw: Dict[str, Any], issues: List[ConfigIssue]) -> None:
    architecture = raw.get("architecture")
    if architecture is None:
        return
    if not isinstance(architecture, dict):
        issues.append(ConfigIssue("architecture", "Expected a mapping"))
        return
    for key in sorted(set(architecture) - {"network", "defaults", "layers"}):
        issues.append(ConfigIssue(f"architecture.{key}", f"Unknown architecture field {key!r}"))

    allowed_sections = {"memory", "implementation", "execution", "transport", "buffering", "pipeline", "parallelism", "partitioning", "tiling"}
    allowed_memory = {"weight_storage", "activation_storage", "gradient_storage", "optimizer_state_storage"}
    allowed_impl = {"backend", "preferred", "allow_fallback", "policy"}
    allowed_execution = {"mode"}
    allowed_transport = {"protocol"}
    allowed_buffering = {"storage"}
    allowed_pipeline = {"ii", "style", "scope", "loops"}
    allowed_parallelism = {"pe", "simd", "unroll"}
    allowed_partitioning = {"factor", "mode", "targets"}
    allowed_tiling = {"sizes"}
    allowed_loop_controls = {
        "element", "in", "out", "ic", "oc", "m", "n", "k", "mac",
        "row", "col", "pair", "head", "d", "reduce", "normalize",
        "score", "softmax_max", "softmax_exp", "softmax_norm", "value",
        "dq_dv", "dk",
    }
    storage_values = {"auto", "unspecified", "bram", "uram", "ddr", "host", "external", "stream", "recompute"}
    backend_values = {"auto", "unspecified", "hls", "vitis_hls", "vhdl", "rtl", "external"}
    execution_values = {"auto", "unspecified", "sequential", "dataflow", "serialized", "phase_shared", "parallel", "streamed"}

    def validate_body(body: Any, path: str, *, allow_match: bool) -> None:
        if not isinstance(body, dict):
            issues.append(ConfigIssue(path, "Expected a mapping"))
            return
        allowed = set(allowed_sections) | ({"match"} if allow_match else set())
        for key in sorted(set(body) - allowed):
            issues.append(ConfigIssue(f"{path}.{key}", f"Unknown layer architecture field {key!r}"))
        if allow_match:
            match = body.get("match")
            if not isinstance(match, dict) or not match:
                issues.append(ConfigIssue(f"{path}.match", "Expected a non-empty mapping"))
            else:
                for key in sorted(set(match) - {"name", "op_type", "index", "provider"}):
                    issues.append(ConfigIssue(f"{path}.match.{key}", f"Unknown match field {key!r}"))
                if "index" in match and (type(match["index"]) is not int or match["index"] < 0):
                    issues.append(ConfigIssue(f"{path}.match.index", "Expected a non-negative integer"))
        section_fields = {
            "memory": allowed_memory, "implementation": allowed_impl, "execution": allowed_execution,
            "transport": allowed_transport, "buffering": allowed_buffering,
            "pipeline": allowed_pipeline, "parallelism": allowed_parallelism,
            "partitioning": allowed_partitioning, "tiling": allowed_tiling,
        }
        for section, fields in section_fields.items():
            value = body.get(section)
            if value is None:
                continue
            if not isinstance(value, dict):
                issues.append(ConfigIssue(f"{path}.{section}", "Expected a mapping"))
                continue
            for key in sorted(set(value) - fields):
                issues.append(ConfigIssue(f"{path}.{section}.{key}", f"Unknown {section} field {key!r}"))
        pipeline = body.get("pipeline") if isinstance(body.get("pipeline"), dict) else {}
        if "ii" in pipeline and (type(pipeline["ii"]) is not int or pipeline["ii"] < 1):
            issues.append(ConfigIssue(f"{path}.pipeline.ii", "Expected a positive integer"))
        if "loops" in pipeline:
            loops = pipeline["loops"]
            if not isinstance(loops, dict):
                issues.append(ConfigIssue(f"{path}.pipeline.loops", "Expected a mapping of loop-name to positive II"))
            else:
                for loop_name, value in loops.items():
                    if not isinstance(loop_name, str) or not loop_name.strip() or type(value) is not int or value < 1:
                        issues.append(ConfigIssue(f"{path}.pipeline.loops", "Loop II entries require non-empty names and positive integer values"))
                        break
                    if loop_name not in allowed_loop_controls:
                        issues.append(ConfigIssue(f"{path}.pipeline.loops.{loop_name}", f"Unknown loop control {loop_name!r}"))
        parallelism = body.get("parallelism") if isinstance(body.get("parallelism"), dict) else {}
        for key in ("pe", "simd"):
            if key in parallelism and (type(parallelism[key]) is not int or parallelism[key] < 1):
                issues.append(ConfigIssue(f"{path}.parallelism.{key}", "Expected a positive integer"))
        if "unroll" in parallelism:
            unroll = parallelism["unroll"]
            if not isinstance(unroll, dict) or any(not isinstance(k, str) or not k.strip() or type(v) is not int or v < 1 for k, v in (unroll.items() if isinstance(unroll, dict) else [])):
                issues.append(ConfigIssue(f"{path}.parallelism.unroll", "Expected a mapping of loop-name to positive unroll factor"))
            elif isinstance(unroll, dict):
                for loop_name in unroll:
                    if loop_name not in allowed_loop_controls:
                        issues.append(ConfigIssue(f"{path}.parallelism.unroll.{loop_name}", f"Unknown loop control {loop_name!r}"))
        partitioning = body.get("partitioning") if isinstance(body.get("partitioning"), dict) else {}
        if "factor" in partitioning and (type(partitioning["factor"]) is not int or partitioning["factor"] < 1):
            issues.append(ConfigIssue(f"{path}.partitioning.factor", "Expected a positive integer"))
        if "mode" in partitioning and str(partitioning["mode"]).lower() not in {"none", "cyclic", "block", "complete"}:
            issues.append(ConfigIssue(f"{path}.partitioning.mode", "Expected one of ['none', 'cyclic', 'block', 'complete']"))
        if "targets" in partitioning:
            targets = partitioning["targets"]
            if not isinstance(targets, dict) or any(type(v) is not int or v < 1 for v in (targets.values() if isinstance(targets, dict) else [])):
                issues.append(ConfigIssue(f"{path}.partitioning.targets", "Expected a mapping of target-name to positive partition factor"))
        tiling = body.get("tiling") if isinstance(body.get("tiling"), dict) else {}
        if "sizes" in tiling:
            sizes = tiling["sizes"]
            if not isinstance(sizes, dict) or any(type(v) is not int or v < 1 for v in (sizes.values() if isinstance(sizes, dict) else [])):
                issues.append(ConfigIssue(f"{path}.tiling.sizes", "Expected a mapping of tile-dimension to positive size"))

        memory = body.get("memory") if isinstance(body.get("memory"), dict) else {}
        for key in allowed_memory:
            if key in memory and str(memory[key]).strip().lower().replace("-", "_") not in storage_values:
                issues.append(ConfigIssue(f"{path}.memory.{key}", f"Unsupported storage value {memory[key]!r}"))
        impl = body.get("implementation") if isinstance(body.get("implementation"), dict) else {}
        if "backend" in impl and str(impl["backend"]).strip().lower().replace("-", "_") not in backend_values:
            issues.append(ConfigIssue(f"{path}.implementation.backend", f"Unsupported backend {impl['backend']!r}"))
        execution = body.get("execution") if isinstance(body.get("execution"), dict) else {}
        if "mode" in execution and str(execution["mode"]).strip().lower().replace("-", "_") not in execution_values:
            issues.append(ConfigIssue(f"{path}.execution.mode", f"Unsupported execution mode {execution['mode']!r}"))

    if "network" in architecture:
        validate_body(architecture.get("network"), "architecture.network", allow_match=False)
    if "defaults" in architecture:
        validate_body(architecture.get("defaults"), "architecture.defaults", allow_match=False)
    layers = architecture.get("layers", [])
    if not isinstance(layers, list):
        issues.append(ConfigIssue("architecture.layers", "Expected a list"))
    else:
        for index, rule in enumerate(layers):
            validate_body(rule, f"architecture.layers[{index}]", allow_match=True)


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
    _validate_architecture_config(raw, issues)

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
    _validate_quantization_cfg(
        raw,
        issues,
    )
    _validate_analysis_cfg(
        raw,
        issues,
    )
    _validate_validation_cfg(
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
            format=(str(_deep_get(raw, "model.format", "")).strip().lower() or None),
            framework=(str(_deep_get(raw, "model.framework", "")).strip().lower() or None),
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
    quant_mode = str(get("numerics.quantization.mode", "none"))
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
    print(f"Quantization mode     : {quant_mode}")
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
