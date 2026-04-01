from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import os
import yaml


@dataclass
class ConfigIssue:
    path: str
    message: str


class ConfigError(Exception):
    def __init__(self, issues: List[ConfigIssue]):
        self.issues = issues
        super().__init__(self.__str__())

    def __str__(self) -> str:
        lines = ["FPGAI config validation failed:"]
        for it in self.issues:
            lines.append(f" - {it.path}: {it.message}")
        return "\n".join(lines)


def _deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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


PIPELINE_MODES_V1 = {"inference", "training_on_device"}
WEIGHT_MEMORY_MODES = {"embedded", "stream", "ddr"}

_BASE_NUMERIC_KEYS = ["activation", "weight", "bias", "accum"]
_TRAINING_NUMERIC_KEYS = [
    "grad_activation",
    "grad_weight",
    "grad_bias",
    "update_accum",
    "optimizer_state",
    "loss",
]


def _is_valid_precision_spec(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    if x.get("type") != "ap_fixed":
        return False
    tb = x.get("total_bits")
    ib = x.get("int_bits")
    if not isinstance(tb, int) or not isinstance(ib, int):
        return False
    if tb <= 0 or ib <= 0 or ib > tb:
        return False
    return True


def _ap_str(node: Any) -> str:
    if isinstance(node, dict) and node.get("type") == "ap_fixed":
        return f"ap_fixed<{int(node.get('total_bits', 16))},{int(node.get('int_bits', 6))}>"
    return "float"


def _validate_default_numerics(raw: Dict[str, Any], issues: List[ConfigIssue]) -> None:
    for key in _BASE_NUMERIC_KEYS:
        spec = _deep_get(raw, f"numerics.defaults.{key}", None)
        if spec is not None and not _is_valid_precision_spec(spec):
            issues.append(
                ConfigIssue(
                    f"numerics.defaults.{key}",
                    "Expected {type: ap_fixed, total_bits: int, int_bits: int} with int_bits <= total_bits",
                )
            )


def _validate_layerwise_numerics(raw: Dict[str, Any], issues: List[ConfigIssue]) -> None:
    layer_rules = _deep_get(raw, "numerics.layers", [])
    if layer_rules is None:
        return

    if not isinstance(layer_rules, list):
        issues.append(ConfigIssue("numerics.layers", "Expected a list of layer precision rules"))
        return

    for i, rule in enumerate(layer_rules):
        p = f"numerics.layers[{i}]"
        if not isinstance(rule, dict):
            issues.append(ConfigIssue(p, "Each entry must be a mapping"))
            continue

        match = rule.get("match")
        if not isinstance(match, dict):
            issues.append(ConfigIssue(f"{p}.match", "Missing/invalid match mapping"))
        else:
            if not any(k in match for k in ("name", "op_type", "index")):
                issues.append(
                    ConfigIssue(
                        f"{p}.match",
                        "At least one of {name, op_type, index} must be provided",
                    )
                )

        for key in _BASE_NUMERIC_KEYS:
            if key in rule and not _is_valid_precision_spec(rule[key]):
                issues.append(
                    ConfigIssue(
                        f"{p}.{key}",
                        "Expected {type: ap_fixed, total_bits: int, int_bits: int} with int_bits <= total_bits",
                    )
                )


def _validate_analysis_cfg(raw: Dict[str, Any], issues: List[ConfigIssue]) -> None:
    qrep = _deep_get(raw, "analysis.quantization_report", None)
    if qrep is not None and not isinstance(qrep, dict):
        issues.append(ConfigIssue("analysis.quantization_report", "Expected a mapping"))

    psweep = _deep_get(raw, "analysis.precision_sweep", None)
    if psweep is not None and not isinstance(psweep, dict):
        issues.append(ConfigIssue("analysis.precision_sweep", "Expected a mapping"))

    if isinstance(psweep, dict):
        candidates = psweep.get("candidates", [])
        if psweep.get("enabled", False):
            if not isinstance(candidates, list) or not candidates:
                issues.append(
                    ConfigIssue(
                        "analysis.precision_sweep.candidates",
                        "Must be a non-empty list when enabled",
                    )
                )
            else:
                for i, cand in enumerate(candidates):
                    p = f"analysis.precision_sweep.candidates[{i}]"
                    if not isinstance(cand, dict):
                        issues.append(ConfigIssue(p, "Each candidate must be a mapping"))
                        continue
                    defaults = cand.get("defaults", {})
                    if not isinstance(defaults, dict):
                        issues.append(ConfigIssue(f"{p}.defaults", "Must be a mapping"))
                        continue
                    for key in _BASE_NUMERIC_KEYS:
                        spec = defaults.get(key)
                        if spec is None or not _is_valid_precision_spec(spec):
                            issues.append(
                                ConfigIssue(
                                    f"{p}.defaults.{key}",
                                    "Expected valid ap_fixed precision spec",
                                )
                            )


def _validate_weight_mode(raw: Dict[str, Any], issues: List[ConfigIssue]) -> None:
    mode = str(_deep_get(raw, "data_movement.ps_pl.weights.mode", "embedded")).lower()
    if mode not in WEIGHT_MEMORY_MODES:
        issues.append(
            ConfigIssue(
                "data_movement.ps_pl.weights.mode",
                f"Must be one of {sorted(WEIGHT_MEMORY_MODES)}",
            )
        )


def _validate_training_cfg(raw: Dict[str, Any], mode: str, issues: List[ConfigIssue]) -> None:
    training_cfg = _deep_get(raw, "training", None)

    if training_cfg is None:
        if mode == "training_on_device":
            issues.append(
                ConfigIssue(
                    "training",
                    "Missing training section while pipeline.mode=training_on_device",
                )
            )
        return

    if not isinstance(training_cfg, dict):
        issues.append(ConfigIssue("training", "Expected a mapping"))
        return

    optimizer = training_cfg.get("optimizer", {})
    if not isinstance(optimizer, dict):
        issues.append(ConfigIssue("training.optimizer", "Expected a mapping"))
    else:
        opt_type = str(optimizer.get("type", "sgd")).lower()
        if opt_type != "sgd":
            issues.append(ConfigIssue("training.optimizer.type", "Only 'sgd' is supported in the first integrated version"))
        lr = optimizer.get("learning_rate", None)
        if not isinstance(lr, (int, float)) or float(lr) <= 0.0:
            issues.append(ConfigIssue("training.optimizer.learning_rate", "Must be a positive number"))

    loss = training_cfg.get("loss", {})
    if not isinstance(loss, dict):
        issues.append(ConfigIssue("training.loss", "Expected a mapping"))
    else:
        loss_type = str(loss.get("type", "mse")).lower()
        if loss_type not in {"mse", "cross_entropy"}:
            issues.append(ConfigIssue("training.loss.type", "Must be one of ['mse', 'cross_entropy']"))

    execution = training_cfg.get("execution", {})
    if execution is not None and not isinstance(execution, dict):
        issues.append(ConfigIssue("training.execution", "Expected a mapping"))
    elif isinstance(execution, dict):
        batch_size = execution.get("batch_size", 1)
        epochs = execution.get("epochs", 1)
        if not isinstance(batch_size, int) or batch_size <= 0:
            issues.append(ConfigIssue("training.execution.batch_size", "Must be a positive integer"))
        if not isinstance(epochs, int) or epochs <= 0:
            issues.append(ConfigIssue("training.execution.epochs", "Must be a positive integer"))

    dataset = training_cfg.get("dataset", {})
    if dataset is not None and not isinstance(dataset, dict):
        issues.append(ConfigIssue("training.dataset", "Expected a mapping"))

    training_numerics = _deep_get(raw, "numerics.training", None)
    if training_numerics is not None:
        if not isinstance(training_numerics, dict):
            issues.append(ConfigIssue("numerics.training", "Expected a mapping"))
        else:
            for key, spec in training_numerics.items():
                if key not in _TRAINING_NUMERIC_KEYS:
                    issues.append(
                        ConfigIssue(
                            f"numerics.training.{key}",
                            f"Unknown training numeric role '{key}'",
                        )
                    )
                    continue
                if not _is_valid_precision_spec(spec):
                    issues.append(
                        ConfigIssue(
                            f"numerics.training.{key}",
                            "Expected {type: ap_fixed, total_bits: int, int_bits: int} with int_bits <= total_bits",
                        )
                    )


def load_config(path: str) -> FPGAIConfig:
    if not os.path.exists(path):
        raise ConfigError([ConfigIssue("config", f"File not found: {path}")])

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ConfigError([ConfigIssue("root", "Top-level YAML must be a mapping/dict")])

    issues: List[ConfigIssue] = []

    version = raw.get("version", 1)
    if not isinstance(version, int):
        issues.append(ConfigIssue("version", "Must be an integer"))
        version = 1
    if version != 1:
        issues.append(ConfigIssue("version", f"Unsupported version {version} (only 1 supported)"))

    model_path = _deep_get(raw, "model.path", None)
    if not isinstance(model_path, str) or not model_path.strip():
        issues.append(ConfigIssue("model.path", "Missing/invalid model.path"))
        model_path = ""
    else:
        model_path = model_path.strip()
        if not os.path.exists(model_path):
            issues.append(ConfigIssue("model.path", f"File does not exist: {model_path}"))

    mode = _deep_get(raw, "pipeline.mode", "inference")
    if not isinstance(mode, str) or mode not in PIPELINE_MODES_V1:
        issues.append(ConfigIssue("pipeline.mode", f"Must be one of {sorted(PIPELINE_MODES_V1)}"))
        mode = "inference"

    ops = _deep_get(raw, "operators.supported", None)
    if not isinstance(ops, list) or not ops or not all(isinstance(x, str) and x.strip() for x in ops):
        issues.append(ConfigIssue("operators.supported", "Expected non-empty list of strings"))
        ops = []
    ops = [x.strip() for x in ops]

    _validate_default_numerics(raw, issues)
    _validate_layerwise_numerics(raw, issues)
    _validate_analysis_cfg(raw, issues)
    _validate_weight_mode(raw, issues)
    _validate_training_cfg(raw, mode, issues)

    if issues:
        raise ConfigError(issues)

    return FPGAIConfig(
        version=version,
        model=ModelCfg(path=model_path),
        pipeline=PipelineCfg(mode=mode),
        operators=OperatorsCfg(supported=ops),
        raw=raw,
    )


def print_summary(cfg: FPGAIConfig) -> None:
    raw = cfg.raw

    def g(p, d=None):
        return _deep_get(raw, p, d)

    board = g("targets.platform.board", "kv260")
    part = g("targets.platform.part", "xck26-sfvc784-2LV-c")
    clk = g("targets.platform.clocks.0.target_mhz", 200)
    act = g("numerics.defaults.activation", {})
    wgt = g("numerics.defaults.weight", {})
    bias = g("numerics.defaults.bias", {})
    acc = g("numerics.defaults.accum", {})
    layer_rules = g("numerics.layers", []) or []
    compression = bool(g("data_movement.ps_pl.compression.enabled", False))
    weights_mode = str(g("data_movement.ps_pl.weights.mode", "embedded")).lower()
    vitis = bool(g("toolchain.vitis_hls.enabled", True))
    vivado = bool(g("toolchain.vivado.enabled", True))
    verbose = bool(g("debug.verbose", False))
    qrep_enabled = bool(g("analysis.quantization_report.enabled", False))
    psweep_enabled = bool(g("analysis.precision_sweep.enabled", False))

    print("\n================ FPGAI Config Summary ================")
    print(f"Config version        : {cfg.version}")
    print(f"Model path            : {cfg.model.path}")
    print(f"Pipeline mode         : {cfg.pipeline.mode}")
    print("------------------------------------------------------")
    print(f"Target board          : {board}")
    print(f"Target part           : {part}")
    print(f"Target clock (MHz)    : {clk}")
    print("------------------------------------------------------")
    print("Precision kind        : fixed")
    print(f" activation           : {_ap_str(act)}")
    print(f" weight               : {_ap_str(wgt)}")
    print(f" bias                 : {_ap_str(bias)}")
    print(f" accum                : {_ap_str(acc)}")
    print(f"Layerwise overrides   : {len(layer_rules)}")
    training_numerics = g("numerics.training", {}) or {}
    if training_numerics:
        print(" Training numerics    :")
        for k, v in training_numerics.items():
            print(f"  - {k:<16} {_ap_str(v)}")
    print("------------------------------------------------------")
    print("Operator allowlist    :")
    for op in cfg.operators.supported:
        print(f" - {op}")
    print("------------------------------------------------------")
    print(f"Compression enabled   : {compression}")
    print(f"Weights mode          : {weights_mode}")
    print(f"Quant report enabled  : {qrep_enabled}")
    print(f"Precision sweep       : {psweep_enabled}")
    if cfg.pipeline.mode == "training_on_device":
        print("Training mode details :")
        print(f" optimizer            : {g('training.optimizer.type', 'sgd')}")
        print(f" learning_rate        : {g('training.optimizer.learning_rate', 'n/a')}")
        print(f" loss                 : {g('training.loss.type', 'mse')}")
        print(f" batch_size           : {g('training.execution.batch_size', 1)}")
        print(f" epochs               : {g('training.execution.epochs', 1)}")
    print("------------------------------------------------------")
    print(f"Toolchain.vitis_hls   : {vitis}")
    print(f"Toolchain.vivado      : {vivado}")
    print("------------------------------------------------------")
    print(f"Debug.verbose         : {verbose}")
    print("======================================================\n")