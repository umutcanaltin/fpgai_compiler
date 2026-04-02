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

DEFAULT_NUMERIC_ROLES = {"activation", "weight", "bias", "accum"}

TRAINING_NUMERIC_ROLE_ALIASES = {
    # canonical names
    "grad": "grad",
    "grad_accum": "grad_accum",
    "master_weight": "master_weight",
    "optimizer_state": "optimizer_state",
    # backwards-compatible aliases
    "gradient": "grad",
    "gradient_accum": "grad_accum",
    "weight_master": "master_weight",
}


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
    for key in DEFAULT_NUMERIC_ROLES:
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

        for key in DEFAULT_NUMERIC_ROLES:
            if key in rule and not _is_valid_precision_spec(rule[key]):
                issues.append(
                    ConfigIssue(
                        f"{p}.{key}",
                        "Expected {type: ap_fixed, total_bits: int, int_bits: int} with int_bits <= total_bits",
                    )
                )


def _validate_and_normalize_training_numerics(raw: Dict[str, Any], issues: List[ConfigIssue]) -> None:
    numerics = raw.setdefault("numerics", {})
    training = numerics.get("training", {})

    if training is None:
        numerics["training"] = {}
        return

    if not isinstance(training, dict):
        issues.append(ConfigIssue("numerics.training", "Expected a mapping"))
        return

    normalized: Dict[str, Any] = {}

    for role, spec in training.items():
        if role not in TRAINING_NUMERIC_ROLE_ALIASES:
            issues.append(
                ConfigIssue(
                    f"numerics.training.{role}",
                    f"Unknown training numeric role '{role}'",
                )
            )
            continue

        canonical_role = TRAINING_NUMERIC_ROLE_ALIASES[role]

        if not _is_valid_precision_spec(spec):
            issues.append(
                ConfigIssue(
                    f"numerics.training.{role}",
                    "Expected {type: ap_fixed, total_bits: int, int_bits: int} with int_bits <= total_bits",
                )
            )
            continue

        normalized[canonical_role] = spec

    numerics["training"] = normalized


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

                    for key in DEFAULT_NUMERIC_ROLES:
                        spec = defaults.get(key)
                        if spec is None or not _is_valid_precision_spec(spec):
                            issues.append(
                                ConfigIssue(
                                    f"{p}.defaults.{key}",
                                    "Expected valid ap_fixed precision spec",
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
    _validate_and_normalize_training_numerics(raw, issues)
    _validate_analysis_cfg(raw, issues)

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
    training_rules = g("numerics.training", {}) or {}

    compression = bool(g("data_movement.ps_pl.compression.enabled", False))
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
    print(f"Training numerics     : {sorted(training_rules.keys())}")
    print("------------------------------------------------------")
    print("Operator allowlist    :")
    for op in cfg.operators.supported:
        print(f" - {op}")
    print("------------------------------------------------------")
    print(f"Compression enabled   : {compression}")
    print(f"Quant report enabled  : {qrep_enabled}")
    print(f"Precision sweep       : {psweep_enabled}")
    print("------------------------------------------------------")
    print(f"Toolchain.vitis_hls   : {vitis}")
    print(f"Toolchain.vivado      : {vivado}")
    print("------------------------------------------------------")
    print(f"Debug.verbose         : {verbose}")
    print("======================================================\n")