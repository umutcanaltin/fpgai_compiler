from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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
            lines.append(f"  - {it.path}: {it.message}")
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

    # Required for dense-only baseline
    if ("Dense" not in ops) and (("Gemm" in ops) or ("MatMul" in ops)):
        # you moved to canonical ops; encourage Dense
        issues.append(ConfigIssue("operators.supported", "Use canonical op names (include 'Dense')."))

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
    def g(p, d=None): return _deep_get(raw, p, d)

    board = g("targets.platform.board", "kv260")
    part = g("targets.platform.part", "xck26-sfvc784-2LV-c")
    clk = g("targets.platform.clocks.0.target_mhz", 200)

    # numerics summary
    act = g("numerics.defaults.activation", {})
    wgt = g("numerics.defaults.weight", {})
    bias = g("numerics.defaults.bias", {})
    acc = g("numerics.defaults.accum", {})

    def ap(node):
        if isinstance(node, dict) and node.get("type") == "ap_fixed":
            return f"ap_fixed<{int(node.get('total_bits', 16))},{int(node.get('int_bits', 6))}>"
        return "float"

    compression = bool(g("data_movement.ps_pl.compression.enabled", False))
    vitis = bool(g("toolchain.vitis_hls.enabled", True))
    vivado = bool(g("toolchain.vivado.enabled", True))
    verbose = bool(g("debug.verbose", False))

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
    print(f"  activation          : {ap(act)}")
    print(f"  weight              : {ap(wgt)}")
    print(f"  bias                : {ap(bias)}")
    print(f"  accum               : {ap(acc)}")
    print("------------------------------------------------------")
    print("Operator allowlist    :")
    for op in cfg.operators.supported:
        print(f"  - {op}")
    print("------------------------------------------------------")
    print(f"Compression enabled   : {compression}")
    print("------------------------------------------------------")
    print(f"Toolchain.vitis_hls   : {vitis}")
    print(f"Toolchain.vivado      : {vivado}")
    print("------------------------------------------------------")
    print(f"Debug.verbose         : {verbose}")
    print("======================================================\n")
