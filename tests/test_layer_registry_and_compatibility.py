from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from fpgai.analysis.model_compatibility import (
    build_layer_knob_contract,
    build_model_compatibility_report,
)
from fpgai.analysis.model_inspection import inspect_graph
from fpgai.layers.registry import get_layer_capability, layer_registry, supported_layer_types


class _Tensor:
    def __init__(self, shape=(1,), dtype="float32"):
        self.shape = shape
        self.dtype = dtype


class _Op:
    def __init__(self, name: str, op_type: str):
        self.name = name
        self.op_type = op_type
        self.inputs = [f"{name}_in"]
        self.outputs = [f"{name}_out"]


class _Graph:
    name = "compat"
    inputs = ["x"]
    outputs = ["y"]
    constants = {}

    def __init__(self, ops):
        self.ops = ops

    def get_tensor(self, name: str):
        return _Tensor((1, 4), "float32")


def _load_inference_config() -> dict:
    for p in [Path("configs/examples/inference_compile.yml"), Path("configs/examples/default_compile.yml")]:
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    pytest.skip("inference config not available")


def _make_config(raw: dict, tmp_path: Path):
    cfg_path = tmp_path / "compile.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    from fpgai.config.loader import load_config

    return load_config(str(cfg_path))


def test_layer_registry_reports_supported_and_unsupported_layer_contracts() -> None:
    registry = layer_registry(pipeline_mode="inference")

    assert registry["Dense"]["inference"]["supported"] is True
    assert registry["Conv"]["inference"]["status"] in {"supported", "limited"}
    assert registry["MaxPool"]["has_weights"] is False
    assert registry["MaxPool"]["knobs"]["weight_storage"] == "not_applicable_no_weight_tensors"

    custom = get_layer_capability("NonMaxSuppression", pipeline_mode="inference")
    assert custom.inference_supported is False
    assert "No HLS emitter" in custom.inference_detail

    assert "Dense" in supported_layer_types(pipeline_mode="inference")


def test_model_compatibility_scanner_is_honest_about_unsupported_ops() -> None:
    graph = _Graph([_Op("dense0", "Dense"), _Op("nms0", "NonMaxSuppression")])
    inspection = inspect_graph(
        graph,
        model_path="dummy.onnx",
        pipeline_mode="inference",
        allowed_operators=["Dense", "NonMaxSuppression"],
    )

    report = build_model_compatibility_report(inspection)
    assert report["compilation_ready"] is False
    assert "NonMaxSuppression" in report["unsupported_operators"]

    contract = build_layer_knob_contract(report)
    assert contract["all_layers_have_knob_contract"] is True
    assert contract["layers"][0]["knobs"]["precision"] == "applies_to_compute_and_activation_types"


def test_compile_emits_model_compatibility_and_layer_knob_contract(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from fpgai.engine.compiler import Compiler

    raw = copy.deepcopy(_load_inference_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "compat_compile")
    raw.setdefault("build", {})["stages"] = {
        "cpp": True,
        "testbench": True,
        "hls_project": False,
        "hls_synthesis": False,
        "vivado_project": False,
        "vivado_implementation": False,
        "bitstream": False,
        "runtime_package": True,
        "reports": True,
    }

    result = Compiler(_make_config(raw, tmp_path)).compile()
    reports = Path(result.out_dir) / "reports"

    compatibility_json = reports / "model_compatibility.json"
    knob_contract_json = reports / "layer_knob_contract.json"

    assert compatibility_json.exists()
    assert knob_contract_json.exists()

    compatibility = json.loads(compatibility_json.read_text(encoding="utf-8"))
    knob_contract = json.loads(knob_contract_json.read_text(encoding="utf-8"))

    assert compatibility["artifact_kind"] == "model_compatibility"
    assert "operators" in compatibility
    assert knob_contract["artifact_kind"] == "layer_knob_contract"
    assert knob_contract["all_layers_have_knob_contract"] is True
