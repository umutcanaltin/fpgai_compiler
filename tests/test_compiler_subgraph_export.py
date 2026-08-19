import json
from pathlib import Path

from fpgai.config.loader import FPGAIConfig, ModelCfg, OperatorsCfg, PipelineCfg
from fpgai.engine.compiler import Compiler
from fpgai.ir.graph import Graph


def test_compiler_exports_selected_ir_block_as_hls_without_running_tools(tmp_path: Path, monkeypatch):
    graph = Graph("demo")
    graph.add_tensor("x", (1, 4))
    graph.add_tensor("a", (1, 4))
    graph.add_tensor("b", (1, 4))
    graph.inputs = ["x"]
    graph.outputs = ["b"]
    graph.add_op("Relu", ["x"], ["a"], name="relu0")
    graph.add_op("Relu", ["a"], ["b"], name="relu1")

    raw = {
        "version": 1,
        "model": {"path": str(tmp_path / "placeholder.onnx")},
        "pipeline": {"mode": "inference", "outputs": {"top_kernel_name": "deeplearn"}},
        "operators": {"supported": ["Relu"]},
        "numerics": {"kind": "float"},
        "targets": {"platform": {"board": "kv260", "part": "xck26-sfvc784-2LV-c", "clocks": [{"name": "pl_clk0", "target_mhz": 200}]}},
        "memory": {"weights": {"mode": "embedded"}},
        "build": {"stages": {}},
    }
    cfg = FPGAIConfig(
        1,
        ModelCfg(raw["model"]["path"]),
        PipelineCfg("inference"),
        OperatorsCfg(["Relu"]),
        raw,
    )
    compiler = Compiler(cfg)
    monkeypatch.setattr(compiler, "_import_and_prepare_graph", lambda **_: graph)

    out = compiler.export_subgraph(op_names=["relu0"], out_dir=tmp_path / "export", artifact_format="hls")
    manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads((out / "ir/resolved_ir.json").read_text(encoding="utf-8"))
    assert manifest["selected_ops"] == ["relu0"]
    assert manifest["subgraph_inputs"] == ["x"]
    assert manifest["subgraph_outputs"] == ["a"]
    assert manifest["tool_execution"] == {"vitis_hls": False, "vivado": False, "bitstream": False}
    assert (out / "hls/src/deeplearn.cpp").is_file()
    assert [op["name"] for op in resolved["operators"]] == ["relu0"]


def test_compiler_exports_selected_external_operator_as_vhdl_without_running_vivado(tmp_path: Path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    graph = Graph("external_demo")
    graph.add_tensor("x", (1,))
    graph.add_tensor("y", (1,))
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    op = graph.add_op("ScaleBias", ["x"], ["y"], name="scale0")
    op.attrs["_fpgai_external_operator"] = {
        "operator_id": "community.operator.scale_bias",
        "operator_semantics_version": 1,
    }

    raw = {
        "version": 1,
        "model": {"path": str(tmp_path / "placeholder.onnx")},
        "pipeline": {"mode": "inference", "outputs": {"top_kernel_name": "deeplearn"}},
        "operators": {"supported": ["ScaleBias"]},
        "numerics": {"kind": "float"},
        "targets": {"platform": {"board": "kv260", "part": "xck26-sfvc784-2LV-c", "clocks": [{"name": "pl_clk0", "target_mhz": 200}]}},
        "memory": {"weights": {"mode": "embedded"}},
        "ecosystem": {
            "enabled": True,
            "project_root": str(repo),
            "package_directories": [str(repo / "examples/packages")],
            "strict_discovery": True,
        },
        "implementations": {
            "enable": ["community.scale_bias_vhdl"],
            "operators": {
                "community.operator.scale_bias": {
                    "backend": "vhdl",
                    "preferred": ["community.scale_bias_vhdl"],
                    "allow_fallback": False,
                }
            },
        },
        "build": {"stages": {}},
    }
    cfg = FPGAIConfig(
        1,
        ModelCfg(raw["model"]["path"]),
        PipelineCfg("inference"),
        OperatorsCfg(["ScaleBias"]),
        raw,
    )
    compiler = Compiler(cfg)
    monkeypatch.setattr(compiler, "_import_and_prepare_graph", lambda **_: graph)

    out = compiler.export_subgraph(op_names=["scale0"], out_dir=tmp_path / "export_vhdl", artifact_format="vhdl")
    manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "vhdl"
    assert manifest["selected_implementation"]["package_id"] == "community.scale_bias_vhdl"
    assert manifest["tool_execution"] == {"vitis_hls": False, "vivado": False, "bitstream": False}
    assert manifest["hls_dir"] is None
    assert (out / "vhdl/rtl/000_scale_bias_vhdl.vhd").is_file()
    assert (out / "vhdl/rtl/fpgai_export_scale0.vhd").is_file()
    assert (out / "vhdl/run_vivado.tcl").is_file()


def test_compiler_vhdl_export_rejects_multi_operator_subgraph_explicitly(tmp_path: Path, monkeypatch):
    graph = Graph("demo")
    for name in ("x", "a", "b"):
        graph.add_tensor(name, (1, 4))
    graph.inputs = ["x"]
    graph.outputs = ["b"]
    graph.add_op("Relu", ["x"], ["a"], name="relu0")
    graph.add_op("Relu", ["a"], ["b"], name="relu1")
    raw = {
        "version": 1,
        "model": {"path": str(tmp_path / "placeholder.onnx")},
        "pipeline": {"mode": "inference"},
        "operators": {"supported": ["Relu"]},
        "numerics": {"kind": "float"},
        "targets": {"platform": {"board": "kv260", "part": "xck26-sfvc784-2LV-c", "clocks": [{"name": "pl_clk0", "target_mhz": 200}]}},
        "memory": {"weights": {"mode": "embedded"}},
    }
    cfg = FPGAIConfig(1, ModelCfg(raw["model"]["path"]), PipelineCfg("inference"), OperatorsCfg(["Relu"]), raw)
    compiler = Compiler(cfg)
    monkeypatch.setattr(compiler, "_import_and_prepare_graph", lambda **_: graph)
    import pytest
    with pytest.raises(RuntimeError, match="EXPORT013"):
        compiler.export_subgraph(op_names=["relu0", "relu1"], out_dir=tmp_path / "bad_vhdl", artifact_format="vhdl")


def test_compiler_exports_selected_external_operator_as_hls_without_running_vitis(tmp_path: Path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    graph = Graph("external_hls_demo")
    graph.add_tensor("x", (1,))
    graph.add_tensor("y", (1,))
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    op = graph.add_op("ScaleBias", ["x"], ["y"], name="scale0", attrs={"scale": 2.0, "bias": 1.0})
    op.attrs["_fpgai_external_operator"] = {
        "operator_id": "community.operator.scale_bias",
        "operator_semantics_version": 1,
    }

    raw = {
        "version": 1,
        "model": {"path": str(tmp_path / "placeholder.onnx")},
        "pipeline": {"mode": "inference", "outputs": {"top_kernel_name": "deeplearn"}},
        "operators": {"supported": ["ScaleBias"]},
        "numerics": {"kind": "float"},
        "targets": {"platform": {"board": "kv260", "part": "xck26-sfvc784-2LV-c", "clocks": [{"name": "pl_clk0", "target_mhz": 200}]}},
        "memory": {"weights": {"mode": "embedded"}},
        "ecosystem": {
            "enabled": True,
            "project_root": str(repo),
            "package_directories": [str(repo / "examples/packages")],
            "strict_discovery": True,
        },
        "implementations": {
            "enable": ["community.scale_bias_hls"],
            "operators": {
                "community.operator.scale_bias": {
                    "backend": "hls",
                    "preferred": ["community.scale_bias_hls"],
                    "allow_fallback": False,
                }
            },
        },
        "build": {"stages": {}},
    }
    cfg = FPGAIConfig(1, ModelCfg(raw["model"]["path"]), PipelineCfg("inference"), OperatorsCfg(["ScaleBias"]), raw)
    compiler = Compiler(cfg)
    monkeypatch.setattr(compiler, "_import_and_prepare_graph", lambda **_: graph)

    out = compiler.export_subgraph(op_names=["scale0"], out_dir=tmp_path / "export_hls_external", artifact_format="hls")
    manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "hls_cpp"
    assert manifest["selected_implementation"]["package_id"] == "community.scale_bias_hls"
    assert manifest["tool_execution"] == {"vitis_hls": False, "vivado": False, "bitstream": False}
    assert (out / "hls/src/fpgai_export_scale0.cpp").is_file()
    assert (out / "hls/external/community_scale_bias_hls/src/000_scale_bias.cpp").is_file()
    source = (out / "hls/src/fpgai_export_scale0.cpp").read_text(encoding="utf-8")
    assert "scale_bias_hls(input, output, 1, 2.0f, 1.0f);" in source
