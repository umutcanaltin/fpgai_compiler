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
