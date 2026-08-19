from pathlib import Path

from fpgai.engine.compiler import Compiler


_STABLEHLO = r'''
module {
  func.func @main(%a: tensor<1x2xf32>, %b: tensor<2x2xf32>) -> tensor<1x2xf32> {
    %0 = "stablehlo.dot"(%a, %b) : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
    "func.return"(%0): (tensor<1x2xf32>) -> ()
  }
}
'''


def test_compiler_import_path_uses_model_format_not_onnx_identity(tmp_path: Path):
    model = tmp_path / "model.mlir"
    model.write_text(_STABLEHLO, encoding="utf-8")
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(f'''\nversion: 1\nmodel:\n  path: {model}\n  format: stablehlo\n  framework: jax\npipeline:\n  mode: inference\noperators:\n  supported: [MatMul]\n''', encoding="utf-8")
    compiler = Compiler.from_yaml(str(cfg))
    graph = compiler._import_and_prepare_graph(act_kind="none", act_alpha=0.1, act_except_last=True)
    assert [op.op_type for op in graph.ops] == ["MatMul"]
    assert graph.metadata["source"]["format"] == "stablehlo"
    assert graph.metadata["source"]["framework"] == "jax"
