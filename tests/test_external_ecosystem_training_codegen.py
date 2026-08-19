from pathlib import Path

from fpgai.backends.hls.codegen import emit_hls_stub
from fpgai.implementations import build_hls_composition_plan, implementation_contract_from_manifest
from fpgai.ir.graph import Graph


def _graph():
    graph = Graph("external_training")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_tensor("x", (1, 4), "float32")
    graph.add_tensor("y", (1, 4), "float32")
    graph.add_op("ScaleBias", ["x"], ["y"], "scale_bias_0", {
        "scale": 2.0,
        "bias": 1.0,
        "_fpgai_external_operator": {
            "operator_id": "community.operator.scale_bias",
            "package_id": "community.scale_bias_operator",
        },
    })
    return graph


def test_external_stateless_training_emits_forward_and_backward_entrypoints(tmp_path):
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    assert contract.training.forward is True
    assert contract.training.backward_input is True
    plan = build_hls_composition_plan(_graph(), selected_contracts={"scale_bias_0": contract})
    project = emit_hls_stub(
        graph=_graph(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "training_on_device",
            "weights_mode": "embedded",
            "training_cfg": {"loss": {"type": "mse"}, "optimizer": {"type": "sgd", "learning_rate": 0.01}},
            "run_csim": False,
            "run_csynth": False,
        },
        external_composition_plan=plan,
    )
    source = project.top_cpp.read_text()
    assert "void scale_bias_hls(" in source
    assert "void scale_bias_backward_input_hls(" in source
    assert "scale_bias_hls(" in source
    assert "scale_bias_backward_input_hls(" in source
    assert "External training backward-input" in source
    assert list((project.hls_dir / "src" / "external").rglob("*scale_bias.cpp"))


def test_external_training_backward_is_numerically_validated_against_operator_semantics(tmp_path):
    import numpy as np
    from fpgai.operators.external import BackwardInputReferenceResult
    from fpgai.validation.mixed_external_hls import run_portable_host_cpp_training_validation

    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    graph = _graph()
    plan = build_hls_composition_plan(graph, selected_contracts={"scale_bias_0": contract})

    class Context:
        @staticmethod
        def backward_input_reference_for(operator_id):
            assert operator_id == "community.operator.scale_bias"
            def reference(ctx):
                scale = float(ctx.attributes.get("scale", 1.0))
                return BackwardInputReferenceResult((np.asarray(ctx.grad_outputs[0], dtype=np.float32) * scale,))
            return reference

    report = run_portable_host_cpp_training_validation(
        graph=graph, composition_plan=plan, external_context=Context(), out_dir=tmp_path
    )
    assert report["status"] == "passed"
    assert report["nodes"]["scale_bias_0"]["status"] == "passed"
