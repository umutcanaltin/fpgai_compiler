from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fpgai.analysis.ir_architecture import analyze_ir_architecture
from fpgai.frontend.mlir import export_fpgai_mlir, import_fpgai_mlir, mlir_bridge_manifest
from fpgai.ir import Graph, annotate_default_hardware_semantics
from fpgai.ir.passes.infer_shapes import infer_shapes


def _attention_graph() -> Graph:
    g = Graph("tiny_attention")
    g.inputs = ["x"]
    g.outputs = ["norm_out"]
    g.add_tensor("x", (1, 4, 8))
    for name in ("wq", "wk", "wv"):
        g.add_tensor(name, (8, 8))
        g.constants[name] = np.eye(8, dtype=np.float32)
    g.add_tensor("scale", ())
    g.constants["scale"] = np.asarray(0.35355339, dtype=np.float32)
    g.add_tensor("gamma", (8,))
    g.add_tensor("beta", (8,))
    g.constants["gamma"] = np.ones((8,), dtype=np.float32)
    g.constants["beta"] = np.zeros((8,), dtype=np.float32)

    g.add_op("MatMul", ["x", "wq"], ["q"], name="q_proj")
    g.add_op("MatMul", ["x", "wk"], ["k"], name="k_proj")
    g.add_op("MatMul", ["x", "wv"], ["v"], name="v_proj")
    g.add_op("Transpose", ["k"], ["kt"], name="k_transpose", attrs={"perm": [0, 2, 1]})
    g.add_op("MatMul", ["q", "kt"], ["scores"], name="score_matmul")
    g.add_op("Mul", ["scores", "scale"], ["scaled_scores"], name="score_scale")
    g.add_op("Softmax", ["scaled_scores"], ["probs"], name="softmax", attrs={"axis": -1})
    g.add_op("MatMul", ["probs", "v"], ["context"], name="attention_value")
    g.add_op("Add", ["context", "x"], ["residual"], name="residual")
    g.add_op("LayerNormalization", ["residual", "gamma", "beta"], ["norm_out"], name="norm", attrs={"axis": -1})
    return infer_shapes(g)


def test_attention_shape_inference_and_semantic_annotations() -> None:
    g = _attention_graph()
    assert g.get_tensor("q").shape == (1, 4, 8)
    assert g.get_tensor("kt").shape == (1, 8, 4)
    assert g.get_tensor("scores").shape == (1, 4, 4)
    assert g.get_tensor("probs").shape == (1, 4, 4)
    assert g.get_tensor("context").shape == (1, 4, 8)
    assert g.get_tensor("norm_out").shape == (1, 4, 8)

    annotate_default_hardware_semantics(g, pipeline_mode="training_on_device", target_board="kv260")
    assert g.schema == "fpgai.ir/v2"
    assert g.semantics.target_board == "kv260"
    assert g.tensors["wq"].semantics.training.role == "parameter"
    assert g.tensors["wq"].semantics.training.requires_gradient is True
    assert g.tensors["x"].semantics.training.role == "input"


def test_mlir_bridge_round_trips_attention_ir_semantics() -> None:
    g = _attention_graph()
    annotate_default_hardware_semantics(g, pipeline_mode="inference")
    g.tensors["scores"].semantics.transport.protocol = "ready_valid"
    g.tensors["scores"].semantics.transport.ready_valid = True
    g.ops[4].semantics.selected_backend = "vhdl"
    g.semantics.ir_level = "architectural"
    g.semantics.execution["model"] = {"mode": "pipeline"}
    g.semantics.provenance["framework"] = "jax"
    g.ops[4].semantics.execution["layer"] = {"pe": 4, "simd": 2}
    g.ops[4].semantics.provenance["mlir_op"] = "stablehlo.dot_general"

    text = export_fpgai_mlir(g)
    assert 'module attributes {fpgai.schema = "fpgai.mlir-bridge/v1"}' in text
    assert '"fpgai.op"' in text
    assert '"MatMul"' in text
    assert '"LayerNormalization"' in text

    restored = import_fpgai_mlir(text)
    assert restored.name == g.name
    assert restored.semantics.source_ir == "mlir"
    assert restored.get_tensor("scores").shape == (1, 4, 4)
    assert restored.get_tensor("scores").semantics.transport.protocol == "ready_valid"
    assert restored.ops[4].semantics.selected_backend == "vhdl"
    assert restored.semantics.ir_level == "architectural"
    assert restored.semantics.execution["model"]["mode"] == "pipeline"
    assert restored.semantics.provenance["framework"] == "jax"
    assert restored.ops[4].semantics.execution["layer"]["pe"] == 4
    assert restored.ops[4].semantics.provenance["mlir_op"] == "stablehlo.dot_general"
    assert [op.op_type for op in restored.ops] == [op.op_type for op in g.ops]


def test_mlir_manifest_and_ir_architecture_report_distinguish_roles() -> None:
    g = _attention_graph()
    annotate_default_hardware_semantics(g)
    manifest = mlir_bridge_manifest(g)
    assert manifest["schema"] == "fpgai.mlir-bridge/v1"
    assert manifest["ir_schema"] == "fpgai.ir/v2"

    report = analyze_ir_architecture(g)
    assert report["representations"]["onnx"]["role"].startswith("portable model")
    assert report["representations"]["mlir"]["bridge_schema"] == "fpgai.mlir-bridge/v1"
    assert report["representations"]["fpgai_ir"]["owned_by_fpgai"] is True
    assert report["scientific_positioning"]["mlir_replacement_claim"] is False
    assert report["attention_operator_inventory"]["MatMul"] == 5


def test_compile_plan_materializes_hardware_architecture_into_authoritative_ir() -> None:
    from fpgai.engine.models import (
        ArchitecturePlan, BufferingPlan, CompilePlan, LayerMemoryPlan, LayerPlan,
        ParallelismPlan, PartitionPlan, PipelinePlan, PrecisionPlan, TilingPlan,
    )
    from fpgai.ir.passes.mechanism_resolution import materialize_compile_plan_semantics

    g = Graph("resolved_arch")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor("x", (1, 8))
    g.add_tensor("w", (8, 4))
    g.add_tensor("y", (1, 4))
    g.constants["w"] = np.ones((8, 4), dtype=np.float32)
    g.add_op("MatMul", ["x", "w"], ["y"], name="mm")
    arch = ArchitecturePlan(
        precision=PrecisionPlan(mode="fixed"),
        pipeline=PipelinePlan(ii=2, loops={"k": 3}),
        parallelism=ParallelismPlan(pe=4, simd=2, unroll={"n": 4, "k": 2}),
        partitioning=PartitionPlan(factor=2, mode="cyclic", targets={"weights": 2}),
        tiling=TilingPlan(sizes={"m": 1, "n": 4, "k": 2}),
        buffering=BufferingPlan(mode="double"),
        memory=LayerMemoryPlan(weight_mode="embedded", activation_mode="buffer", weight_region="BRAM"),
    )
    lp = LayerPlan(node_name="mm", op_type="MatMul", architecture=arch)
    cp = CompilePlan(target_board="kv260", target_part="part", clock_mhz=200.0, execution_order=["mm"], layer_plans=[lp])
    report = materialize_compile_plan_semantics(g, cp)
    sem = g.ops[0].semantics
    assert report["materialized_operator_count"] == 1
    assert sem.schedule["pipeline"]["ii"] == 2
    assert sem.schedule["pipeline"]["loops"]["k"] == 3
    assert sem.schedule["parallelism"]["pe"] == 4
    assert sem.schedule["parallelism"]["simd"] == 2
    assert sem.schedule["partitioning"]["mode"] == "cyclic"
    assert sem.schedule["tiling"]["sizes"]["n"] == 4
    assert sem.buffering["mode"] == "double"
    assert sem.resource_constraints["resolved_memory"]["weight_region"] == "BRAM"
    assert g.semantics.resource_constraints["resolved_architecture"]["architecture_signature"] == cp.architecture_signature


def test_resolved_ir_snapshot_is_deterministic_and_covers_scientific_semantics() -> None:
    from fpgai.analysis.ir_architecture import ir_scientific_capability_matrix, resolved_ir_snapshot
    from fpgai.engine.models import CompilePlan, LayerPlan
    from fpgai.ir.passes.mechanism_resolution import materialize_compile_plan_semantics

    g = Graph("scientific_ir")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor("x", (1, 4))
    g.add_tensor("w", (4, 4))
    g.add_tensor("y", (1, 4))
    g.constants["w"] = np.eye(4, dtype=np.float32)
    g.add_op("MatMul", ["x", "w"], ["y"], name="mm")
    annotate_default_hardware_semantics(g, pipeline_mode="training_on_device", target_board="kv260")
    cp = CompilePlan(target_board="kv260", layer_plans=[LayerPlan(node_name="mm", op_type="MatMul")])
    materialize_compile_plan_semantics(g, cp)
    g.semantics.runtime_contract["sequence"] = {"sequence": [{"command": "run_training"}]}
    a = resolved_ir_snapshot(g, compile_plan=cp, runtime_sequence=g.semantics.runtime_contract["sequence"])
    b = resolved_ir_snapshot(g, compile_plan=cp, runtime_sequence=g.semantics.runtime_contract["sequence"])
    assert a["schema"] == "fpgai.resolved-ir/v1"
    assert a["resolved_ir_fingerprint_sha256"] == b["resolved_ir_fingerprint_sha256"]
    assert a["operators"][0]["semantics"]["schedule"]["architecture"]
    assert a["tensors"]["w"]["semantics"]["training"]["role"] == "parameter"
    assert a["resolved_plans"]["compile"]["architecture_signature"] == cp.architecture_signature
    matrix = ir_scientific_capability_matrix(g)
    assert matrix["dimensions"]["computation"] is True
    assert matrix["dimensions"]["pipeline_schedule"] is True
    assert matrix["dimensions"]["parallelism"] is True
    assert matrix["dimensions"]["training"] is True
    assert matrix["dimensions"]["runtime"] is True


def test_ir_explicitly_records_progressive_level_hierarchy_and_memory_initialization() -> None:
    from fpgai.engine.models import ArchitecturePlan, CompilePlan, LayerMemoryPlan, LayerPlan, ParallelismPlan, PipelinePlan
    from fpgai.ir.passes.mechanism_resolution import materialize_compile_plan_semantics

    g = Graph("hierarchical_ir")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor("x", (1, 4))
    g.add_tensor("w", (4, 4))
    g.add_tensor("y", (1, 4))
    g.constants["w"] = np.eye(4, dtype=np.float32)
    g.add_op("MatMul", ["x", "w"], ["y"], name="mm")
    annotate_default_hardware_semantics(g, pipeline_mode="training_on_device", target_board="kv260")
    assert g.semantics.ir_level == "functional"
    assert g.tensors["w"].semantics.memory.persistence == "model"
    assert g.tensors["w"].semantics.memory.mutable is True

    arch = ArchitecturePlan(
        pipeline=PipelinePlan(ii=1, style="dataflow", scope="layer", loops={"k": 1}),
        parallelism=ParallelismPlan(pe=8, simd=4, unroll={"k": 4}),
        memory=LayerMemoryPlan(weight_mode="embedded", weight_region="URAM"),
    )
    cp = CompilePlan(
        target_board="kv260",
        layer_plans=[LayerPlan(node_name="mm", op_type="MatMul", architecture=arch)],
        notes={"network_execution": {"mode": "pipeline", "pipeline_depth": 2}},
    )
    materialize_compile_plan_semantics(g, cp)
    assert g.semantics.ir_level == "architectural"
    assert g.semantics.execution["model"]["mode"] == "pipeline"
    sem = g.ops[0].semantics
    assert sem.execution["layer"]["pe"] == 8
    assert sem.execution["layer"]["simd"] == 4
    assert sem.execution["loops"]["unroll"]["k"] == 4
    assert sem.execution["loops"]["pipeline_ii"] == 1

    report = analyze_ir_architecture(g)
    assert report["representations"]["fpgai_ir"]["owned_by_fpgai"] is True
    assert "FPGA architecture" in report["representations"]["fpgai_ir"]["role"]
