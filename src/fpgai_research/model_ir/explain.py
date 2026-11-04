"""Auditable stage-by-stage Model IR explanation for the paper and its readers.

One operator and tensor identity is retained from Functional, through
Architectural, to Lowered Model IR and the generated HLS manifest. The
representation intentionally does not invent implementation cycle counts.
"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256

from fpgai_research.architecture.phi import iteration_mapping_receipt
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.semantics import verify_refinement


def explain_progressive_ir(lowered, *, mapping_bound: int = 100_000) -> dict:
    """Render the paper's functional→architecture→lowered→HLS provenance chain."""
    evidence = verify_refinement(lowered)
    arch = lowered.architectural
    functional = arch.functional
    tensor_by_name = {t.name: t for t in functional.tensors}
    producers = {name: None for name in tensor_by_name}
    consumers = {name: [] for name in tensor_by_name}
    for op in functional.operators:
        for name in op.inputs:
            consumers[name].append(op.name)
        for name in op.outputs:
            producers[name] = op.name
    ir_tensors = []
    for tensor in functional.tensors:
        ir_tensors.append({"id": tensor.name, "shape": list(tensor.shape),
                           "dtype": tensor.dtype, "constant": tensor.data is not None,
                           "producer": producers[tensor.name],
                           "consumers": consumers[tensor.name],
                           "constant_sha256": (sha256(repr(tensor.data).encode()).hexdigest()
                                               if tensor.data is not None else None)})
    emitted = emit_hls(lowered)
    manifest = {entry["operator"]: entry for entry in emitted.manifest["operators"]}
    stages = []
    for operator, point, mechanism in zip(functional.operators, arch.points, lowered.operators):
        if operator.name not in manifest:
            raise ValueError("Missing generated HLS provenance for Model IR operator")
        phi = iteration_mapping_receipt(point, max_iterations=mapping_bound)
        stages.append({
            "functional": {"id": operator.name, "kind": operator.kind,
                           "input_tensors": list(operator.inputs),
                           "output_tensors": list(operator.outputs),
                           "attributes": dict(operator.attributes)},
            "architectural": {"id": point.operator,
                              "requested": asdict(point.requested),
                              "effective": asdict(point.effective),
                              "decision_sources": dict(point.decision_sources),
                              "source_scope": point.source_scope,
                              "phi_sha256": phi["complete_mapping_sha256"],
                              "logical_iterations": phi["logical_iterations"],
                              "phi_sample": phi["mapping_preview"],
                              "semantic_properties": phi["oracle"]},
            "lowered": {"id": mechanism.name, "kind": mechanism.kind,
                        "tile_shape": list(mechanism.tile_shape),
                        "replicas": mechanism.replica_count,
                        "pe_count": mechanism.pe_count,
                        "simd_lanes": mechanism.simd_lanes,
                        "serialization": mechanism.fold_factor,
                        "pipeline_ii_request": mechanism.pipeline_ii,
                        "numeric_type": mechanism.numerical_type,
                        "tail_policy": mechanism.tail_policy},
            "hls": {"id": manifest[operator.name]["operator"],
                    "cpp_sha256": emitted.manifest["cpp_sha256"],
                    "materialization": manifest[operator.name]["lowered_mechanisms"],
                    "evidence": "E1-generated-source-only"},
        })
    return {"schema": "fpgai-research/progressive-model-ir-explain/v1",
            "evidence": "E0-IR-plus-E1-generated-HLS", "numerical_equivalence": None,
            "synthesis_results": None, "routed_results": None,
            "invariants": evidence,
            "graph": {"tensor_identities": ir_tensors,
                      "inputs": list(functional.inputs),
                      "outputs": list(functional.outputs),
                      "operators": [op.name for op in functional.operators]},
            "operator_stage_traces": stages}
