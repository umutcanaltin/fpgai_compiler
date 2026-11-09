"""Independent graph Phi issue order: bounded source conformance and mutants.

The candidate trace may pass value/read safety while violating the declared
operator source issue order. No HLS, GCC or original evidence is executed.
"""
from dataclasses import replace

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_graph_phi_fixtures import sample_graph
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir import graph_phi_contract as graph
from fpgai_research.model_ir.independent_phi_issue_reference import (
    reference_graph_phi_reader_order, validate_graph_phi_issue_conformance,
)


def _arch(kind="nested", *, mixed=False):
    model = sample_graph(kind, m=2, n=4)
    base = ArchitectureRequest()
    overrides = {}
    if mixed:
        for index, op in enumerate(model.operators):
            if op.kind == "MatMul":
                overrides[op.name] = replace(
                    base, pes=2 if index % 2 == 0 else 3,
                    simd=2 if index % 2 == 0 else 1,
                    serialization=2 if index % 2 == 0 else 1)
    return resolve(model, base, overrides)


@pytest.mark.parametrize("kind", ["mixed", "nested", "triple_reader"])
@pytest.mark.parametrize("mixed", [False, True])
def test_graph_phi_independent_ranks_accept_generated_source(kind, mixed):
    arch = _arch(kind, mixed=mixed)
    events, _ = graph.graph_source_events(arch)
    references = reference_graph_phi_reader_order(arch)
    assert references
    result = validate_graph_phi_issue_conformance(arch, events)
    assert result["status"] == "VERIFIED_E0"
    assert result["checked_operators"] == len(references)
    assert result["checked_reads"] == sum(map(len, references.values()))
    assert result["original_vendor_calls"] == 0
    assert len(result["observed_issue_signature_sha256"]) == 64
    certified = graph.graph_contract(arch, graph.graph_disjoint_offsets(arch))
    assert certified["independent_phi_issue_status"] == "VERIFIED_E0"
    assert certified["independent_phi_issue_checked_reads"] == result["checked_reads"]
    assert certified["independent_phi_issue_signature_sha256"] == result["observed_issue_signature_sha256"]


def test_value_safe_reordered_matmul_reads_violate_phi_source_order():
    arch = _arch()
    events, _ = graph.graph_source_events(arch)
    first = next(i for i in range(len(events) - 1)
                 if events[i].kind == events[i + 1].kind == "READ"
                 and events[i].consumer == events[i + 1].consumer
                 and events[i].consumer != "EXTERNAL"
                 and events[i].input_port == events[i + 1].input_port == 0
                 and events[i].output_element == events[i + 1].output_element
                 and events[i].element != events[i + 1].element)
    changed = list(events)
    changed[first], changed[first + 1] = changed[first + 1], changed[first]
    changed = tuple(changed)
    assert graph.verify_graph_events(
        arch, graph.graph_disjoint_offsets(arch), changed)["status"] == "VERIFIED_E0"
    check = validate_graph_phi_issue_conformance(arch, changed)
    assert check["status"] == "REJECTED"
    assert check["reason"] == "GRAPH_PHI_SOURCE_ISSUE_ORDER_OR_OWNER_MISMATCH"
    assert check["first_read_index"] >= 0


def test_independent_order_blocks_generator_common_mode_mutation(monkeypatch):
    arch = _arch()
    real = graph.mapped_matmul_order

    def corrupted(point, **kwargs):
        sequence = list(real(point, **kwargs))
        if point.operator != "reader0":
            return tuple(sequence)
        index = next(i for i in range(len(sequence) - 1)
                     if sequence[i][:2] == sequence[i + 1][:2]
                     and sequence[i][2] != sequence[i + 1][2])
        sequence[index], sequence[index + 1] = sequence[index + 1], sequence[index]
        return tuple(sequence)

    monkeypatch.setattr(graph, "mapped_matmul_order", corrupted)
    events, _ = graph.graph_source_events(arch)
    assert graph.verify_graph_events(
        arch, graph.graph_disjoint_offsets(arch), events)["status"] == "VERIFIED_E0"
    assert validate_graph_phi_issue_conformance(arch, events)["status"] == "REJECTED"
    with pytest.raises(ValueError, match="GRAPH_PHI_SOURCE_ISSUE_MISMATCH"):
        graph.graph_contract(arch, graph.graph_disjoint_offsets(arch))


def test_independent_graph_phi_reader_fails_closed_on_budget_and_shape():
    arch = _arch()
    events, _ = graph.graph_source_events(arch)
    with pytest.raises(ValueError, match="budget"):
        reference_graph_phi_reader_order(arch, max_reads=1)
    with pytest.raises(ValueError, match="immutable typed events"):
        validate_graph_phi_issue_conformance(arch, list(events))
