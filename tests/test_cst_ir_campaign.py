"""CPU-only experiment and cross-operator element witness regression tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.composition import compile_composed_architecture
from fpgai_research.experiments.cst_ir_campaign import run, _model
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.exchange_witness import check_element_exchange


def test_actual_phi_to_tensor_element_witness_links_all_consumer_reductions():
    arch = resolve(_model(3, 4, 5, 2), ArchitectureRequest())
    compiled = compile_composed_architecture(
        arch, intermediate_layouts={"h": "column_major"})
    report = check_element_exchange(compiled.composed_ir)
    assert report["native_vitis_calls"] == 0
    assert len(report["relations"]) == 1
    edge = report["relations"][0]
    assert edge["status"] == "BOUNDED_PHI_EXCHANGE_VERIFIED"
    assert edge["elements"] == 15
    assert edge["logical_producer_consumer_events"] == 3 * 5 * (1 + 2)
    assert edge["all_logical_values_distinct"]
    assert edge["producer_before_consumers_symbolically"]
    assert edge["preview"][0]["kind"] == "producer_completed_after_reduction"


def test_element_witness_reports_bound_exceeded_without_false_pass():
    arch = resolve(_model(3, 4, 5, 2), ArchitectureRequest())
    compiled = compile_composed_architecture(arch)
    report = check_element_exchange(compiled.composed_ir, max_events=1)
    assert report["relations"][0]["status"] == "BOUND_EXCEEDED"
    with pytest.raises(ValueError, match="Positive maximum"):
        check_element_exchange(compiled.composed_ir, max_events=0)


def test_compiler_cpu_experiment_is_create_only_and_source_paired(tmp_path):
    dest = tmp_path / "cpu_cst"
    report = run(output=dest, max_cases=8, host_check_pairs=0)
    assert report["cases"] == 8
    assert report["matched_pairs"] == 4
    assert report["distinct_functional_graphs"] == 2
    assert report["graphs_with_at_least_two_architecture_mappings"] == 2
    assert report["max_effective_mappings_per_functional_graph"] == 2
    assert report["vitis_calls"] == 0
    assert report["distinct_generated_cpp"] >= report["distinct_effective_architectures"]
    assert report["all_alternative_layouts_materialized"]
    assert report["host_checked_cases"] == 0
    with (dest / "cases.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    for first, second in zip(rows[::2], rows[1::2]):
        assert first["functional_sha256"] == second["functional_sha256"]
        assert first["effective_architecture_sha256"] == second["effective_architecture_sha256"]
        assert first["cpp_sha256"] != second["cpp_sha256"]
    assert json.loads((dest / "summary.json").read_text())["cases"] == 8
    with pytest.raises(FileExistsError, match="Create-only"):
        run(output=dest, max_cases=8, host_check_pairs=0)


def test_campaign_rejects_invalid_or_unsafely_large_requests(tmp_path):
    with pytest.raises(ValueError, match="max_cases"):
        run(output=tmp_path / "bad", max_cases=1, host_check_pairs=0)
    with pytest.raises(ValueError, match="host_check_pairs"):
        run(output=tmp_path / "bad2", max_cases=4, host_check_pairs=3)
