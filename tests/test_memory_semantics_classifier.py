from __future__ import annotations

from pathlib import Path

import pytest

from fpgai.reports.memory_semantics import classify_generated_memory_semantics


ROOT = Path("paper_experiments/full_pipeline_gate/sprint27j_paper_validation/runs_hls")


def _run(name: str) -> Path:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"generated run not available: {p}")
    return p


def test_embedded_memory_row_is_classified_as_embedded_constants() -> None:
    sem = classify_generated_memory_semantics(_run("kv260_memory_bram"))
    assert sem.mode == "embedded_constants"
    assert not sem.has_weights_mem
    assert not sem.has_runtime_payload
    assert sem.has_const_weight_arrays


def test_ddr_memory_row_is_classified_as_preload_not_tiled() -> None:
    sem = classify_generated_memory_semantics(_run("kv260_memory_ddr"))
    assert sem.mode == "ddr_preload_full"
    assert sem.has_weights_mem
    assert sem.has_weights_m_axi
    assert sem.has_runtime_payload
    assert sem.has_full_weight_arrays
    assert not sem.has_tile_weight_buffer


def test_new_schema_ddr_memory_row_is_classified_as_preload_not_tiled() -> None:
    sem = classify_generated_memory_semantics(_run("kv260_memory_ddr_new_schema"))
    assert sem.mode == "ddr_preload_full"
    assert sem.has_weights_mem
    assert sem.has_weights_m_axi
    assert sem.has_runtime_payload
    assert sem.has_full_weight_arrays
    assert not sem.has_tile_weight_buffer


def test_uram_memory_row_is_classified_as_uram_preload_full() -> None:
    sem = classify_generated_memory_semantics(_run("kv260_memory_uram"))
    assert sem.mode == "uram_preload_full"
    assert sem.has_weights_mem
    assert sem.has_weights_m_axi
    assert sem.has_runtime_payload
    assert sem.has_full_weight_arrays
    assert sem.has_uram_weight_bind


def test_precision_rows_are_embedded_constants_not_memory_experiments() -> None:
    for name in ["kv260_precision_fx8_3", "kv260_precision_fx12_4", "kv260_precision_fx16_6"]:
        sem = classify_generated_memory_semantics(_run(name))
        assert sem.mode == "embedded_constants"
