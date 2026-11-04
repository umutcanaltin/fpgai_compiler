"""CST Model IR row-lifetime prototype: source, numerical and temporal witnesses."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_interleaved import compile_row_interleaved
from fpgai_research.compiler.composition import compile_composed_architecture
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.model_ir.row_lifetime import resolve_row_interleaving
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.cpp_phi_trace import trace_cpp_phi


def two_matmuls(*, rows=3, middle=5):
    # Non-square intermediate makes row-epoch aliasing observable.
    inner, output = 3, 2
    return FunctionalModel(
        tensors=(
            Tensor("x", (rows, inner)),
            Tensor("wa", (inner, middle), data=(0.125,) * (inner * middle)),
            Tensor("hidden", (rows, middle)),
            Tensor("wb", (middle, output), data=(0.0625,) * (middle * output)),
            Tensor("y", (rows, output)),
        ),
        operators=(
            Operator("left", "MatMul", ("x", "wa"), ("hidden",)),
            Operator("right", "MatMul", ("hidden", "wb"), ("y",)),
        ), inputs=("x",), outputs=("y",),
    )


def independent_local_architecture(f):
    base = ArchitectureRequest()
    return resolve(f, base, {
        "left": replace(base, parallelism=2, pes=2, simd=2,
                        serialization=2, tile=(1, 3, 2), pipeline_ii=2),
        "right": replace(base, parallelism=3, pes=2, simd=3,
                         serialization=2, tile=(1, 2, 3), pipeline_ii=2),
    })


def test_row_lifetime_ir_has_temporal_aliasing_and_reduced_scratch():
    f = two_matmuls()
    arch = independent_local_architecture(f)
    compiled = compile_row_interleaved(arch)
    c = compiled.ir.row_contract
    assert compiled.ir.base.boundaries[0].storage_layout == "row_buffer"
    assert compiled.ir.base.boundaries[0].physical_elements == 5
    assert compiled.ir.base.boundaries[0].exchange_policy == "producer_row_complete_then_consumer_row"
    assert c.tensor == "hidden"
    assert c.shape == (3, 5)
    assert c.logical_tensor_elements == 15
    assert c.physical_row_buffer_elements == 5
    assert c.producer_phi_sha256 != c.consumer_phi_sha256
    assert "row" in c.disjoint_lifetime_epochs
    assert compiled.emission.manifest["intermediate_scratch_elements"] == 5
    assert compiled.emission.manifest["full_materialized_intermediate_elements"] == 15
    assert compiled.evidence["vitis_csynth_executed"] is False
    assert "data_t row_buffer[5]" in compiled.emission.cpp
    assert "row_buffer[ni]=acc;" in compiled.emission.cpp
    assert "row_buffer[kk]*w1" in compiled.emission.cpp


@pytest.mark.parametrize("rows,middle", [(2, 3), (3, 5), (5, 7)])
def test_actual_row_interleaved_cpp_numerics_and_phi(rows, middle):
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler unavailable")
    model = two_matmuls(rows=rows, middle=middle)
    arch = independent_local_architecture(model)
    conventional = compile_composed_architecture(arch)
    candidate = compile_row_interleaved(arch)
    assert candidate.emission.manifest["functional_sha256"] == conventional.emission.manifest["functional_sha256"]
    assert candidate.emission.manifest["cpp_sha256"] != conventional.emission.manifest["cpp_sha256"]
    x = np.linspace(-0.125, 0.125, rows * 3,
                    dtype=np.float32).reshape(rows, 3)
    baseline = validate_host(model, conventional.emission, x)
    actual = validate_host(model, candidate.emission, x)
    assert baseline["accepted"]
    assert actual["accepted"]
    assert actual["max_absolute_error"] < 1e-5
    # This checks executed C++ iteration coordinates against Phi, not an
    # analytical duplicate of the new emitter or an actual Vitis cycle.
    trace = trace_cpp_phi(candidate.ir.base.lowered,
                          emission=candidate.emission, max_iterations=10000)
    assert trace["source_phi_agreement"]
    assert trace["logical_iterations"] == rows * 3 * middle + rows * middle * 2


def test_actual_cpp_row_ready_consumed_order(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler unavailable")
    model = two_matmuls()
    comp = compile_row_interleaved(independent_local_architecture(model))
    source = tmp_path / "row.cpp"
    exe = tmp_path / "row"
    source.write_text(comp.emission.cpp + """
int main() {
 float a[9]={}; float y[6]={};
 fpgai_model(a,y); return 0;
}
""")
    command = ["g++", "-std=c++17", "-O0", "-DFPGAI_HOST_FLOAT",
               "-DFPGAI_TRACE_ROWS", "-Wno-unknown-pragmas",
               str(source), "-o", str(exe)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert build.returncode == 0, build.stderr[-2000:]
    execution = subprocess.run([str(exe)], capture_output=True,
                               text=True, timeout=45)
    assert execution.returncode == 0
    assert execution.stdout.splitlines() == [
        "FPGAI_ROW_READY 0", "FPGAI_ROW_CONSUMED 0",
        "FPGAI_ROW_READY 1", "FPGAI_ROW_CONSUMED 1",
        "FPGAI_ROW_READY 2", "FPGAI_ROW_CONSUMED 2",
    ]


def test_illegal_row_compositions_are_blocked():
    model = two_matmuls()
    base = ArchitectureRequest()
    incompatible_tile = resolve(model, base, {
        "left": replace(base, tile=(2, 3, 2)),
    })
    with pytest.raises(ValueError, match="M tiling"):
        resolve_row_interleaving(incompatible_tile)
    with pytest.raises(ValueError, match="requires exactly two"):
        extended = FunctionalModel(
            tensors=model.tensors + (Tensor("post", (3, 2)),),
            operators=model.operators +
                (Operator("relu", "Relu", ("y",), ("post",)),),
            inputs=("x",), outputs=("post",),
        )
        resolve_row_interleaving(resolve(extended, base))
