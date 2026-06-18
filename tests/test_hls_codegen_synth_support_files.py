from __future__ import annotations

from dataclasses import dataclass
import re
from types import SimpleNamespace

import numpy as np

from fpgai.backends.hls.codegen import emit_hls_stub


@dataclass
class TensorSpec:
    shape: tuple[int, ...]


@dataclass
class Op:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict


class TinyDenseGraph:
    inputs = ["x"]
    outputs = ["y"]

    def __init__(self) -> None:
        self.ops = [
            Op(
                name="dense0",
                op_type="Dense",
                inputs=["x", "W0", "B0"],
                outputs=["y"],
                attrs={"precision_tag": "dense0"},
            )
        ]
        self.tensors = {
            "x": TensorSpec((1, 16)),
            "W0": TensorSpec((8, 16)),
            "B0": TensorSpec((8,)),
            "y": TensorSpec((1, 8)),
        }
        self.constants = {
            "W0": np.ones((8, 16), dtype=np.float32),
            "B0": np.zeros((8,), dtype=np.float32),
        }

    def get_tensor(self, name: str):
        return self.tensors.get(name)


def test_emit_hls_stub_writes_synthesis_support_files(tmp_path) -> None:
    project = emit_hls_stub(
        graph=TinyDenseGraph(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "weights_mode": "embedded",
            "run_csim": False,
            "run_csynth": False,
        },
        compile_plan=SimpleNamespace(layer_plans=[]),
    )

    assert (project.hls_dir / "src" / "fpgai_params.cpp").exists()
    assert (project.hls_dir / "include" / "layers" / "batchnorm.h").exists()

    types = (project.hls_dir / "include" / "fpgai_types.h").read_text(
        encoding="utf-8"
    )
    params = (project.hls_dir / "include" / "fpgai_params.h").read_text(
        encoding="utf-8"
    )
    tcl = project.run_tcl.read_text(encoding="utf-8")

    assert "typedef float dense0_act_t;" in types
    assert "static const dense0_wgt_t W0[8][16]" in params
    assert "static const dense0_bias_t B0[8]" in params
    assert "add_files ${SRC_DIR}/fpgai_params.cpp" in tcl


def test_generated_params_do_not_emit_invalid_integer_float_literals(tmp_path) -> None:
    project = emit_hls_stub(
        graph=TinyDenseGraph(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "weights_mode": "embedded",
            "run_csim": False,
            "run_csynth": False,
        },
        compile_plan=SimpleNamespace(layer_plans=[]),
    )

    params = (project.hls_dir / "include" / "fpgai_params.h").read_text(
        encoding="utf-8"
    )

    invalid_literals = re.findall(
        r"(?<![A-Za-z0-9_.])[-+]?[0-9]+f\b",
        params,
    )
    assert invalid_literals == []
    assert "0.0f" in params
    assert "1.0f" in params
