from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from fpgai.runtime.package_builder import emit_runtime_package


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generated_board_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_plan() -> dict:
    return {
        "schema": "fpgai.persistent-state-plan/v1",
        "tensor_count": 2,
        "backend_required": True,
        "tensors": [
            {
                "name": "layer0_k",
                "dtype": "float32",
                "shape": [1, 3, 8, 4],
                "storage": "ddr",
                "residency": "external",
                "owner": "transformer.layer.0",
                "state_group": "transformer.layer.0.kv",
            },
            {
                "name": "layer0_v",
                "dtype": "float32",
                "shape": [1, 3, 8, 4],
                "storage": "uram",
                "residency": "on_chip",
                "owner": "transformer.layer.0",
                "state_group": "transformer.layer.0.kv",
            },
        ],
    }


def test_runtime_buffer_plan_materializes_only_external_persistent_state(tmp_path: Path) -> None:
    out = tmp_path / "compile"
    out.mkdir()
    emit_runtime_package(out, pipeline_mode="inference", persistent_state_plan=_state_plan())
    manifest = json.loads((out / "runtime_package" / "package_manifest.json").read_text())
    buffers = {item["name"]: item for item in manifest["runtime_buffer_plan"]["buffers"]}
    assert "state__layer0_k" in buffers
    assert "state__layer0_v" not in buffers
    state = buffers["state__layer0_k"]
    assert state["role"] == "persistent_state"
    assert state["direction"] == "bidirectional"
    assert state["shape"] == [1, 3, 8, 4]
    assert state["state_name"] == "layer0_k"
    assert state["storage"] == "ddr"
    assert state["owner"] == "transformer.layer.0"


def test_generated_board_runtime_reads_writes_and_resets_external_state_buffers(tmp_path: Path) -> None:
    out = tmp_path / "compile"
    out.mkdir()
    emit_runtime_package(out, pipeline_mode="inference", persistent_state_plan=_state_plan())
    package = out / "runtime_package"
    board = _load_module(package / "board_runtime.py")
    plan = json.loads((package / "buffer_plan.json").read_text())

    def allocate_fn(*, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    buffers = board.allocate_buffers_from_plan(plan, allocate_fn=allocate_fn, package_dir=package)

    class Backend:
        pass

    runtime = board.FPGAIBoardRuntime(Backend(), package_dir=package, buffers=buffers)
    values = np.arange(96, dtype=np.float32).reshape(1, 3, 8, 4)
    result = runtime.write_state(values, name="layer0_k")
    assert result == {"written": "layer0_k"}
    np.testing.assert_array_equal(runtime.read_state(name="layer0_k"), values)
    exported = runtime.export_state()
    assert set(exported) == {"layer0_k"}
    np.testing.assert_array_equal(exported["layer0_k"], values)
    assert runtime.reset_state(name="layer0_k") == {"reset": ["layer0_k"]}
    np.testing.assert_array_equal(runtime.read_state(name="layer0_k"), np.zeros_like(values))


def test_generated_board_runtime_preserves_backend_state_override(tmp_path: Path) -> None:
    out = tmp_path / "compile"
    out.mkdir()
    emit_runtime_package(out, pipeline_mode="inference", persistent_state_plan=_state_plan())
    board = _load_module(out / "runtime_package" / "board_runtime.py")

    class Backend:
        def read_state(self, *, name=None):
            return {"backend": name}

        def write_state(self, payload, *, name=None):
            return {"backend_write": name, "payload": payload}

        def reset_state(self, name=None):
            return {"backend_reset": name}

    runtime = board.FPGAIBoardRuntime(Backend(), package_dir=out / "runtime_package")
    assert runtime.read_state(name="x") == {"backend": "x"}
    assert runtime.write_state([1], name="x") == {"backend_write": "x", "payload": [1]}
    assert runtime.reset_state(name="x") == {"backend_reset": "x"}
