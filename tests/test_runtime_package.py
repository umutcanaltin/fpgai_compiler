from __future__ import annotations

import json
from pathlib import Path

from fpgai.runtime.package import emit_runtime_package


def test_emit_runtime_package_copies_existing_artifacts_and_records_missing_hw(tmp_path: Path) -> None:
    out_dir = tmp_path / "build"
    out_dir.mkdir()

    (out_dir / "manifest.json").write_text('{"top_kernel_name": "deeplearn"}', encoding="utf-8")
    (out_dir / "input.bin").write_bytes(b"input")
    (out_dir / "hls_artifact_metadata.json").write_text('{"schema_version": 1}', encoding="utf-8")
    (out_dir / "hls_schedule_summary.json").write_text('{"report_count": 0}', encoding="utf-8")
    (out_dir / "hls_ii_comparison.json").write_text('{"summary": {}}', encoding="utf-8")

    logs = out_dir / "hls" / "logs"
    logs.mkdir(parents=True)
    (logs / "vitis_hls_stdout.log").write_text("ok", encoding="utf-8")

    result = emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="inference",
        top_name="deeplearn",
        hls_artifacts={"hls_ran": True},
    )

    assert result["status"] == "created"
    assert result["path"] == "runtime_package/package_manifest.json"
    assert result["deployable_overlay_present"] is False
    assert result["bitstream_present"] is False
    assert result["hwh_present"] is False
    assert result["xsa_present"] is False

    package_manifest = out_dir / "runtime_package" / "package_manifest.json"
    data = json.loads(package_manifest.read_text(encoding="utf-8"))

    assert data["board"] == "kv260"
    assert data["pipeline_mode"] == "inference"
    assert data["top_name"] == "deeplearn"
    assert data["hls_artifacts"]["hls_ran"] is True
    assert data["hardware"]["deployable_overlay_present"] is False
    assert data["files"]["compile_manifest"]["package_path"] == "manifest.json"
    assert data["files"]["input_bin"]["package_path"] == "inputs/input.bin"
    assert data["files"]["hls_logs"][0]["package_path"].startswith("hls/logs/")
    assert (out_dir / "runtime_package" / "README_RUNTIME.md").exists()


def test_emit_runtime_package_records_deployable_overlay_when_hw_files_exist(tmp_path: Path) -> None:
    out_dir = tmp_path / "build"
    hw = out_dir / "vivado_bridge" / "bitstream"
    hw.mkdir(parents=True)

    (hw / "design.bit").write_bytes(b"bit")
    (hw / "design.hwh").write_text("hwh", encoding="utf-8")

    result = emit_runtime_package(
        out_dir,
        board="pynq_z2",
        pipeline_mode="inference",
        top_name="deeplearn",
    )

    assert result["deployable_overlay_present"] is True
    assert result["bitstream_present"] is True
    assert result["hwh_present"] is True
    assert result["xsa_present"] is False

    data = json.loads((out_dir / "runtime_package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert data["hardware"]["deployable_overlay_present"] is True
    assert data["files"]["bitstream"]["package_path"] == "hardware/design.bit"
    assert data["files"]["hwh"]["package_path"] == "hardware/design.hwh"


def test_compiler_wires_runtime_package_into_manifest_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "emit_runtime_package" in source
    assert "runtime_package=runtime_package" in source
    assert '"runtime_package": kwargs.get("runtime_package")' in source
    assert '"done" if kwargs.get("runtime_package") is not None else "skipped"' in source


def test_emit_runtime_package_creates_runtime_weight_payload_for_uram(tmp_path: Path) -> None:
    import struct

    out_dir = tmp_path / "compile"
    params_dir = out_dir / "hls" / "src"
    params_dir.mkdir(parents=True)

    (params_dir / "fpgai_params.cpp").write_text(
        """
#include "fpgai_params.h"

namespace fpgai {
op0_wgt_t W0[2];
op0_bias_t B0[1];
const op0_wgt_t W0_init[2] = { 1.0, -2.0 };
const op0_bias_t B0_init[1] = { 0.5 };
} // namespace fpgai
""",
        encoding="utf-8",
    )

    result = emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="inference",
        top_name="deeplearn",
        weights_mode="uram_import_full",
    )

    assert result["runtime_weight_payload_required"] is True
    assert result["runtime_weight_payload_present"] is True
    assert result["runtime_weight_total_words"] == 3

    package_dir = out_dir / "runtime_package"
    weights_bin = package_dir / "weights" / "weights.bin"
    layout_path = package_dir / "weights" / "weight_layout.json"
    manifest_path = package_dir / "package_manifest.json"

    assert weights_bin.exists()
    assert layout_path.exists()

    words = struct.unpack("<III", weights_bin.read_bytes())
    expected = tuple(
        struct.unpack("<I", struct.pack("<f", v))[0]
        for v in [1.0, -2.0, 0.5]
    )
    assert words == expected

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    assert layout["format"] == "packed32"
    assert layout["total_words"] == 3
    assert layout["entries"] == [
        {"name": "W0", "kind": "weight", "offset_words": 0, "count_words": 2},
        {"name": "B0", "kind": "bias", "offset_words": 2, "count_words": 1},
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime_weights"]["required"] is True
    assert manifest["runtime_weights"]["present"] is True
    assert manifest["runtime_weights"]["total_words"] == 3
    assert manifest["files"]["weights_bin"]["package_path"] == "weights/weights.bin"
    assert manifest["files"]["weight_layout"]["package_path"] == "weights/weight_layout.json"


def test_emit_runtime_package_does_not_create_weight_payload_for_embedded_bram(tmp_path: Path) -> None:
    out_dir = tmp_path / "compile"
    params_dir = out_dir / "hls" / "src"
    params_dir.mkdir(parents=True)

    (params_dir / "fpgai_params.cpp").write_text(
        """
#include "fpgai_params.h"

namespace fpgai {
const op0_wgt_t W0[2] = { 1.0, -2.0 };
const op0_bias_t B0[1] = { 0.5 };
} // namespace fpgai
""",
        encoding="utf-8",
    )

    result = emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="inference",
        top_name="deeplearn",
        weights_mode="bram_static",
    )

    assert result["runtime_weight_payload_required"] is False
    assert result["runtime_weight_payload_present"] is False
    assert result["runtime_weight_total_words"] == 0

    package_dir = out_dir / "runtime_package"
    assert not (package_dir / "weights" / "weights.bin").exists()
    assert not (package_dir / "weights" / "weight_layout.json").exists()

    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_weights"]["required"] is False
    assert manifest["runtime_weights"]["present"] is False
    assert manifest["runtime_weights"]["total_words"] == 0
    assert "weights_bin" not in manifest["files"]
    assert "weight_layout" not in manifest["files"]


def test_runtime_package_ddr_tiled_requires_weight_payload(tmp_path: Path) -> None:
    import json

    from fpgai.runtime.package import emit_runtime_package

    src = tmp_path / "hls/src"
    src.mkdir(parents=True)
    (src / "fpgai_params.cpp").write_text(
        """
#include "fpgai_params.h"
namespace fpgai {
const op0_wgt_t W0_init[4] = { 1, 2, 3, 4 };
const op0_bias_t B0_init[2] = { 0, 0 };
}
"""
    )

    emit_runtime_package(tmp_path, weights_mode="ddr_tiled")
    manifest = json.loads((tmp_path / "runtime_package/package_manifest.json").read_text())
    runtime_weights = manifest["runtime_weights"]

    assert runtime_weights["weights_mode"] == "ddr_tiled"
    assert runtime_weights["required"] is True
    assert runtime_weights["present"] is True
    assert runtime_weights["import_required"] is True
    assert runtime_weights["export_supported"] is False
    assert runtime_weights["reload_before_each_compute"] is False
    assert runtime_weights["total_words"] == 6


def test_runtime_package_bram_import_export_marks_export_supported(tmp_path: Path) -> None:
    import json

    src = tmp_path / "hls/src"
    src.mkdir(parents=True)
    (src / "fpgai_params.cpp").write_text(
        """
#include "fpgai_params.h"
namespace fpgai {
const op0_wgt_t W0_init[2] = { 1.0, 2.0 };
const op0_bias_t B0_init[1] = { 0.0 };
}
""",
        encoding="utf-8",
    )

    emit_runtime_package(tmp_path, weights_mode="bram_import_export_full")
    manifest = json.loads((tmp_path / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    runtime_weights = manifest["runtime_weights"]

    assert runtime_weights["weights_mode"] == "bram_import_export_full"
    assert runtime_weights["required"] is True
    assert runtime_weights["import_required"] is True
    assert runtime_weights["export_supported"] is True
    assert runtime_weights["reload_before_each_compute"] is False
    assert runtime_weights["total_words"] == 3


def test_emit_runtime_package_records_m_axi_input_output_movement(tmp_path: Path) -> None:
    from fpgai.engine.models import CommunicationEdge, CommunicationPlan

    out_dir = tmp_path / "build"
    out_dir.mkdir()

    communication_plan = CommunicationPlan(
        edges=[
            CommunicationEdge(
                tensor_name="x",
                direction="PS_TO_PL",
                notes={"kind": "input", "interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
            ),
            CommunicationEdge(
                tensor_name="y",
                direction="PL_TO_PS",
                notes={"kind": "output", "interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
            ),
        ]
    )

    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="inference",
        top_name="deeplearn",
        communication_plan=communication_plan,
    )

    data = json.loads((out_dir / "runtime_package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert data["runtime_io"]["inputs"]["import"]["resolved"] == "m_axi_import_full"
    assert data["runtime_io"]["outputs"]["export"]["resolved"] == "m_axi_export_full"
    assert data["runtime_io"]["inputs"]["import"]["transport"] == "ps_runtime"



def test_emit_runtime_package_records_m_axi_tiled_input_output_movement(tmp_path: Path) -> None:
    from fpgai.engine.models import CommunicationEdge, CommunicationPlan

    out_dir = tmp_path / "build_tiled"
    out_dir.mkdir()

    communication_plan = CommunicationPlan(
        edges=[
            CommunicationEdge(
                tensor_name="x",
                direction="PS_TO_PL",
                notes={"kind": "input", "interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
            ),
            CommunicationEdge(
                tensor_name="y",
                direction="PL_TO_PS",
                notes={"kind": "output", "interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
            ),
        ]
    )

    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="inference",
        top_name="deeplearn",
        communication_plan=communication_plan,
    )

    data = json.loads((out_dir / "runtime_package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert data["runtime_io"]["inputs"]["import"]["resolved"] == "m_axi_import_tiled"
    assert data["runtime_io"]["outputs"]["export"]["resolved"] == "m_axi_export_tiled"
    assert data["runtime_io"]["inputs"]["import"]["policy"] == "tiled"

def test_runtime_package_records_activation_storage(tmp_path: Path) -> None:
    import json
    from types import SimpleNamespace
    from fpgai.runtime.package import emit_runtime_package

    emit_runtime_package(
        tmp_path,
        weights_mode="bram_static",
        memory_plan=SimpleNamespace(notes={"resolved_activation_storage": "uram"}),
    )
    manifest = json.loads((tmp_path / "runtime_package/package_manifest.json").read_text())
    assert manifest["runtime_activation_storage"] == {
        "storage": "uram",
        "resolved": "activation_uram",
        "local_buffers": True,
    }


def test_runtime_package_optimizer_state_capture_api_and_files(tmp_path: Path) -> None:
    import importlib.util
    import struct

    out_dir = tmp_path / "compile"
    ref_dir = out_dir / "training_reference"
    ref_dir.mkdir(parents=True)
    (out_dir / "optimizer_state_after.bin").write_bytes(struct.pack("<ff", 0.1, -0.2))
    (ref_dir / "optimizer_state_after_ref.bin").write_bytes(struct.pack("<ff", 0.1, -0.2))

    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
    )

    package_dir = out_dir / "runtime_package"
    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_optimizer_state"]["capture_supported_by_api"] is True
    assert manifest["runtime_optimizer_state"]["captured_state_present"] is True
    assert manifest["runtime_optimizer_state"]["reference_state_present"] is True
    assert manifest["files"]["optimizer_state_after_bin"]["package_path"] == "outputs/optimizer_state_after.bin"
    assert manifest["files"]["optimizer_state_after_ref_bin"]["package_path"] == "reference/optimizer_state_after_ref.bin"

    api_path = package_dir / "runtime_api.py"
    api_source = api_path.read_text(encoding="utf-8")
    assert "def capture_optimizer_state" in api_source
    assert "board_payload" in api_source

    spec = importlib.util.spec_from_file_location("runtime_api_capture_test", api_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = package_dir / "outputs" / "captured_from_test.bin"
    returned = module.export_optimizer_state(board_payload=b"abcd", capture_path=target)
    assert returned == b"abcd"
    assert target.read_bytes() == b"abcd"


def test_runtime_api_can_capture_gradient_export_payload(tmp_path: Path) -> None:
    out_dir = tmp_path / 'runtime_gradient_capture'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'manifest.json').write_text('{}', encoding='utf-8')
    (out_dir / 'gradients_after.bin').write_bytes(b'abcd')
    (out_dir / 'training_reference').mkdir(parents=True, exist_ok=True)
    (out_dir / 'training_reference' / 'grads_ref.bin').write_bytes(b'abcd')

    from fpgai.runtime.package import emit_runtime_package

    emit_runtime_package(
        out_dir,
        board='kv260',
        pipeline_mode='training_on_device',
        top_name='deeplearn',
        runtime_sequence={'sequence': [{'command': 'export_gradients', 'args': {}}]},
    )
    manifest = json.loads((out_dir / 'runtime_package' / 'package_manifest.json').read_text(encoding='utf-8'))
    assert manifest['runtime_gradient_export']['captured_gradients_present'] is True
    assert manifest['runtime_gradient_export']['reference_gradients_present'] is True
    api_source = (out_dir / 'runtime_package' / 'runtime_api.py').read_text(encoding='utf-8')
    assert 'def capture_gradients' in api_source
    assert 'def export_gradients(*, board_payload' in api_source


def test_runtime_package_generates_board_runtime_adapter_with_hls_modes(tmp_path: Path) -> None:
    out_dir = tmp_path / "board_runtime_package"
    out_dir.mkdir(parents=True)

    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        runtime_sequence={
            "sequence": [
                {"command": "run_training", "args": {"steps": 1}},
                {"command": "export_gradients", "args": {}},
                {"command": "export_optimizer_state", "args": {}},
            ]
        },
    )

    package_dir = out_dir / "runtime_package"
    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["board_runtime"]["present"] is True
    assert manifest["board_runtime"]["hls_modes"] == {
        "run_training": 2,
        "accumulate_gradients": 3,
        "apply_accumulated_gradients": 4,
        "reset_accumulators": 5,
        "export_gradients": 8,
        "export_optimizer_state": 9,
    }

    board_runtime_source = (package_dir / "board_runtime.py").read_text(encoding="utf-8")
    assert "class FPGAIBoardRuntime" in board_runtime_source
    assert "FPGAI_MODE_EXPORT_GRADIENTS = 8" in board_runtime_source
    assert "FPGAI_MODE_EXPORT_OPTIMIZER_STATE = 9" in board_runtime_source
    assert "read_buffer('gradients_mem')" in board_runtime_source or 'read_buffer(logical_name)' in board_runtime_source

    runtime_api_source = (package_dir / "runtime_api.py").read_text(encoding="utf-8")
    assert "def bind_backend" in runtime_api_source
    assert "_BOUND_BACKEND.export_gradients" in runtime_api_source
    assert "_BOUND_BACKEND.export_optimizer_state" in runtime_api_source


def test_runtime_api_bound_backend_executes_export_capture_sequence(tmp_path: Path) -> None:
    import importlib.util

    out_dir = tmp_path / "bound_backend_runtime"
    out_dir.mkdir(parents=True)
    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        runtime_sequence={
            "sequence": [
                {"command": "run_training", "args": {"steps": 2}},
                {"command": "export_gradients", "args": {}},
                {"command": "export_optimizer_state", "args": {}},
            ]
        },
    )
    package_dir = out_dir / "runtime_package"
    api_path = package_dir / "runtime_api.py"
    spec = importlib.util.spec_from_file_location("runtime_api_bound_backend_test", api_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeBackend:
        def __init__(self):
            self.modes: list[int] = []

        def call_mode(self, mode: int, **kwargs):
            self.modes.append(mode)
            return {"mode": mode, "kwargs": kwargs}

        def read_buffer(self, name: str) -> bytes:
            if name == "gradients_mem":
                return b"gradients"
            if name == "optimizer_state_mem":
                return b"optimizer"
            raise KeyError(name)

    fake = FakeBackend()
    module.bind_backend(fake)
    results = module.run_sequence()

    assert fake.modes == [2, 2, 8, 9]
    assert results[1] == b"gradients"
    assert results[2] == b"optimizer"
    assert (package_dir / "outputs" / "gradients_after.bin").read_bytes() == b"gradients"
    assert (package_dir / "outputs" / "optimizer_state_after.bin").read_bytes() == b"optimizer"


def test_runtime_package_generates_concrete_pynq_kv260_backend(tmp_path: Path) -> None:
    out_dir = tmp_path / "pynq_backend_runtime"
    out_dir.mkdir(parents=True)
    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        runtime_sequence={"sequence": [{"command": "run_training", "args": {"steps": 1}}]},
    )

    package_dir = out_dir / "runtime_package"
    source = (package_dir / "board_runtime.py").read_text(encoding="utf-8")
    assert "class PynqDmaMmioBackend" in source
    assert "def create_pynq_backend" in source
    assert "from pynq import Overlay" in source
    assert "self.ip.write(self.mode_offset, int(mode))" in source
    assert "self.ip.write(self.ap_ctrl_offset, self.start_mask)" in source
    assert "def bind_buffer" in source
    assert "read_buffer('gradients_mem')" in source
    assert "read_buffer('optimizer_state_mem')" in source

    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))
    assert "PynqDmaMmioBackend" in (package_dir / "board_runtime.py").read_text(encoding="utf-8")
    assert "create_pynq_backend" in manifest["board_runtime"]["backend_contract"]


def test_pynq_backend_fake_ip_executes_modes_and_reads_bound_buffers(tmp_path: Path) -> None:
    import importlib.util

    out_dir = tmp_path / "pynq_backend_fake_runtime"
    out_dir.mkdir(parents=True)
    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        runtime_sequence={"sequence": []},
    )
    package_dir = out_dir / "runtime_package"
    spec = importlib.util.spec_from_file_location("board_runtime_fake_test", package_dir / "board_runtime.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeIP:
        def __init__(self):
            self.writes = []
        def write(self, offset, value):
            self.writes.append((offset, value))
        def read(self, offset):
            return 0x2

    ip = FakeIP()
    backend = module.PynqDmaMmioBackend(
        overlay=ip,
        buffers={"gradients_mem": b"grad", "optimizer_state_mem": b"opt"},
        mode_offset=0x18,
    )

    assert backend.export_gradients() == b"grad"
    assert backend.export_optimizer_state() == b"opt"
    assert (0x18, module.FPGAI_MODE_EXPORT_GRADIENTS) in ip.writes
    assert (0x18, module.FPGAI_MODE_EXPORT_OPTIMIZER_STATE) in ip.writes


def test_runtime_package_emits_buffer_and_execution_plans(tmp_path: Path) -> None:
    out_dir = tmp_path / "runtime_buffer_plan"
    out_dir.mkdir(parents=True)
    (out_dir / "input.bin").write_bytes(b"\x00" * 16)
    (out_dir / "output.bin").write_bytes(b"\x00" * 8)
    (out_dir / "gradients_after.bin").write_bytes(b"\x00" * 12)
    (out_dir / "optimizer_state_after.bin").write_bytes(b"\x00" * 20)

    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        runtime_sequence={
            "sequence": [
                {"command": "run_training", "args": {"steps": 1}},
                {"command": "export_gradients", "args": {}},
                {"command": "export_optimizer_state", "args": {}},
            ]
        },
    )

    package_dir = out_dir / "runtime_package"
    buffer_plan = json.loads((package_dir / "buffer_plan.json").read_text(encoding="utf-8"))
    execution_plan = json.loads((package_dir / "runtime_execution_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))

    buffers = {entry["name"]: entry for entry in buffer_plan["buffers"]}
    assert {"input", "output", "labels", "gradients_mem", "optimizer_state_mem"}.issubset(buffers)
    assert buffers["input"]["direction"] == "ps_to_pl"
    assert buffers["output"]["direction"] == "pl_to_ps"
    assert buffers["gradients_mem"]["required_for_modes"] == [8]
    assert buffers["optimizer_state_mem"]["required_for_modes"] == [9]
    assert buffers["input"]["words"] == 4
    assert buffers["output"]["words"] == 2

    seq = execution_plan["sequence"]
    assert seq[0]["command"] == "run_training"
    assert seq[0]["mode"] == 2
    assert seq[0]["sync_before"] == ["input", "labels"]
    assert seq[0]["sync_after"] == ["output"]
    assert seq[1]["capture"] == "outputs/gradients_after.bin"
    assert seq[2]["capture"] == "outputs/optimizer_state_after.bin"

    assert manifest["runtime_buffer_plan"]["buffers"] == buffer_plan["buffers"]
    assert manifest["files"]["buffer_plan"]["package_path"] == "buffer_plan.json"
    assert manifest["files"]["runtime_execution_plan"]["package_path"] == "runtime_execution_plan.json"


def test_runtime_api_allocates_binds_and_syncs_fake_pynq_buffers(tmp_path: Path) -> None:
    import importlib.util

    out_dir = tmp_path / "runtime_buffer_sync"
    out_dir.mkdir(parents=True)
    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        runtime_sequence={
            "sequence": [
                {"command": "run_training", "args": {"steps": 1}},
                {"command": "export_gradients", "args": {}},
                {"command": "export_optimizer_state", "args": {}},
            ]
        },
    )

    package_dir = out_dir / "runtime_package"
    spec = importlib.util.spec_from_file_location("runtime_api_buffer_sync_test", package_dir / "runtime_api.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeBuffer:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.sync_to_device_count = 0
            self.sync_from_device_count = 0

        def sync_to_device(self):
            self.sync_to_device_count += 1

        def sync_from_device(self):
            self.sync_from_device_count += 1

        def tobytes(self):
            return b"buffer"

    allocated: list[FakeBuffer] = []

    def fake_allocate(*, shape, dtype):
        buf = FakeBuffer(shape, dtype)
        allocated.append(buf)
        return buf

    class FakeBackend:
        def __init__(self):
            self.modes: list[int] = []
            self.bound: dict[str, FakeBuffer] = {}

        def bind_buffer(self, name, buf):
            self.bound[name] = buf

        def call_mode(self, mode: int, **kwargs):
            self.modes.append(mode)
            return {"mode": mode, "kwargs": kwargs}

        def read_buffer(self, name: str) -> bytes:
            if name == "gradients_mem":
                return b"gradients"
            if name == "optimizer_state_mem":
                return b"optimizer"
            raise KeyError(name)

    buffers = module.allocate_runtime_buffers(allocate_fn=fake_allocate)
    assert {"input", "output", "labels", "gradients_mem", "optimizer_state_mem"}.issubset(buffers)
    assert len(allocated) == len(module.load_buffer_plan()["buffers"])

    fake = FakeBackend()
    module.bind_backend(fake, buffers=buffers)
    results = module.run_sequence()

    assert fake.modes == [2, 8, 9]
    assert results[1] == b"gradients"
    assert results[2] == b"optimizer"
    assert buffers["input"].sync_to_device_count == 1
    assert buffers["labels"].sync_to_device_count == 1
    assert buffers["output"].sync_from_device_count == 1
    assert buffers["gradients_mem"].sync_from_device_count >= 1
    assert buffers["optimizer_state_mem"].sync_from_device_count >= 1
    assert (package_dir / "outputs" / "gradients_after.bin").read_bytes() == b"gradients"
    assert (package_dir / "outputs" / "optimizer_state_after.bin").read_bytes() == b"optimizer"



def test_runtime_sequence_writes_success_execution_report(tmp_path: Path) -> None:
    import importlib.util

    out_dir = tmp_path / "runtime_execution_report_success"
    out_dir.mkdir(parents=True)
    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        runtime_sequence={
            "sequence": [
                {"command": "run_training", "args": {"steps": 1}},
                {"command": "export_gradients", "args": {}},
                {"command": "export_optimizer_state", "args": {}},
            ]
        },
    )

    package_dir = out_dir / "runtime_package"
    spec = importlib.util.spec_from_file_location("runtime_api_execution_report_success", package_dir / "runtime_api.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeBuffer:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.sync_to_device_count = 0
            self.sync_from_device_count = 0

        def sync_to_device(self):
            self.sync_to_device_count += 1

        def sync_from_device(self):
            self.sync_from_device_count += 1

        def tobytes(self):
            return b"buffer"

    def fake_allocate(*, shape, dtype):
        return FakeBuffer(shape, dtype)

    class FakeBackend:
        def __init__(self):
            self.modes: list[int] = []
            self.bound: dict[str, FakeBuffer] = {}

        def bind_buffer(self, name, buf):
            self.bound[name] = buf

        def call_mode(self, mode: int, **kwargs):
            self.modes.append(mode)
            return {"mode": mode}

        def read_buffer(self, name: str) -> bytes:
            if name == "gradients_mem":
                return b"gradients"
            if name == "optimizer_state_mem":
                return b"optimizer"
            raise KeyError(name)

    buffers = module.allocate_runtime_buffers(allocate_fn=fake_allocate)
    module.bind_backend(FakeBackend(), buffers=buffers)
    report = module.run_sequence(return_report=True)

    report_path = package_dir / "runtime_execution_report.json"
    report_md = package_dir / "runtime_execution_report.md"
    assert report_path.exists()
    assert report_md.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == report
    assert saved["status"] == "passed"
    assert saved["backend"]["type"] == "FakeBackend"
    assert [entry["command"] for entry in saved["sequence"]] == [
        "run_training",
        "export_gradients",
        "export_optimizer_state",
    ]
    assert [entry["status"] for entry in saved["sequence"]] == ["passed", "passed", "passed"]
    assert saved["sequence"][0]["sync_before"] == ["input", "labels"]
    assert saved["sequence"][0]["sync_after"] == ["output"]
    assert saved["sequence"][0]["latency_ms"] >= 0
    assert {cap["path"] for cap in saved["captures"]} == {
        "outputs/gradients_after.bin",
        "outputs/optimizer_state_after.bin",
    }
    assert (package_dir / "outputs" / "gradients_after.bin").read_bytes() == b"gradients"
    assert (package_dir / "outputs" / "optimizer_state_after.bin").read_bytes() == b"optimizer"
    md = report_md.read_text(encoding="utf-8")
    assert "FPGAI Runtime Execution Report" in md
    assert "run_training" in md
    assert "export_gradients" in md


def test_runtime_sequence_failure_writes_report_before_raising(tmp_path: Path) -> None:
    import importlib.util
    import pytest

    out_dir = tmp_path / "runtime_execution_report_failure"
    out_dir.mkdir(parents=True)
    emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        runtime_sequence={"sequence": [{"command": "run_training", "args": {"steps": 1}}]},
    )

    package_dir = out_dir / "runtime_package"
    spec = importlib.util.spec_from_file_location("runtime_api_execution_report_failure", package_dir / "runtime_api.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeBackend:
        def bind_buffer(self, name, buf):
            pass

        def call_mode(self, mode: int, **kwargs):
            raise RuntimeError("synthetic board failure")

    module.bind_backend(FakeBackend(), buffers={})
    with pytest.raises(RuntimeError, match="synthetic board failure"):
        module.run_sequence(strict=True)

    report = json.loads((package_dir / "runtime_execution_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["sequence"][0]["command"] == "run_training"
    assert report["sequence"][0]["status"] == "failed"
    assert "synthetic board failure" in report["sequence"][0]["error"]
    assert report["errors"][0]["type"] == "RuntimeError"
    assert (package_dir / "runtime_execution_report.md").exists()

    report2 = module.run_sequence(strict=False, return_report=True)
    assert report2["status"] == "failed"
    assert report2["errors"][0]["command"] == "run_training"


def _write_minimal_movement_validation_artifacts(out_dir: Path, *, source: str, runtime_io: dict, dma_count: int = 0, m_axi_bundles: int = 0, buffers: list[dict] | None = None) -> None:
    (out_dir / "hls/src").mkdir(parents=True, exist_ok=True)
    (out_dir / "hls/src/deeplearn.cpp").write_text(source, encoding="utf-8")
    package = out_dir / "runtime_package"
    package.mkdir(parents=True, exist_ok=True)
    buffer_entries = buffers if buffers is not None else [
        {"name": "input", "role": "model_input"},
        {"name": "output", "role": "model_output"},
    ]
    (package / "buffer_plan.json").write_text(
        json.dumps({"buffers": buffer_entries}),
        encoding="utf-8",
    )
    (package / "package_manifest.json").write_text(json.dumps({"runtime_io": runtime_io}), encoding="utf-8")
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "vivado_bd_validation.json").write_text(json.dumps({"status": "not_requested"}), encoding="utf-8")
    (reports / "board_fit.json").write_text(
        json.dumps(
            {
                "derived_requirements": {
                    "interface_requirements": {
                        "dma_count": dma_count,
                        "m_axi_bundles": m_axi_bundles,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_movement_contract_validates_direct_m_axi_input_output_config(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_data_movement_reports, emit_movement_contract_validation

    out_dir = tmp_path / "direct_m_axi"
    out_dir.mkdir()
    _write_minimal_movement_validation_artifacts(
        out_dir,
        source="""
void deeplearn(const ap_uint<32>* input_mem, ap_uint<32>* output_mem) {
#pragma HLS INTERFACE m_axi port=input_mem offset=slave bundle=gmem_input
#pragma HLS INTERFACE m_axi port=output_mem offset=slave bundle=gmem_output
  static const int FPGAI_INPUT_TILE_SIZE = 64;
  static const int FPGAI_OUTPUT_TILE_SIZE = 64;
  // m_axi tiled input import: input_mem -> input_tile -> layer_in.
  // m_axi tiled output export: layer_out -> output_tile -> output_mem.
}
""",
        runtime_io={
            "inputs": {"import": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "resolved": "m_axi_import_tiled"}},
            "outputs": {"export": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "resolved": "m_axi_export_tiled"}},
        },
        m_axi_bundles=2,
    )

    emit_data_movement_reports(
        out_dir,
        raw_config={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
            },
            "runtime": {"sequence": ["run_inference"]},
        },
        pipeline_mode="inference",
        weights_mode="embedded",
        memory_plan=None,
        communication_plan=None,
        runtime_sequence={"sequence": ["run_inference"]},
    )
    artifacts = emit_movement_contract_validation(out_dir)
    plan = json.loads((out_dir / "reports/data_movement_plan.json").read_text(encoding="utf-8"))
    validation = json.loads(Path(artifacts["movement_contract_validation_json"]).read_text(encoding="utf-8"))

    tensors = {row["role"]: row for row in plan["tensors"]}
    assert tensors["inputs"]["interface"] == "m_axi"
    assert tensors["inputs"]["tiled"] is True
    assert tensors["outputs"]["interface"] == "m_axi"
    assert validation["status"] == "passed"
    generated = {check["name"]: check for check in validation["checks"]["generated_cpp"]}
    runtime = {check["name"]: check for check in validation["checks"]["runtime_package"]}
    assert generated["cpp_input_m_axi_port_matches_plan"]["passed"] is True
    assert generated["cpp_output_m_axi_tiling_matches_plan"]["passed"] is True
    assert runtime["runtime_input_resolved_movement_matches_plan"]["passed"] is True
    assert runtime["runtime_output_resolved_movement_matches_plan"]["passed"] is True


def test_movement_contract_validates_direct_axi_stream_tiled_input_output_config(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_data_movement_reports, emit_movement_contract_validation

    out_dir = tmp_path / "direct_axis"
    out_dir.mkdir()
    _write_minimal_movement_validation_artifacts(
        out_dir,
        source="""
void deeplearn(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream) {
  static const int FPGAI_AXIS_INPUT_TILE_SIZE = 64;
  static const int FPGAI_AXIS_OUTPUT_TILE_SIZE = 64;
  axis_t packet = in_stream.read();
  packet.last = 1;
  out_stream.write(packet);
}
""",
        runtime_io={
            "inputs": {"import": {"interface": "axi_stream", "transport": "dma", "policy": "tiled", "resolved": "dma_stream_import_tiled"}},
            "outputs": {"export": {"interface": "axi_stream", "transport": "dma", "policy": "tiled", "resolved": "dma_stream_export_tiled"}},
        },
        dma_count=1,
    )

    emit_data_movement_reports(
        out_dir,
        raw_config={
            "data_movement": {
                "inputs": {"interface": "axi_stream", "transport": "dma", "policy": "tiled"},
                "outputs": {"interface": "axi_stream", "transport": "dma", "policy": "tiled"},
            },
            "runtime": {"sequence": ["run_inference"]},
        },
        pipeline_mode="inference",
        weights_mode="embedded",
        memory_plan=None,
        communication_plan=None,
        runtime_sequence={"sequence": ["run_inference"]},
    )
    artifacts = emit_movement_contract_validation(out_dir)
    transfer = json.loads((out_dir / "reports/ps_pl_transfer_plan.json").read_text(encoding="utf-8"))
    validation = json.loads(Path(artifacts["movement_contract_validation_json"]).read_text(encoding="utf-8"))

    assert transfer["requirements"]["axi_dma"] is True
    assert transfer["requirements"]["tlast"] is True
    assert validation["status"] == "passed"
    generated = {check["name"]: check for check in validation["checks"]["generated_cpp"]}
    runtime = {check["name"]: check for check in validation["checks"]["runtime_package"]}
    assert generated["cpp_input_stream_port_matches_plan"]["passed"] is True
    assert generated["cpp_output_stream_tlast_matches_plan"]["passed"] is True
    assert runtime["runtime_input_resolved_movement_matches_plan"]["passed"] is True
    assert runtime["runtime_output_resolved_movement_matches_plan"]["passed"] is True


def test_movement_contract_validates_training_auxiliary_tensor_roles(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_data_movement_reports, emit_movement_contract_validation

    out_dir = tmp_path / "training_aux_roles"
    out_dir.mkdir()
    _write_minimal_movement_validation_artifacts(
        out_dir,
        source="""
void deeplearn(ap_uint<32>* label_mem, ap_uint<32>* weights_mem, ap_uint<32>* gradients_mem, ap_uint<32>* optimizer_state_mem) {
#pragma HLS INTERFACE m_axi port=label_mem offset=slave bundle=gmem_labels
#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights
#pragma HLS INTERFACE m_axi port=gradients_mem offset=slave bundle=gmem_gradients
#pragma HLS INTERFACE m_axi port=optimizer_state_mem offset=slave bundle=gmem_optimizer_state
  // mode 1 = import_weights
  // mode 2 = export_weights
  // mode 8 = export_gradients
  // mode 9 = export_optimizer_state
  // FPGAI gradient export tiled mode
  ap_uint<32> gradient_export_tile[64];
}
""",
        runtime_io={
            "inputs": {"import": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_import_full"}},
            "outputs": {"export": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_export_full"}},
        },
        buffers=[
            {"name": "input", "role": "model_input"},
            {"name": "output", "role": "model_output"},
            {"name": "labels", "role": "training_labels"},
            {"name": "weights", "role": "weight_import"},
            {"name": "gradients_mem", "role": "gradient_export"},
            {"name": "optimizer_state_mem", "role": "optimizer_state_export"},
        ],
        m_axi_bundles=4,
    )

    emit_data_movement_reports(
        out_dir,
        raw_config={
            "data_movement": {
                "labels": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                "gradients": {"export": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}},
                "optimizer_state": {"export": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"}},
            },
            "runtime": {"sequence": ["import_weights", "run_training", "export_weights", "export_gradients", "export_optimizer_state"]},
        },
        pipeline_mode="training_on_device",
        weights_mode="import_export",
        memory_plan=None,
        communication_plan=None,
        runtime_sequence={"sequence": ["import_weights", "run_training", "export_weights", "export_gradients", "export_optimizer_state"]},
    )
    artifacts = emit_movement_contract_validation(out_dir)
    validation = json.loads(Path(artifacts["movement_contract_validation_json"]).read_text(encoding="utf-8"))

    assert validation["status"] == "passed"
    generated = {check["name"]: check for check in validation["checks"]["generated_cpp"]}
    runtime = {check["name"]: check for check in validation["checks"]["runtime_package"]}
    assert generated["cpp_labels_m_axi_port_matches_plan"]["passed"] is True
    assert generated["cpp_weight_m_axi_port_matches_plan"]["passed"] is True
    assert generated["cpp_gradient_m_axi_port_matches_plan"]["passed"] is True
    assert generated["cpp_optimizer_state_m_axi_port_matches_plan"]["passed"] is True
    assert runtime["runtime_labels_buffer_matches_plan"]["passed"] is True
    assert runtime["runtime_weights_buffer_matches_plan"]["passed"] is True
    assert runtime["runtime_gradients_buffer_matches_plan"]["passed"] is True
    assert runtime["runtime_optimizer_state_buffer_matches_plan"]["passed"] is True


def test_movement_contract_rejects_unrequested_training_export_paths(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_data_movement_reports, emit_movement_contract_validation

    out_dir = tmp_path / "unrequested_exports"
    out_dir.mkdir()
    _write_minimal_movement_validation_artifacts(
        out_dir,
        source="""
void deeplearn(const float* input_mem, float* output_mem) {
#pragma HLS INTERFACE m_axi port=input_mem offset=slave bundle=gmem_input
#pragma HLS INTERFACE m_axi port=output_mem offset=slave bundle=gmem_output
}
""",
        runtime_io={
            "inputs": {"import": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full", "resolved": "m_axi_import_full"}},
            "outputs": {"export": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full", "resolved": "m_axi_export_full"}},
        },
        buffers=[
            {"name": "input", "role": "model_input"},
            {"name": "output", "role": "model_output"},
        ],
        m_axi_bundles=2,
    )

    emit_data_movement_reports(
        out_dir,
        raw_config={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime"},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime"},
            },
            "runtime": {"sequence": ["run_inference"]},
        },
        pipeline_mode="inference",
        weights_mode="embedded",
        memory_plan=None,
        communication_plan=None,
        runtime_sequence={"sequence": ["run_inference"]},
    )
    artifacts = emit_movement_contract_validation(out_dir)
    validation = json.loads(Path(artifacts["movement_contract_validation_json"]).read_text(encoding="utf-8"))

    assert validation["status"] == "passed"
    generated = {check["name"]: check for check in validation["checks"]["generated_cpp"]}
    runtime = {check["name"]: check for check in validation["checks"]["runtime_package"]}
    assert generated["cpp_gradient_export_path_absent_when_not_requested"]["passed"] is True
    assert generated["cpp_optimizer_state_export_path_absent_when_not_requested"]["passed"] is True
    assert runtime["runtime_gradients_buffer_matches_plan"]["passed"] is True
    assert runtime["runtime_optimizer_state_buffer_matches_plan"]["passed"] is True


def test_direct_data_movement_schema_materializes_into_communication_plan() -> None:
    from types import SimpleNamespace

    from fpgai.engine.communication import make_communication_plan
    from fpgai.engine.models import MemoryPlan

    raw = {
        "data_movement": {
            "inputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
            "outputs": {"interface": "axi_stream", "transport": "dma", "policy": "full"},
            "labels": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full", "enabled": True},
        }
    }
    plan = make_communication_plan(SimpleNamespace(raw=raw), MemoryPlan(notes={"policy_name": "Balanced"}))
    by_kind = {edge.notes.get("kind"): edge for edge in plan.edges}

    assert by_kind["input"].notes["interface"] == "m_axi"
    assert by_kind["input"].notes["transport"] == "ps_runtime"
    assert by_kind["input"].notes["policy"] == "tiled"
    assert by_kind["input"].notes["mode"] == "ddr"
    assert by_kind["output"].notes["interface"] == "axi_stream"
    assert by_kind["output"].notes["transport"] == "dma"
    assert by_kind["output"].notes["mode"] == "stream"
    assert by_kind["aux"].notes["interface"] == "m_axi"


def test_direct_data_movement_schema_materializes_into_hls_io_resolver() -> None:
    from fpgai.backends.hls.emit.top_cpp import _fpgai_io_movement_kind
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_training_movement

    raw = {
        "data_movement": {
            "inputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
            "outputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
            "labels": {"interface": "axi_stream", "transport": "dma", "policy": "tiled"},
            "gradients": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
        }
    }

    assert _fpgai_io_movement_kind({"raw_cfg": raw}, "input") == "m_axi_full"
    assert _fpgai_io_movement_kind({"raw_cfg": raw}, "output") == "m_axi_tiled"
    assert _fpgai_training_movement(raw, "labels", "import") == {
        "interface": "axi_stream",
        "transport": "dma",
        "policy": "tiled",
    }
    assert _fpgai_training_movement(raw, "gradients", "export") == {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "full",
    }


def test_movement_contract_failure_is_loud_in_report_and_summary(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import (
        emit_data_movement_reports,
        emit_movement_contract_validation,
        movement_contract_validation_summary,
    )

    out_dir = tmp_path / "movement_contract_failure"
    out_dir.mkdir()
    _write_minimal_movement_validation_artifacts(
        out_dir,
        source="""
void deeplearn(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream) {
  axis_t packet = in_stream.read();
  out_stream.write(packet);
}
""",
        runtime_io={
            "inputs": {"import": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_import_full"}},
            "outputs": {"export": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_export_full"}},
        },
        dma_count=1,
    )

    emit_data_movement_reports(
        out_dir,
        raw_config={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
            },
            "runtime": {"sequence": ["run_inference"]},
        },
        pipeline_mode="inference",
        weights_mode="embedded",
        memory_plan=None,
        communication_plan=None,
        runtime_sequence={"sequence": ["run_inference"]},
    )
    artifacts = emit_movement_contract_validation(out_dir)
    validation = json.loads(Path(artifacts["movement_contract_validation_json"]).read_text(encoding="utf-8"))
    summary = movement_contract_validation_summary(out_dir)
    md = Path(artifacts["movement_contract_validation_md"]).read_text(encoding="utf-8")

    assert validation["schema_version"] == 2
    assert validation["status"] == "failed"
    assert validation["passed"] is False
    assert validation["blocking_failure"] is True
    assert validation["summary"]["failed_checks"] >= 1
    assert validation["failed_checks"]
    assert summary["status"] == "failed"
    assert summary["passed"] is False
    assert summary["blocking_failure"] is True
    assert summary["failed_checks"] == validation["summary"]["failed_checks"]
    assert "Blocking failures" in md
    assert "**FAIL**" in md


def test_tiled_mapping_disabled_does_not_create_tiled_movement_report(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_data_movement_reports

    out_dir = tmp_path / "tiled_mapping_disabled"
    out_dir.mkdir()
    emit_data_movement_reports(
        out_dir,
        raw_config={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False, "tile_size": 32}},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
            },
            "runtime": {"sequence": ["run_inference"]},
        },
        pipeline_mode="inference",
        weights_mode="embedded",
        memory_plan=None,
        communication_plan=None,
        runtime_sequence={"sequence": ["run_inference"]},
    )

    plan = json.loads((out_dir / "reports/data_movement_plan.json").read_text(encoding="utf-8"))
    transfer = json.loads((out_dir / "reports/ps_pl_transfer_plan.json").read_text(encoding="utf-8"))
    tensors = {row["role"]: row for row in plan["tensors"]}
    assert tensors["inputs"]["tiled"] is False
    assert tensors["inputs"]["policy"] == "full"
    assert tensors["inputs"]["tile_size"] is None
    assert transfer["requirements"]["ddr_tiled"] is False


def test_tiled_mapping_enabled_records_tile_size_in_plan_runtime_and_communication(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from fpgai.engine.communication import make_communication_plan
    from fpgai.engine.models import MemoryPlan
    from fpgai.reporting.data_movement import emit_data_movement_reports, emit_movement_contract_validation
    from fpgai.runtime.package import emit_runtime_package

    raw = {
        "data_movement": {
            "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 32}},
            "outputs": {"interface": "axi_stream", "transport": "dma", "tiled": {"enabled": True, "tile_size": 16}},
        },
        "runtime": {"sequence": ["run_inference"]},
    }
    comm = make_communication_plan(SimpleNamespace(raw=raw), MemoryPlan(notes={"policy_name": "Balanced"}))
    by_kind = {edge.notes.get("kind"): edge for edge in comm.edges}
    assert by_kind["input"].notes["policy"] == "tiled"
    assert by_kind["input"].notes["tile_size"] == 32
    assert by_kind["output"].notes["tile_size"] == 16

    out_dir = tmp_path / "tiled_mapping_enabled"
    out_dir.mkdir()
    emit_runtime_package(out_dir, board="kv260", pipeline_mode="inference", top_name="deeplearn", communication_plan=comm)
    _write_minimal_movement_validation_artifacts(
        out_dir,
        source="""
void deeplearn(const ap_uint<32>* input_mem, hls::stream<axis_t>& out_stream) {
#pragma HLS INTERFACE m_axi port=input_mem offset=slave bundle=gmem_input
  static const int FPGAI_INPUT_TILE_SIZE = 32;
  static const int FPGAI_AXIS_OUTPUT_TILE_SIZE = 16;
  // m_axi tiled input import: input_mem -> input_tile -> layer_in.
  axis_t packet;
  packet.last = 1;
  out_stream.write(packet);
}
""",
        runtime_io=json.loads((out_dir / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))["runtime_io"],
        dma_count=1,
        m_axi_bundles=1,
    )
    emit_data_movement_reports(
        out_dir,
        raw_config=raw,
        pipeline_mode="inference",
        weights_mode="embedded",
        memory_plan=None,
        communication_plan=comm,
        runtime_sequence={"sequence": ["run_inference"]},
    )
    artifacts = emit_movement_contract_validation(out_dir)
    plan = json.loads((out_dir / "reports/data_movement_plan.json").read_text(encoding="utf-8"))
    transfer = json.loads((out_dir / "reports/ps_pl_transfer_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads(Path(artifacts["movement_contract_validation_json"]).read_text(encoding="utf-8"))

    tensors = {row["role"]: row for row in plan["tensors"]}
    assert tensors["inputs"]["tiled"] is True
    assert tensors["inputs"]["tile_size"] == 32
    assert tensors["outputs"]["tile_size"] == 16
    assert transfer["requirements"]["tiled"] is True
    assert transfer["requirements"]["ddr_tiled"] is True
    assert transfer["requirements"]["tiled_transfers"] == ["inputs", "outputs"]
    assert manifest["runtime_io"]["inputs"]["import"]["tile_size"] == 32
    assert manifest["runtime_io"]["outputs"]["export"]["tile_size"] == 16
    assert validation["status"] == "passed"
    generated = {check["name"]: check for check in validation["checks"]["generated_cpp"]}
    assert generated["cpp_input_m_axi_tile_size_matches_plan"]["passed"] is True
    assert generated["cpp_output_axis_tile_size_matches_plan"]["passed"] is True


def test_movement_contract_fails_on_tiled_tile_size_mismatch(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_data_movement_reports, emit_movement_contract_validation

    out_dir = tmp_path / "tile_size_mismatch"
    out_dir.mkdir()
    _write_minimal_movement_validation_artifacts(
        out_dir,
        source="""
void deeplearn(const ap_uint<32>* input_mem, ap_uint<32>* output_mem) {
#pragma HLS INTERFACE m_axi port=input_mem offset=slave bundle=gmem_input
#pragma HLS INTERFACE m_axi port=output_mem offset=slave bundle=gmem_output
  static const int FPGAI_INPUT_TILE_SIZE = 64;
  // m_axi tiled input import: input_mem -> input_tile -> layer_in.
}
""",
        runtime_io={
            "inputs": {"import": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "resolved": "m_axi_import_tiled", "tiled": True, "tile_size": 32}},
            "outputs": {"export": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full", "resolved": "m_axi_export_full"}},
        },
        m_axi_bundles=2,
    )
    emit_data_movement_reports(
        out_dir,
        raw_config={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "tile_size": 32},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
            },
            "runtime": {"sequence": ["run_inference"]},
        },
        pipeline_mode="inference",
        weights_mode="embedded",
        memory_plan=None,
        communication_plan=None,
        runtime_sequence={"sequence": ["run_inference"]},
    )
    artifacts = emit_movement_contract_validation(out_dir)
    validation = json.loads(Path(artifacts["movement_contract_validation_json"]).read_text(encoding="utf-8"))

    assert validation["status"] == "failed"
    assert "cpp_input_m_axi_tile_size_matches_plan" in validation["summary"]["failed_check_names"]


def test_w0_lite_config_contract_classifies_canonical_deprecated_and_unknown_keys() -> None:
    from fpgai.config.contract import build_config_contract_report

    report = build_config_contract_report(
        {
            "pipeline": {"mode": "inference"},
            "optimization": {"pipeline": {"ii": 2}},
            "hls": {"pipeline_ii": 3},
            "data_movement": {
                "inputs": {"import": {"interface": "m_axi"}},
                "outputs": {"interface": "axi_stream"},
            },
            "made_up_section": {"foo": 1},
        }
    )

    by_path = {item["path"]: item for item in report["manual_yaml_sources"]}
    assert by_path["pipeline.mode"]["status"] == "canonical"
    assert by_path["optimization.pipeline.ii"]["status"] == "canonical"
    assert by_path["hls.pipeline_ii"]["status"] == "deprecated_alias"
    assert by_path["hls.pipeline_ii"]["replacement"] == "optimization.pipeline.ii"
    assert by_path["data_movement.inputs.import.interface"]["status"] == "deprecated_alias"
    assert by_path["data_movement.inputs.import.interface"]["replacement"] == "data_movement.inputs.interface"
    assert by_path["made_up_section"]["status"] == "unknown_top_level"
    assert report["status"] == "audit_only"
    assert report["passed"] is True
    assert report["blocking_failure"] is False


def test_w0_lite_config_contract_markdown_mentions_migration() -> None:
    from fpgai.config.contract import build_config_contract_report, render_config_contract_markdown

    report = build_config_contract_report({"training": {"batch_size": 4}})
    text = render_config_contract_markdown(report)

    assert "# Config contract audit" in text
    assert "training.batch_size" in text
    assert "training.batch.size" in text
    assert "manual YAML override > policy default > compiler default" in text


def test_w0_lite_repo_yaml_audit_scans_config_files(tmp_path: Path) -> None:
    from fpgai.config.contract import build_repo_yaml_audit_report

    root = tmp_path
    cfg_dir = root / "configs" / "examples"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "legacy.yml").write_text(
        """
pipeline:
  mode: inference
hls:
  pipeline_ii: 2
data_movement:
  inputs:
    interface: m_axi
made_up_section:
  value: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = build_repo_yaml_audit_report(root)

    assert report["artifact_kind"] == "repo_yaml_schema_audit"
    assert report["status"] == "audit_only"
    assert report["passed"] is True
    assert report["blocking_failure"] is False
    assert report["summary"]["files_scanned"] == 1
    assert report["summary"]["files_with_deprecated_aliases"] == 1
    assert report["summary"]["files_with_unknown_or_unclassified_keys"] == 1
    assert report["files_with_deprecated_aliases"] == ["configs/examples/legacy.yml"]
    assert report["files_with_unknown_or_unclassified_keys"] == ["configs/examples/legacy.yml"]
    file_report = report["files"][0]
    deprecated_paths = {item["path"] for item in file_report["deprecated_aliases"]}
    unknown_paths = {item["path"] for item in file_report["unknown_or_unclassified"]}
    assert "hls.pipeline_ii" in deprecated_paths
    assert "made_up_section" in unknown_paths


def test_w0_lite_repo_yaml_audit_markdown_summarizes_findings(tmp_path: Path) -> None:
    from fpgai.config.contract import build_repo_yaml_audit_report, render_repo_yaml_audit_markdown

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "example.yml").write_text(
        "training:\n  batch_size: 4\nunknown_root:\n  x: 1\n",
        encoding="utf-8",
    )

    text = render_repo_yaml_audit_markdown(build_repo_yaml_audit_report(tmp_path))

    assert "# Repository YAML schema audit" in text
    assert "configs/example.yml" in text
    assert "training.batch_size" in text
    assert "training.batch.size" in text
    assert "unknown_root" in text


def test_w0_lite_classifies_sweep_paper_and_legacy_yaml_groups() -> None:
    from fpgai.config.contract import classify_config_path

    assert classify_config_path("defaults.board")["status"] == "sweep_template"
    assert classify_config_path("materialize_configs.parameter_mappings.pe.path")["status"] == "sweep_template"
    assert classify_config_path("paper.claim_policy")["status"] == "paper_artifact_spec"
    assert classify_config_path("inputs.artifacts.summary")["status"] == "paper_artifact_spec"
    assert classify_config_path("memory.storage.weights")["status"] == "deprecated_alias"
    assert classify_config_path("memory.storage.weights")["replacement"] == "memory.weight_storage"
    assert classify_config_path("training.execution.epochs")["status"] == "deprecated_alias"
    assert classify_config_path("training.execution.epochs")["replacement"] == "training.batch.epochs"
    assert classify_config_path("data_movement.ps_pl.compression.enabled")["status"] == "legacy_or_internal"
    assert classify_config_path("project")["status"] == "section_container"


def test_w0_lite_repo_audit_has_migration_queue_and_no_noise_for_templates(tmp_path: Path) -> None:
    from fpgai.config.contract import build_repo_yaml_audit_report

    cfg_dir = tmp_path / "configs" / "sweeps"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "sweep.yml").write_text(
        """
name: example_sweep
defaults:
  board: kv260
materialize_configs:
  parameter_mappings:
    pe:
      path: optimization.parallel.pe
      create: true
design_points:
  - name: p1
    pe: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "legacy.yml").write_text(
        """
memory:
  storage:
    weights: bram
training:
  execution:
    epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = build_repo_yaml_audit_report(tmp_path)

    assert report["schema_version"] == 2
    assert report["summary"]["files_scanned"] == 2
    assert report["summary"]["files_with_unknown_or_unclassified_keys"] == 0
    statuses = report["summary"]["aggregate_statuses"]
    assert statuses["sweep_template"] > 0
    assert statuses["deprecated_alias"] > 0
    queue = {item["mapping"] for item in report["migration_queue"]}
    assert "memory.storage.weights -> memory.weight_storage" in queue
    assert "training.execution.epochs -> training.batch.epochs" in queue


def test_w0_lite_safe_example_configs_are_canonical_and_parseable() -> None:
    import yaml
    from fpgai.config.contract import build_config_contract_report
    from fpgai.config.loader import load_config

    example_paths = sorted(Path("examples").glob("*/*.yml"))
    assert example_paths, "Q0 examples should exist"
    production_paths = [p for p in example_paths if "reference" not in p.parts and "paper" not in p.parts]
    template_paths = [p for p in example_paths if "reference" in p.parts or "paper" in p.parts]
    assert production_paths, "Q0 production examples should exist"
    assert template_paths, "Q0 reference/template examples should exist"

    for path in template_paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), path

    for path in production_paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), path
        load_config(str(path))
        contract = build_config_contract_report(raw)
        deprecated = [
            item["path"]
            for item in contract.get("manual_yaml_sources", [])
            if item.get("status") == "deprecated_alias"
        ]
        assert deprecated == [], f"{path} still has deprecated aliases: {deprecated}"


def test_artifact_smoke_audit_detects_required_compiler_outputs(tmp_path: Path) -> None:
    from fpgai.reporting.artifact_smoke import audit_compile_artifacts, build_artifact_smoke_suite

    out_dir = tmp_path / "compile_out"
    reports = out_dir / "reports"
    hls_src = out_dir / "hls" / "src"
    runtime_pkg = out_dir / "runtime_package"
    reports.mkdir(parents=True)
    hls_src.mkdir(parents=True)
    runtime_pkg.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps({"pipeline_mode": "inference", "top_kernel_name": "deeplearn", "build_stages": {"cpp": True}}),
        encoding="utf-8",
    )
    (hls_src / "deeplearn.cpp").write_text("void deeplearn() {}\n", encoding="utf-8")
    (runtime_pkg / "package_manifest.json").write_text("{}", encoding="utf-8")
    for name in [
        "resolved_config",
        "config_contract",
        "generated_hls_explanation",
        "generated_cpp_readability",
        "generated_cpp_validation",
        "data_movement_plan",
        "ps_pl_transfer_plan",
        "board_fit",
        "hls_synthesis_report",
        "estimate_vs_hls",
        "vivado_validation_report",
        "vivado_bd_validation",
        "vivado_implementation_report",
        "bitstream_report",
    ]:
        (reports / f"{name}.json").write_text(json.dumps({"status": "not_requested"}), encoding="utf-8")
    (reports / "movement_contract_validation.json").write_text(
        json.dumps({"status": "passed", "passed": True, "blocking_failure": False}),
        encoding="utf-8",
    )

    audit = audit_compile_artifacts(out_dir)
    assert audit["status"] == "passed"
    assert audit["evidence_levels"]["compiler_estimated"] is True
    assert audit["evidence_levels"]["hls_truth"] is False
    suite = build_artifact_smoke_suite([out_dir])
    assert suite["passed"] is True
    assert suite["summary"]["runs"] == 1


def test_q0_selected_examples_compile_and_emit_artifact_smoke_reports(tmp_path: Path) -> None:
    import pytest
    import yaml

    pytest.importorskip("onnx")
    from fpgai.config.loader import load_config
    from fpgai.engine.compiler import Compiler
    from fpgai.reporting.artifact_smoke import audit_compile_artifacts, build_artifact_smoke_suite

    selected = [
        Path("examples/inference/mnist_mlp_embedded.yml"),
        Path("examples/inference/mnist_mlp_import_weights.yml"),
    ]
    out_dirs = []
    for index, path in enumerate(selected):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw.setdefault("project", {})["out_dir"] = str(tmp_path / f"example_{index}")
        cfg_path = tmp_path / f"example_{index}.yml"
        cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        result = Compiler(load_config(str(cfg_path))).compile()
        out_dir = Path(result.out_dir)
        out_dirs.append(out_dir)
        audit = audit_compile_artifacts(out_dir)
        assert audit["passed"] is True
        assert audit["artifacts"]["manifest"]["exists"] is True
        assert audit["artifacts"]["runtime_package"]["exists"] is True
        assert audit["reports"]["data_movement_plan"]["exists"] is True
        assert audit["reports"]["movement_contract_validation"]["blocking_failure"] is False
        assert audit["evidence_levels"]["compiler_estimated"] is True

    suite = build_artifact_smoke_suite(out_dirs)
    assert suite["passed"] is True
    assert suite["summary"]["passed"] == len(selected)



def test_q0_batch2_expected_example_suite_exists_and_is_canonical() -> None:
    import yaml
    from fpgai.config.contract import build_config_contract_report
    from fpgai.config.loader import load_config

    expected_compile_configs = {
        "examples/inference/mnist_mlp_embedded.yml",
        "examples/inference/mnist_mlp_import_weights.yml",
        "examples/inference/cnn_stream_input.yml",
        "examples/inference/cnn_m_axi_input.yml",
        "examples/training/mnist_mlp_training_sgd.yml",
        "examples/training/mnist_mlp_training_momentum.yml",
        "examples/training/mnist_mlp_training_adam.yml",
        "examples/training/mnist_mlp_cross_entropy.yml",
        "examples/training/batch_accumulation.yml",
        "examples/training/tiled_training_m_axi.yml",
        "examples/training/tiled_training_axi_stream.yml",
        "examples/boards/pynq_z2_inference.yml",
        "examples/boards/kv260_inference.yml",
        "examples/boards/kr260_training.yml",
        "examples/build/cpp_only.yml",
        "examples/build/hls_project.yml",
        "examples/build/hls_synthesis.yml",
        "examples/build/vivado_project.yml",
        "examples/build/vivado_bitstream.yml",
    }
    expected_templates = {
        "examples/reference/full_options_reference.yml",
        "examples/paper/precision_sweep.yml",
        "examples/paper/memory_strategy_sweep.yml",
        "examples/paper/pipeline_parallel_sweep.yml",
    }

    for rel in sorted(expected_compile_configs):
        path = Path(rel)
        assert path.exists(), rel
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), rel
        load_config(rel)
        contract = build_config_contract_report(raw)
        deprecated = [
            item["path"]
            for item in contract.get("manual_yaml_sources", [])
            if item.get("status") == "deprecated_alias"
        ]
        assert deprecated == [], f"{rel} still has deprecated aliases: {deprecated}"

    for rel in sorted(expected_templates):
        path = Path(rel)
        assert path.exists(), rel
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), rel



def test_q0_non_tiled_training_examples_match_stream_training_backend() -> None:
    import yaml

    non_tiled_training = [
        Path("examples/training/mnist_mlp_training_sgd.yml"),
        Path("examples/training/mnist_mlp_training_momentum.yml"),
        Path("examples/training/mnist_mlp_training_adam.yml"),
        Path("examples/training/mnist_mlp_cross_entropy.yml"),
        Path("examples/training/batch_accumulation.yml"),
        Path("examples/boards/kr260_training.yml"),
    ]
    for path in non_tiled_training:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        movement = raw.get("data_movement", {})
        for role in ("inputs", "labels", "outputs"):
            role_cfg = movement.get(role, {})
            assert role_cfg.get("interface") == "axi_stream", f"{path}:{role}"
            assert role_cfg.get("transport") == "dma", f"{path}:{role}"
            assert (role_cfg.get("tiled") or {}).get("enabled") is False, f"{path}:{role}"


def test_training_compile_path_emits_truth_status_reports_even_when_not_requested(tmp_path: Path) -> None:
    from fpgai.reporting.artifact_smoke import audit_compile_artifacts

    out_dir = tmp_path / "training_truth_reports"
    reports = out_dir / "reports"
    hls_src = out_dir / "hls" / "src"
    runtime_pkg = out_dir / "runtime_package"
    reports.mkdir(parents=True)
    hls_src.mkdir(parents=True)
    runtime_pkg.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline_mode": "training_on_device",
                "top_kernel_name": "deeplearn",
                "build_stages": {"cpp": True, "runtime_package": True, "hls_synthesis": False, "vivado_project": False},
            }
        ),
        encoding="utf-8",
    )
    (hls_src / "deeplearn.cpp").write_text("void deeplearn() {}\n", encoding="utf-8")
    (runtime_pkg / "package_manifest.json").write_text("{}", encoding="utf-8")
    for name in [
        "resolved_config",
        "config_contract",
        "generated_hls_explanation",
        "generated_cpp_readability",
        "generated_cpp_validation",
        "data_movement_plan",
        "ps_pl_transfer_plan",
        "board_fit",
        "hls_synthesis_report",
        "estimate_vs_hls",
        "vivado_validation_report",
        "vivado_bd_validation",
        "vivado_implementation_report",
        "bitstream_report",
    ]:
        (reports / f"{name}.json").write_text(json.dumps({"status": "not_requested"}), encoding="utf-8")
    (reports / "movement_contract_validation.json").write_text(
        json.dumps({"status": "passed", "passed": True, "blocking_failure": False}),
        encoding="utf-8",
    )
    audit = audit_compile_artifacts(out_dir)
    assert audit["passed"] is True
    assert audit["reports"]["hls_synthesis_report"]["exists"] is True
    assert audit["reports"]["vivado_validation_report"]["exists"] is True
    assert audit["reports"]["bitstream_report"]["exists"] is True

def test_artifact_smoke_requires_vivado_tcl_when_vivado_project_requested(tmp_path: Path) -> None:
    from fpgai.reporting.artifact_smoke import audit_compile_artifacts

    out_dir = tmp_path / "vivado_requested"
    reports = out_dir / "reports"
    hls_src = out_dir / "hls" / "src"
    runtime_pkg = out_dir / "runtime_package"
    reports.mkdir(parents=True)
    hls_src.mkdir(parents=True)
    runtime_pkg.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline_mode": "inference",
                "top_kernel_name": "deeplearn",
                "build_stages": {"cpp": True, "runtime_package": True, "vivado_project": True},
            }
        ),
        encoding="utf-8",
    )
    (hls_src / "deeplearn.cpp").write_text("void deeplearn() {}\n", encoding="utf-8")
    (runtime_pkg / "package_manifest.json").write_text("{}", encoding="utf-8")
    for name in [
        "resolved_config",
        "config_contract",
        "generated_hls_explanation",
        "generated_cpp_readability",
        "generated_cpp_validation",
        "data_movement_plan",
        "ps_pl_transfer_plan",
        "board_fit",
        "hls_synthesis_report",
        "estimate_vs_hls",
        "vivado_validation_report",
        "vivado_bd_validation",
        "vivado_implementation_report",
        "bitstream_report",
    ]:
        (reports / f"{name}.json").write_text(json.dumps({"status": "not_requested"}), encoding="utf-8")
    (reports / "movement_contract_validation.json").write_text(
        json.dumps({"status": "passed", "passed": True, "blocking_failure": False}),
        encoding="utf-8",
    )

    missing = audit_compile_artifacts(out_dir)
    assert missing["passed"] is False
    assert "vivado_project_tcl" in missing["blocking_failures"]
    assert "vivado_bd_tcl" in missing["blocking_failures"]

    vivado = out_dir / "vivado"
    vivado.mkdir()
    (vivado / "project.tcl").write_text("# project\n", encoding="utf-8")
    (vivado / "bd.tcl").write_text("# bd\n", encoding="utf-8")

    present = audit_compile_artifacts(out_dir)
    assert present["passed"] is True
    assert present["manifest_summary"]["vivado_project_requested"] is True



def test_movement_contract_ignores_helper_stream_tokens_for_m_axi_top(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_movement_contract_validation

    out_dir = tmp_path / "m_axi_with_helper_stream_tokens"
    reports = out_dir / "reports"
    hls_src = out_dir / "hls" / "src"
    runtime_pkg = out_dir / "runtime_package"
    reports.mkdir(parents=True)
    hls_src.mkdir(parents=True)
    runtime_pkg.mkdir(parents=True)

    (hls_src / "deeplearn.cpp").write_text(
        "\n".join(
            [
                "#include <hls_stream.h>",
                "struct axis_t { unsigned int data; int keep; int strb; int last; };",
                "static inline float read_helper(hls::stream<axis_t>& in) { return 0.0f; }",
                "static inline void write_helper(hls::stream<axis_t>& out) { axis_t packet; out.write(packet); }",
                "void deeplearn(",
                "  const ap_uint<32>* input_mem,",
                "  ap_uint<32>* output_mem,",
                "  int mode) {",
                "#pragma HLS INTERFACE m_axi port=input_mem offset=slave bundle=gmem_input",
                "#pragma HLS INTERFACE m_axi port=output_mem offset=slave bundle=gmem_output",
                "  output_mem[0] = input_mem[0];",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (hls_src / "tb.cpp").write_text(
        "\n".join(
            [
                "#include <hls_stream.h>",
                "struct axis_t { unsigned int data; int keep; int strb; int last; };",
                "extern \"C\" void deeplearn(",
                "  hls::stream<axis_t>& in_stream,",
                "  hls::stream<axis_t>& out_stream,",
                "  int mode);",
            ]
        ),
        encoding="utf-8",
    )

    (reports / "data_movement_plan.json").write_text(
        json.dumps(
            {
                "tensors": [
                    {"role": "inputs", "requested": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                    {"role": "outputs", "requested": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                    {"role": "weights", "requested": False, "import_requested": False, "export_requested": False},
                    {"role": "gradients", "requested": False, "export_requested": False},
                    {"role": "optimizer_state", "requested": False, "export_requested": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    (reports / "ps_pl_transfer_plan.json").write_text(json.dumps({"requirements": {"axi_dma": False, "m_axi": True}}), encoding="utf-8")
    (reports / "vivado_bd_validation.json").write_text(json.dumps({"status": "not_requested"}), encoding="utf-8")
    (reports / "board_fit.json").write_text(json.dumps({"derived_requirements": {"interface_requirements": {"dma_count": 0, "m_axi_bundles": 2}}}), encoding="utf-8")
    (runtime_pkg / "buffer_plan.json").write_text(json.dumps({"buffers": []}), encoding="utf-8")
    (runtime_pkg / "package_manifest.json").write_text(
        json.dumps(
            {
                "runtime_io": {
                    "inputs": {"import": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full", "resolved": "m_axi_import_full"}},
                    "outputs": {"export": {"interface": "m_axi", "transport": "ps_runtime", "policy": "full", "resolved": "m_axi_export_full"}},
                }
            }
        ),
        encoding="utf-8",
    )

    result = emit_movement_contract_validation(out_dir)
    assert result["movement_contract_validation_status"] == "passed"
    payload = json.loads((reports / "movement_contract_validation.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["summary"]["failed_checks"] == 0

def test_movement_contract_accepts_training_stream_port_names(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_movement_contract_validation

    out_dir = tmp_path / "training_stream_contract"
    reports = out_dir / "reports"
    hls_src = out_dir / "hls" / "src"
    runtime_pkg = out_dir / "runtime_package"
    reports.mkdir(parents=True)
    hls_src.mkdir(parents=True)
    runtime_pkg.mkdir(parents=True)

    (hls_src / "deeplearn.cpp").write_text(
        "\n".join(
            [
                "#include <hls_stream.h>",
                "struct axis_t { unsigned int data; int keep; int strb; int last; };",
                "static inline float read_f32(hls::stream<axis_t>& stream) { return 0.0f; }",
                "static inline void write_f32(hls::stream<axis_t>& out, float value, bool last=false) { axis_t packet; packet.last=last?1:0; out.write(packet); }",
                "void deeplearn(",
                "  hls::stream<axis_t>& in,",
                "  hls::stream<axis_t>& out,",
                "  hls::stream<axis_t>& aux,",
                "  ap_uint<32>* weights_mem,",
                "  ap_uint<32>* gradients_mem,",
                "  int mode) {",
                "#pragma HLS INTERFACE axis port=in",
                "#pragma HLS INTERFACE axis port=out",
                "#pragma HLS INTERFACE axis port=aux",
                "#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights",
                "#pragma HLS INTERFACE m_axi port=gradients_mem offset=slave bundle=gmem_gradients",
                "  static const int FPGAI_MODE_EXPORT_GRADIENTS = 8;",
                "  float x = read_f32(in);",
                "  float y = read_f32(aux);",
                "  if (mode == FPGAI_MODE_EXPORT_GRADIENTS) gradients_mem[0] = 0;",
                "  write_f32(out, x + y, true);",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (reports / "data_movement_plan.json").write_text(
        json.dumps(
            {
                "tensors": [
                    {"role": "inputs", "requested": True, "interface": "axi_stream", "transport": "dma", "policy": "full"},
                    {"role": "outputs", "requested": True, "interface": "axi_stream", "transport": "dma", "policy": "full"},
                    {"role": "labels", "requested": True, "interface": "axi_stream", "transport": "dma", "policy": "full"},
                    {"role": "weights", "requested": True, "import_requested": True, "mutable": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                    {"role": "gradients", "requested": True, "export_requested": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                    {"role": "optimizer_state", "requested": False, "export_requested": False, "interface": "not_requested", "transport": "not_requested"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (reports / "ps_pl_transfer_plan.json").write_text(json.dumps({"requirements": {"axi_dma": False, "m_axi": False}}), encoding="utf-8")
    (reports / "vivado_bd_validation.json").write_text(json.dumps({"status": "not_requested"}), encoding="utf-8")
    (reports / "board_fit.json").write_text(json.dumps({"derived_requirements": {"interface_requirements": {"dma_count": 0, "m_axi_bundles": 0}}}), encoding="utf-8")
    (runtime_pkg / "buffer_plan.json").write_text(
        json.dumps({"buffers": [{"name": "labels"}, {"name": "weights"}, {"name": "gradients_mem"}]}),
        encoding="utf-8",
    )
    (runtime_pkg / "package_manifest.json").write_text(
        json.dumps(
            {
                "runtime_io": {
                    "inputs": {"import": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_import_full"}},
                    "outputs": {"export": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_export_full"}},
                }
            }
        ),
        encoding="utf-8",
    )

    result = emit_movement_contract_validation(out_dir)
    assert result["movement_contract_validation_status"] == "passed"
    payload = json.loads((reports / "movement_contract_validation.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["summary"]["failed_checks"] == 0


def test_movement_contract_allows_unused_weight_mode_handlers(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_movement_contract_validation

    out_dir = tmp_path / "training_unused_weight_modes"
    reports = out_dir / "reports"
    hls_src = out_dir / "hls" / "src"
    runtime_pkg = out_dir / "runtime_package"
    reports.mkdir(parents=True)
    hls_src.mkdir(parents=True)
    runtime_pkg.mkdir(parents=True)

    (hls_src / "deeplearn.cpp").write_text(
        "\n".join(
            [
                "#include <hls_stream.h>",
                "struct axis_t { unsigned int data; int keep; int strb; int last; };",
                "static inline float read_f32(hls::stream<axis_t>& stream) { return 0.0f; }",
                "static inline void write_f32(hls::stream<axis_t>& out, float value, bool last=false) { axis_t packet; packet.last=last?1:0; out.write(packet); }",
                "void deeplearn(",
                "  hls::stream<axis_t>& in,",
                "  hls::stream<axis_t>& out,",
                "  hls::stream<axis_t>& aux,",
                "  ap_uint<32>* weights_mem,",
                "  ap_uint<32>* gradients_mem,",
                "  int mode) {",
                "#pragma HLS INTERFACE axis port=in",
                "#pragma HLS INTERFACE axis port=out",
                "#pragma HLS INTERFACE axis port=aux",
                "#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights",
                "#pragma HLS INTERFACE m_axi port=gradients_mem offset=slave bundle=gmem_gradients",
                "  static const int FPGAI_MODE_IMPORT_WEIGHTS = 3;",
                "  static const int FPGAI_MODE_EXPORT_WEIGHTS = 4;",
                "  static const int FPGAI_MODE_EXPORT_GRADIENTS = 8;",
                "  if (mode == FPGAI_MODE_IMPORT_WEIGHTS) weights_mem[0] = weights_mem[0];",
                "  if (mode == FPGAI_MODE_EXPORT_WEIGHTS) weights_mem[0] = weights_mem[0];",
                "  if (mode == FPGAI_MODE_EXPORT_GRADIENTS) gradients_mem[0] = 0;",
                "  float x = read_f32(in);",
                "  float y = read_f32(aux);",
                "  write_f32(out, x + y, true);",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (reports / "data_movement_plan.json").write_text(
        json.dumps(
            {
                "tensors": [
                    {"role": "inputs", "requested": True, "interface": "axi_stream", "transport": "dma", "policy": "full"},
                    {"role": "outputs", "requested": True, "interface": "axi_stream", "transport": "dma", "policy": "full"},
                    {"role": "labels", "requested": True, "interface": "axi_stream", "transport": "dma", "policy": "full"},
                    {"role": "weights", "requested": True, "mutable": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "full", "runtime_modes": []},
                    {"role": "gradients", "requested": True, "export_requested": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                    {"role": "optimizer_state", "requested": False, "export_requested": False, "interface": "not_requested", "transport": "not_requested"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (reports / "ps_pl_transfer_plan.json").write_text(json.dumps({"requirements": {"axi_dma": False, "m_axi": True}}), encoding="utf-8")
    (reports / "vivado_bd_validation.json").write_text(json.dumps({"status": "not_requested"}), encoding="utf-8")
    (reports / "board_fit.json").write_text(json.dumps({"derived_requirements": {"interface_requirements": {"dma_count": 0, "m_axi_bundles": 2}}}), encoding="utf-8")
    (runtime_pkg / "buffer_plan.json").write_text(
        json.dumps({"buffers": [{"name": "labels"}, {"name": "weights"}, {"name": "gradients_mem"}]}),
        encoding="utf-8",
    )
    (runtime_pkg / "package_manifest.json").write_text(
        json.dumps(
            {
                "runtime_io": {
                    "inputs": {"import": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_import_full"}},
                    "outputs": {"export": {"interface": "axi_stream", "transport": "dma", "policy": "full", "resolved": "dma_stream_export_full"}},
                }
            }
        ),
        encoding="utf-8",
    )

    result = emit_movement_contract_validation(out_dir)
    assert result["movement_contract_validation_status"] == "passed"
    payload = json.loads((reports / "movement_contract_validation.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert "cpp_weight_import_mode_absent_when_not_in_sequence" not in payload["summary"].get("failed_check_names", [])
    assert "cpp_weight_export_mode_absent_when_not_in_sequence" not in payload["summary"].get("failed_check_names", [])


def test_movement_contract_accepts_training_hybrid_m_axi_tensor_paths(tmp_path: Path) -> None:
    from fpgai.reporting.data_movement import emit_movement_contract_validation

    out_dir = tmp_path / "training_hybrid_m_axi_contract"
    reports = out_dir / "reports"
    hls_src = out_dir / "hls" / "src"
    runtime_pkg = out_dir / "runtime_package"
    reports.mkdir(parents=True)
    hls_src.mkdir(parents=True)
    runtime_pkg.mkdir(parents=True)

    (hls_src / "deeplearn.cpp").write_text(
        "\n".join(
            [
                "#include <hls_stream.h>",
                "struct axis_t { unsigned int data; int keep; int strb; int last; };",
                "#define FPGAI_TRAIN_INPUT_TILE_SIZE 64",
                "#define FPGAI_TRAIN_LABEL_TILE_SIZE 64",
                "#define FPGAI_TRAIN_OUTPUT_TILE_SIZE 64",
                "void deeplearn(",
                "  hls::stream<axis_t>& in,",
                "  hls::stream<axis_t>& out,",
                "  hls::stream<axis_t>& aux,",
                "  ap_uint<32>* weights_mem,",
                "  ap_uint<32>* input_mem,",
                "  ap_uint<32>* label_mem,",
                "  ap_uint<32>* output_mem,",
                "  ap_uint<32>* gradients_mem,",
                "  int mode) {",
                "#pragma HLS INTERFACE axis port=in",
                "#pragma HLS INTERFACE axis port=out",
                "#pragma HLS INTERFACE axis port=aux",
                "#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights",
                "#pragma HLS INTERFACE m_axi port=input_mem offset=slave bundle=gmem_input",
                "#pragma HLS INTERFACE m_axi port=label_mem offset=slave bundle=gmem_labels",
                "#pragma HLS INTERFACE m_axi port=output_mem offset=slave bundle=gmem_output",
                "#pragma HLS INTERFACE m_axi port=gradients_mem offset=slave bundle=gmem_gradients",
                "  // FPGAI training tiled input import: PS DDR -> m_axi input_mem -> local activation tile.",
                "  // FPGAI training tiled label import: PS DDR -> m_axi label_mem -> local label tile.",
                "  // FPGAI training tiled output export: local output tile -> m_axi output_mem -> PS DDR.",
                "  static act_t input_tile[FPGAI_TRAIN_INPUT_TILE_SIZE];",
                "  static act_t label_tile[FPGAI_TRAIN_LABEL_TILE_SIZE];",
                "  static act_t output_tile[FPGAI_TRAIN_OUTPUT_TILE_SIZE];",
                "  static const int FPGAI_MODE_EXPORT_GRADIENTS = 8;",
                "  static grad_wgt_t gradient_export_tile[FPGAI_GRADIENT_EXPORT_TILE_SIZE];",
                "  float label = read_f32(aux);",
                "  if (mode == FPGAI_MODE_EXPORT_GRADIENTS) gradients_mem[0] = 0;",
                "  output_mem[0] = input_mem[0] + label_mem[0] + weights_mem[0];",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (reports / "data_movement_plan.json").write_text(
        json.dumps(
            {
                "tensors": [
                    {"role": "inputs", "requested": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "tile_size": 64},
                    {"role": "outputs", "requested": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "tile_size": 64},
                    {"role": "labels", "requested": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "tile_size": 64},
                    {"role": "weights", "requested": True, "import_requested": True, "mutable": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                    {"role": "gradients", "requested": True, "export_requested": True, "interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                    {"role": "optimizer_state", "requested": False, "export_requested": False, "interface": "not_requested", "transport": "not_requested"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (reports / "ps_pl_transfer_plan.json").write_text(json.dumps({"requirements": {"axi_dma": False, "m_axi": True}}), encoding="utf-8")
    (reports / "vivado_bd_validation.json").write_text(json.dumps({"status": "not_requested"}), encoding="utf-8")
    (reports / "board_fit.json").write_text(json.dumps({"derived_requirements": {"interface_requirements": {"dma_count": 0, "m_axi_bundles": 5}}}), encoding="utf-8")
    (runtime_pkg / "buffer_plan.json").write_text(
        json.dumps({"buffers": [{"name": "labels"}, {"name": "weights"}, {"name": "gradients_mem"}]}),
        encoding="utf-8",
    )
    (runtime_pkg / "package_manifest.json").write_text(
        json.dumps(
            {
                "runtime_io": {
                    "inputs": {"import": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "resolved": "m_axi_import_tiled"}},
                    "outputs": {"export": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "resolved": "m_axi_export_tiled"}},
                }
            }
        ),
        encoding="utf-8",
    )

    result = emit_movement_contract_validation(out_dir)
    assert result["movement_contract_validation_status"] == "passed"
    payload = json.loads((reports / "movement_contract_validation.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["summary"]["failed_checks"] == 0


def test_training_m_axi_tile_alias_macros_are_emitted() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_insert_training_io_and_gradient_ports

    source = "\n".join(
        [
            "using namespace fpgai;",
            "void deeplearn(",
            "  hls::stream<axis_t>& in,",
            "  hls::stream<axis_t>& out,",
            "  hls::stream<axis_t>& aux,",
            "  int mode",
            ") {",
            "#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL",
            "#pragma HLS INTERFACE s_axilite port=return bundle=CTRL",
            "  if (mode == 1) { return; }",
            "}",
        ]
    )
    cfg = {
        "data_movement": {
            "inputs": {"import": {"interface": "m_axi", "policy": "tiled", "transport": "ps_runtime"}},
            "labels": {"import": {"interface": "m_axi", "policy": "tiled", "transport": "ps_runtime"}},
            "outputs": {"export": {"interface": "m_axi", "policy": "tiled", "transport": "ps_runtime"}},
            "gradients": {"export": {"interface": "m_axi", "policy": "tiled", "transport": "ps_runtime"}},
        }
    }

    updated = _fpgai_insert_training_io_and_gradient_ports(source, raw_cfg=cfg)
    assert "#define FPGAI_TRAIN_INPUT_TILE_SIZE 64" in updated
    assert "#define FPGAI_INPUT_TILE_SIZE FPGAI_TRAIN_INPUT_TILE_SIZE" in updated
    assert "#define FPGAI_TRAIN_LABEL_TILE_SIZE 64" in updated
    assert "#define FPGAI_LABEL_TILE_SIZE FPGAI_TRAIN_LABEL_TILE_SIZE" in updated
    assert "#define FPGAI_TRAIN_OUTPUT_TILE_SIZE 64" in updated
    assert "#define FPGAI_OUTPUT_TILE_SIZE FPGAI_TRAIN_OUTPUT_TILE_SIZE" in updated


def test_runtime_package_validation_passes_for_deployable_kv260_package(tmp_path: Path) -> None:
    out_dir = tmp_path / "deployable_kv260"
    hw = out_dir / "vivado_bridge" / "bitstream"
    hls_src = out_dir / "hls" / "src"
    hw.mkdir(parents=True)
    hls_src.mkdir(parents=True)
    (out_dir / "manifest.json").write_text('{"top_kernel_name": "deeplearn"}', encoding="utf-8")
    (out_dir / "input.bin").write_bytes(b"\x00" * 16)
    (out_dir / "output.bin").write_bytes(b"\x00" * 8)
    (out_dir / "gradients_after.bin").write_bytes(b"\x00" * 12)
    (hw / "fpgai_bd_wrapper.bit").write_bytes(b"bitstream")
    (hw / "fpgai_bd_wrapper.hwh").write_text("hwh", encoding="utf-8")
    (hw / "fpgai_bd.xsa").write_bytes(b"xsa")
    (hls_src / "fpgai_params.cpp").write_text(
        """
namespace fpgai {
const op0_wgt_t W0_init[2] = {1.0, 2.0};
const op0_bias_t B0_init[1] = {0.0};
}
""",
        encoding="utf-8",
    )

    result = emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        weights_mode="ddr_tiled_mutable",
        build_stages={"vivado_project": True, "vivado_implementation": True, "bitstream": True, "runtime_package": True},
        runtime_sequence={"sequence": [{"command": "run_training", "args": {"steps": 1}}, {"command": "export_gradients", "args": {}}]},
    )

    assert result["deployable_overlay_present"] is True
    assert result["runtime_package_validation_status"] == "passed"
    assert result["runtime_package_deployability_ready"] is True

    package_dir = out_dir / "runtime_package"
    validation = json.loads((package_dir / "runtime_package_validation.json").read_text(encoding="utf-8"))
    reports_validation = json.loads((out_dir / "reports" / "runtime_package_validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))

    assert validation["status"] == "passed"
    assert validation["deployability_ready"] is True
    assert validation["board_execution_claimed"] is False
    assert reports_validation == validation
    assert manifest["runtime_package_validation"]["status"] == "passed"
    assert manifest["runtime_package_validation"]["deployability_ready"] is True
    assert "runtime_package_validation_json" in manifest["files"]


def test_runtime_package_validation_fails_missing_requested_bitstream(tmp_path: Path) -> None:
    out_dir = tmp_path / "missing_bitstream"
    out_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text('{}', encoding="utf-8")

    result = emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="training_on_device",
        top_name="deeplearn",
        build_stages={"vivado_project": True, "vivado_implementation": True, "bitstream": True, "runtime_package": True},
        runtime_sequence={"sequence": [{"command": "run_training", "args": {"steps": 1}}]},
    )

    assert result["runtime_package_validation_status"] == "failed"
    assert result["runtime_package_deployability_ready"] is False
    validation = json.loads((out_dir / "reports" / "runtime_package_validation.json").read_text(encoding="utf-8"))
    hardware_checks = {row["name"]: row for row in validation["checks"]["hardware"]}
    assert hardware_checks["hardware_bitstream_status_matches_request"]["status"] == "failed"
    assert hardware_checks["hardware_xsa_status_matches_request"]["status"] == "failed"


def test_runtime_package_validation_reports_static_boundary(tmp_path: Path) -> None:
    out_dir = tmp_path / "boundary"
    out_dir.mkdir(parents=True)
    emit_runtime_package(out_dir, board="kv260", pipeline_mode="inference", top_name="deeplearn")
    validation = json.loads((out_dir / "runtime_package" / "runtime_package_validation.json").read_text(encoding="utf-8"))
    assert validation["board_execution_claimed"] is False
    assert "does not execute on the FPGA board" in validation["truth_boundary"]
    md = (out_dir / "reports" / "runtime_package_validation.md").read_text(encoding="utf-8")
    assert "Runtime Package Validation" in md
    assert "board_execution_claimed" in md


def test_runtime_package_validation_accepts_label_mem_training_alias(tmp_path):
    from fpgai.runtime.package_validation import emit_runtime_package_validation
    import json

    root = tmp_path / "build"
    pkg = root / "runtime_package"
    pkg.mkdir(parents=True)
    (root / "reports").mkdir()

    (pkg / "README_RUNTIME.md").write_text("runtime", encoding="utf-8")
    (pkg / "runtime_api.py").write_text(
        "def load_manifest(): pass\n"
        "def load_buffer_plan(): pass\n"
        "def allocate_runtime_buffers(): pass\n"
        "def bind_backend(): pass\n"
        "def run_sequence(): pass\n",
        encoding="utf-8",
    )
    (pkg / "board_runtime.py").write_text(
        "class FPGAIBoardRuntime: pass\n"
        "class PynqDmaMmioBackend: pass\n"
        "def create_pynq_backend(): pass\n"
        "FPGAI_MODE_RUN_TRAINING = 2\n",
        encoding="utf-8",
    )
    (pkg / "buffer_plan.json").write_text(json.dumps({"buffers": [
        {"name": "input", "direction": "ps_to_pl", "words": 8, "bytes": 32},
        {"name": "label_mem", "direction": "ps_to_pl", "words": 1, "bytes": 4},
        {"name": "output", "direction": "pl_to_ps", "words": 1, "bytes": 4},
    ]}), encoding="utf-8")
    seq = {"sequence": [{"command": "run_training"}]}
    (pkg / "run_sequence.json").write_text(json.dumps(seq), encoding="utf-8")
    (pkg / "runtime_execution_plan.json").write_text(json.dumps({"sequence": [{"command": "run_training", "sync_before": ["input", "label_mem"], "sync_after": ["output"]}]}), encoding="utf-8")
    manifest = {
        "build_stages": {"bitstream": False, "vivado_implementation": False},
        "hardware": {"deployable_overlay_present": False, "bitstream": {"present": False}, "hwh": {"present": False}, "xsa": {"present": False}},
        "runtime_sequence": seq,
        "files": {},
        "runtime_api": {"functions": []},
    }
    (pkg / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = emit_runtime_package_validation(root, pkg)
    assert result["failed_count"] == 0
    assert result["status"] == "passed"


def test_runtime_package_validation_resolves_label_alias_in_execution_plan(tmp_path):
    from fpgai.runtime.package_validation import emit_runtime_package_validation
    import json

    root = tmp_path / "build_alias_resolution"
    pkg = root / "runtime_package"
    pkg.mkdir(parents=True)
    (root / "reports").mkdir()

    (pkg / "README_RUNTIME.md").write_text("runtime", encoding="utf-8")
    (pkg / "runtime_api.py").write_text(
        "def load_manifest(): pass\n"
        "def load_buffer_plan(): pass\n"
        "def allocate_runtime_buffers(): pass\n"
        "def bind_backend(): pass\n"
        "def run_sequence(): pass\n",
        encoding="utf-8",
    )
    (pkg / "board_runtime.py").write_text(
        "class FPGAIBoardRuntime: pass\n"
        "class PynqDmaMmioBackend: pass\n"
        "def create_pynq_backend(): pass\n"
        "FPGAI_MODE_RUN_TRAINING = 2\n",
        encoding="utf-8",
    )
    (pkg / "buffer_plan.json").write_text(json.dumps({"buffers": [
        {"name": "input", "direction": "ps_to_pl", "words": 8, "bytes": 32},
        {"name": "label_mem", "direction": "ps_to_pl", "words": 1, "bytes": 4},
        {"name": "output", "direction": "pl_to_ps", "words": 1, "bytes": 4},
    ]}), encoding="utf-8")
    seq = {"sequence": [{"command": "run_training"}]}
    (pkg / "run_sequence.json").write_text(json.dumps(seq), encoding="utf-8")
    (pkg / "runtime_execution_plan.json").write_text(json.dumps({
        "sequence": [{"command": "run_training", "sync_before": ["input", "labels"], "sync_after": ["output"]}]
    }), encoding="utf-8")
    manifest = {
        "build_stages": {"bitstream": False, "vivado_implementation": False},
        "hardware": {"deployable_overlay_present": False, "bitstream": {"present": False}, "hwh": {"present": False}, "xsa": {"present": False}},
        "runtime_sequence": seq,
        "files": {},
        "runtime_api": {"functions": []},
    }
    (pkg / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = emit_runtime_package_validation(root, pkg)
    validation = json.loads((pkg / "runtime_package_validation.json").read_text(encoding="utf-8"))
    step_checks = [
        row for row in validation["checks"]["buffers_and_sequence"]
        if row["name"].startswith("execution_step_buffers_resolve")
    ]
    assert result["failed_count"] == 0
    assert step_checks and step_checks[0]["status"] == "passed"
    assert step_checks[0]["actual"]["resolved"]["labels"] == "label_mem"


def test_runtime_package_validation_accepts_physical_m_axi_role_buffer_names(tmp_path):
    from fpgai.runtime.package_validation import emit_runtime_package_validation
    import json

    root = tmp_path / "physical_names"
    pkg = root / "runtime_package"
    pkg.mkdir(parents=True)
    (root / "reports").mkdir()
    (pkg / "README_RUNTIME.md").write_text("runtime", encoding="utf-8")
    (pkg / "runtime_api.py").write_text(
        "def load_manifest(): pass\n"
        "def load_buffer_plan(): pass\n"
        "def allocate_runtime_buffers(): pass\n"
        "def bind_backend(): pass\n"
        "def run_sequence(): pass\n",
        encoding="utf-8",
    )
    (pkg / "board_runtime.py").write_text(
        "class FPGAIBoardRuntime: pass\n"
        "class PynqDmaMmioBackend: pass\n"
        "def create_pynq_backend(): pass\n"
        "FPGAI_MODE_RUN_TRAINING = 2\n",
        encoding="utf-8",
    )
    (pkg / "buffer_plan.json").write_text(json.dumps({"buffers": [
        {"name": "input_mem", "direction": "ps_to_pl", "words": 8, "bytes": 32},
        {"name": "label_mem", "direction": "ps_to_pl", "words": 1, "bytes": 4},
        {"name": "output_mem", "direction": "pl_to_ps", "words": 1, "bytes": 4},
        {"name": "gradients", "direction": "pl_to_ps", "words": 4, "bytes": 16},
    ]}), encoding="utf-8")
    seq = {"sequence": [{"command": "run_training"}, {"command": "export_gradients"}]}
    (pkg / "run_sequence.json").write_text(json.dumps(seq), encoding="utf-8")
    (pkg / "runtime_execution_plan.json").write_text(json.dumps({
        "sequence": [
            {"command": "run_training", "sync_before": ["input", "labels"], "sync_after": ["output"]},
            {"command": "export_gradients", "sync_before": [], "sync_after": ["gradients_mem"]},
        ]
    }), encoding="utf-8")
    manifest = {
        "build_stages": {"bitstream": False, "vivado_implementation": False},
        "hardware": {"deployable_overlay_present": False, "bitstream": {"present": False}, "hwh": {"present": False}, "xsa": {"present": False}},
        "runtime_sequence": seq,
        "files": {},
        "runtime_api": {"functions": []},
    }
    (pkg / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = emit_runtime_package_validation(root, pkg)
    validation = json.loads((pkg / "runtime_package_validation.json").read_text(encoding="utf-8"))
    assert result["failed_count"] == 0
    assert validation["failed_checks"] == []


def test_runtime_package_validation_accepts_zero_byte_manifest_file_entry(tmp_path: Path) -> None:
    from fpgai.runtime.package_validation import emit_runtime_package_validation

    root = tmp_path / "zero_byte_manifest_entry"
    pkg = root / "runtime_package"
    logs = pkg / "hls" / "logs"
    logs.mkdir(parents=True)
    (root / "reports").mkdir(parents=True)

    (pkg / "README_RUNTIME.md").write_text("runtime", encoding="utf-8")
    (pkg / "runtime_api.py").write_text(
        "def load_manifest(): pass\n"
        "def load_buffer_plan(): pass\n"
        "def allocate_runtime_buffers(): pass\n"
        "def bind_backend(): pass\n"
        "def run_sequence(): pass\n",
        encoding="utf-8",
    )
    (pkg / "board_runtime.py").write_text(
        "class FPGAIBoardRuntime: pass\n"
        "class PynqDmaMmioBackend: pass\n"
        "def create_pynq_backend(): pass\n"
        "FPGAI_MODE_RUN_TRAINING = 2\n",
        encoding="utf-8",
    )
    (pkg / "buffer_plan.json").write_text(json.dumps({"buffers": [
        {"name": "input", "direction": "ps_to_pl", "words": 1, "bytes": 4},
        {"name": "output", "direction": "pl_to_ps", "words": 1, "bytes": 4},
    ]}), encoding="utf-8")
    (pkg / "run_sequence.json").write_text(json.dumps({"sequence": []}), encoding="utf-8")
    (pkg / "runtime_execution_plan.json").write_text(json.dumps({"sequence": []}), encoding="utf-8")
    (logs / "vitis_hls_stderr.log").write_bytes(b"")

    manifest = {
        "build_stages": {"bitstream": False, "vivado_implementation": False},
        "hardware": {"deployable_overlay_present": False, "bitstream": {"present": False}, "hwh": {"present": False}, "xsa": {"present": False}},
        "runtime_sequence": {"sequence": []},
        "files": {
            "hls_logs": [
                {"package_path": "hls/logs/vitis_hls_stderr.log", "bytes": 0},
            ],
        },
        "runtime_api": {"functions": []},
    }
    (pkg / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = emit_runtime_package_validation(root, pkg)
    validation = json.loads((pkg / "runtime_package_validation.json").read_text(encoding="utf-8"))
    file_checks = {
        row["name"]: row
        for row in validation["checks"]["package_files"]
    }

    assert file_checks["manifest_file_entry_present:hls_logs"]["status"] == "passed"
    assert result["failed_count"] == 0
