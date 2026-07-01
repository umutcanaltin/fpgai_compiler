from types import SimpleNamespace

from fpgai.engine.memory import make_memory_plan
from fpgai.engine.models import CompilePlan, LayerDescriptor, LayerPlan


def _graph():
    return SimpleNamespace(
        inputs=["x"],
        outputs=["y"],
        ops=[],
    )


def _compile_plan(weight_storage: str) -> CompilePlan:
    return CompilePlan(
        target_board="kv260",
        target_part="xck26-sfvc784-2LV-c",
        clock_mhz=100.0,
        execution_order=["dense0"],
        layer_plans=[
            LayerPlan(
                node_name="dense0",
                op_type="Dense",
                precision_mode="fixed",
                act_bits=16,
                weight_bits=16,
                tile={},
                unroll={"in": 1, "out": 1},
                pipeline_ii=1,
                weight_mode="embedded",
                activation_mode="stream",
                buffering="single",
                backend_kernel="dense",
                notes={
                    "partition_factor": 1,
                    "partition_mode": "none",
                },
            )
        ],
        global_resource_budget={},
        notes={
            "parallel_policy": "Latency-First",
            "weight_storage": weight_storage,
            "weight_region_preference": ["URAM", "BRAM", "DDR"],
            "activation_region_preference": ["BRAM", "URAM", "DDR"],
            "allow_double_buffer": False,
        },
    )


def _descriptor() -> LayerDescriptor:
    return LayerDescriptor(
        node_name="dense0",
        op_type="Dense",
        inputs=["x", "w", "b"],
        outputs=["y"],
        input_shapes=[(1, 16), (16, 4), (4,)],
        output_shapes=[(1, 4)],
        param_names=["dense0.weight", "dense0.bias"],
        param_bytes=160,
        activation_bytes_in=32,
        activation_bytes_out=8,
        macs=64,
        attrs={},
        compute_hint="compute_bound",
        backend_kernel="dense",
    )


def test_manual_uram_weight_storage_drives_memory_plan_region():
    plan = make_memory_plan(_graph(), [_descriptor()], _compile_plan("uram"))

    weight_regions = {
        placement.region
        for placement in plan.placements
        if placement.kind == "weight"
    }

    assert weight_regions == {"URAM"}


def test_manual_bram_weight_storage_drives_memory_plan_region():
    plan = make_memory_plan(_graph(), [_descriptor()], _compile_plan("bram"))

    weight_regions = {
        placement.region
        for placement in plan.placements
        if placement.kind == "weight"
    }

    assert weight_regions == {"BRAM"}


def test_memory_storage_uram_import_full_resolves_to_runtime_uram_mode() -> None:
    from fpgai.engine.planner import _choose_weight_mode

    raw = {
        "memory": {"storage": {"weights": "uram"}},
        "data_movement": {"weights": {"import": {"interface": "m_axi", "policy": "full"}}},
    }

    assert _choose_weight_mode(None, raw) == "uram"


def test_memory_storage_ddr_planner_resolves_to_real_tiled_backend() -> None:
    from fpgai.engine.compiler import Compiler
    from fpgai.engine.planner import _choose_weight_mode

    raw = {
        "memory": {"storage": {"weights": "ddr"}},
        "data_movement": {"weights": {"import": {"interface": "m_axi", "policy": "tiled"}}},
    }

    assert _choose_weight_mode(None, raw) == "ddr"
    compiler = object.__new__(Compiler)
    semantics = compiler._resolve_weight_movement_semantics(raw)
    assert compiler._resolve_hls_weights_mode(raw) == "ddr_tiled"
    assert semantics["memory_semantics_mode"] == "ddr_tiled"
    assert semantics["full_local_weight_replica"] is False
    assert semantics["tile_weight_buffer"] is True
    assert semantics["scalable_external_weight_execution"] is True


def test_memory_storage_bram_keeps_embedded_mode() -> None:
    from fpgai.engine.planner import _choose_weight_mode

    raw = {
        "memory": {"storage": {"weights": "bram"}},
        "data_movement": {"ps_pl": {"weights": {"mode": "embedded"}}},
    }

    assert _choose_weight_mode(None, raw) == "embedded"


def test_generated_uram_memory_design_has_real_uram_source_if_present() -> None:
    from pathlib import Path

    run = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_memory_uram")
    src = run / "hls/src/deeplearn.cpp"
    if not src.exists():
        return

    text = src.read_text()
    assert "Requested weights mode: uram" in text
    assert "Runtime-loaded URAM weight storage" in text
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in text
    assert "#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=uram" in text
    assert "#pragma HLS BIND_STORAGE variable=W1 type=ram_2p impl=uram" in text


def test_generated_uram_memory_design_has_nonzero_hls_uram_if_present() -> None:
    from pathlib import Path
    import xml.etree.ElementTree as ET

    p = Path(
        "paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/"
        "kv260_memory_uram/hls/fpgai_hls_proj/sol1/syn/report/deeplearn_csynth.xml"
    )
    if not p.exists():
        return

    root = ET.parse(p).getroot()
    elem = root.find("AreaEstimates/Resources/URAM")
    assert elem is not None
    assert elem.text is not None
    assert int(float(elem.text.strip())) > 0


def test_generated_ddr_memory_design_has_real_ddr_source_if_present() -> None:
    from pathlib import Path

    run = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_memory_ddr")
    src = run / "hls/src/deeplearn.cpp"
    if not src.exists():
        return

    text = src.read_text()
    assert "Requested weights mode: ddr" in text
    assert "const ap_uint<32>* weights_mem" in text
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in text
    assert "bundle=gmem_weights" in text
    assert "Runtime DDR weight load" in text
    assert "fpgai_load_ddr_vector" in text
    assert "impl=uram" not in text


def test_generated_ddr_runtime_package_has_weight_payload_if_present() -> None:
    from pathlib import Path
    import json

    run = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_memory_ddr")
    manifest_path = run / "runtime_package/package_manifest.json"
    if not manifest_path.exists():
        return

    pkg = run / "runtime_package"
    manifest = json.loads(manifest_path.read_text())
    runtime_weights = manifest["runtime_weights"]

    assert runtime_weights["weights_mode"] == "ddr"
    assert runtime_weights["required"] is True
    assert runtime_weights["present"] is True
    assert runtime_weights["total_words"] == 46
    assert (pkg / "weights/weights.bin").exists()
    assert (pkg / "weights/weight_layout.json").exists()

    layout = json.loads((pkg / "weights/weight_layout.json").read_text())
    assert layout["format"] == "packed32"
    assert layout["total_words"] == 46
    assert [entry["name"] for entry in layout["entries"]] == ["W0", "B0", "W1", "B1"]


def test_compiler_weight_mode_resolver_accepts_new_weights_load_schema() -> None:
    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)

    raw = {
        "memory": {"storage": {"weights": "bram"}},
        "data_movement": {
            "weights": {
                "import": {
                    "interface": "m_axi",
                    "transport": "ps_runtime",
                    "policy": "full",
                }
            }
        },
    }
    assert compiler._resolve_hls_weights_mode(raw) == "ddr"
    semantics = compiler._resolve_weight_movement_semantics(raw)
    assert semantics["memory_semantics_mode"] == "bram_import_full"
    assert semantics["reload_before_each_compute"] is False

    raw = {
        "memory": {"storage": {"weights": "uram"}},
        "data_movement": {
            "weights": {
                "import": {"interface": "m_axi", "policy": "full"}
            }
        },
    }
    assert compiler._resolve_hls_weights_mode(raw) == "uram"
    assert compiler._resolve_weight_movement_semantics(raw)["memory_semantics_mode"] == "uram_import_full"


def test_communication_plan_accepts_new_input_output_schema(tmp_path) -> None:
    from types import SimpleNamespace

    from fpgai.engine.communication import make_communication_plan
    from fpgai.engine.memory import MemoryPlan

    cfg = SimpleNamespace(
        raw={
            "data_movement": {
                "inputs": {
                    "import": {
                        "tensor_name": "x_new",
                        "region": "HOST",
                        "size_words": 11,
                        "interface": "axi_stream",
                        "transport": "dma",
                        "policy": "full",
                    }
                },
                "outputs": {
                    "export": {
                        "tensor_name": "y_new",
                        "region": "HOST",
                        "size_words": 7,
                        "interface": "axi_stream",
                        "transport": "dma",
                        "policy": "full",
                    }
                },
            }
        }
    )
    memory_plan = MemoryPlan(placements=[], notes={"policy_name": "Balanced"})

    plan = make_communication_plan(cfg, memory_plan)

    input_edges = [edge for edge in plan.edges if getattr(edge, "tensor_name", None) == "x_new"]
    output_edges = [edge for edge in plan.edges if getattr(edge, "tensor_name", None) == "y_new"]

    assert input_edges, [edge.__dict__ for edge in plan.edges]
    assert output_edges, [edge.__dict__ for edge in plan.edges]

    assert getattr(input_edges[0], "size_bytes") == 44
    assert getattr(output_edges[0], "size_bytes") == 28
    assert input_edges[0].notes["direction_name"] == "import"
    assert output_edges[0].notes["direction_name"] == "export"
    assert input_edges[0].notes["transport"] == "dma"
    assert output_edges[0].notes["transport"] == "dma"


def test_compiler_weight_mode_resolver_accepts_uram_static_schema() -> None:
    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)
    raw = {
        "memory": {"storage": {"weights": "uram"}},
        "data_movement": {
            "weights": {
                "import": {
                    "interface": "compile_time",
                    "transport": "none",
                    "policy": "static",
                }
            }
        },
    }

    semantics = compiler._resolve_weight_movement_semantics(raw)
    assert semantics["memory_semantics_mode"] == "uram_static"
    assert semantics["hls_weights_mode"] == "embedded"
    assert semantics["runtime_weight_payload_required"] is False


def test_top_cpp_emits_function_scope_bram_static_weight_arrays() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [
                Op(
                    name="dense0",
                    op_type="Dense",
                    inputs=["x", "W0", "B0"],
                    outputs=["y"],
                    attrs={"precision_tag": "op0"},
                )
            ]
            self.tensors = {
                "x": TensorSpec((1, 4)),
                "W0": TensorSpec((3, 4)),
                "B0": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((3, 4), dtype=np.float32),
                "B0": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "bram_static"}),
        communication_plan=None,
        raw_cfg={},
    )

    assert "FPGAI bram_static weight storage" in source
    assert "static op0_wgt_t W0[12];" in source
    assert "#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=bram" in source
    assert "W0[i] = fpgai::W0[i];" in source
    assert "dense_out_in<4, 3" in source


def test_top_cpp_emits_function_scope_uram_static_weight_arrays() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [
                Op(
                    name="dense0",
                    op_type="Dense",
                    inputs=["x", "W0", "B0"],
                    outputs=["y"],
                    attrs={"precision_tag": "op0"},
                )
            ]
            self.tensors = {
                "x": TensorSpec((1, 4)),
                "W0": TensorSpec((3, 4)),
                "B0": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((3, 4), dtype=np.float32),
                "B0": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "uram_static"}),
        communication_plan=None,
        raw_cfg={},
    )

    assert "FPGAI uram_static weight storage" in source
    assert "static op0_wgt_t W0[12];" in source
    assert "#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=uram" in source
    assert "W0[i] = fpgai::W0[i];" in source
    assert "dense_out_in<4, 3" in source


def test_top_cpp_emits_dense_ddr_tiled_without_full_weight_replica() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [
                Op(
                    name="dense0",
                    op_type="Dense",
                    inputs=["x", "W0", "B0"],
                    outputs=["y"],
                    attrs={"precision_tag": "op0"},
                )
            ]
            self.tensors = {
                "x": TensorSpec((1, 4)),
                "W0": TensorSpec((3, 4)),
                "B0": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((3, 4), dtype=np.float32),
                "B0": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="ddr_tiled",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "ddr_tiled"}),
        communication_plan=None,
        raw_cfg={},
    )

    assert "Requested weights mode: ddr_tiled" in source
    assert "const ap_uint<32>* weights_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in source
    assert "dense_out_in_ddr_tiled<4, 3, 4, 3" in source
    assert "weight_tile[TILE_OUT][TILE_IN]" in source
    assert "fpgai_load_ddr_vector" not in source
    assert "Runtime DDR weight load" not in source
    assert "static op0_wgt_t W0" not in source
    assert "static op0_bias_t B0" not in source
    assert "W0," not in source
    assert "B0);" not in source



def test_top_cpp_emits_conv_ddr_tiled_without_full_weight_replica() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [
                Op(
                    name="conv0",
                    op_type="Conv",
                    inputs=["x", "W0", "B0"],
                    outputs=["y"],
                    attrs={"precision_tag": "op0"},
                )
            ]
            self.tensors = {
                "x": TensorSpec((1, 1, 4, 4)),
                "W0": TensorSpec((2, 1, 3, 3)),
                "B0": TensorSpec((2,)),
                "y": TensorSpec((1, 2, 2, 2)),
            }
            self.constants = {
                "W0": np.ones((2, 1, 3, 3), dtype=np.float32),
                "B0": np.zeros((2,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="ddr_tiled",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "ddr_tiled"}),
        communication_plan=None,
        raw_cfg={},
    )

    assert "Requested weights mode: ddr_tiled" in source
    assert "DDR-tiled Dense/Conv inference" in source
    assert "const ap_uint<32>* weights_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in source
    assert "conv2d_ddr_tiled<4, 4, 1, 2, 2, 2, 3, 1, 0" in source
    assert "conv_weight_tile[TILE_OC][TILE_IC][K][K]" in source
    assert "fpgai_load_ddr_vector" not in source
    assert "static op0_wgt_t W0" not in source
    assert "static op0_bias_t B0" not in source
    assert "conv2d<" not in source


def test_top_cpp_emits_mixed_conv_dense_ddr_tiled_with_correct_offsets() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [
                Op("conv0", "Conv", ["x", "W0", "B0"], ["conv_y"], {"precision_tag": "op0"}),
                Op("Flatten", "Flatten", ["conv_y"], ["flat"], {}),
                Op("dense0", "Dense", ["flat", "W1", "B1"], ["y"], {"precision_tag": "op2"}),
            ]
            self.tensors = {
                "x": TensorSpec((1, 1, 4, 4)),
                "W0": TensorSpec((2, 1, 3, 3)),
                "B0": TensorSpec((2,)),
                "conv_y": TensorSpec((1, 2, 2, 2)),
                "flat": TensorSpec((1, 8)),
                "W1": TensorSpec((3, 8)),
                "B1": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((2, 1, 3, 3), dtype=np.float32),
                "B0": np.zeros((2,), dtype=np.float32),
                "W1": np.ones((3, 8), dtype=np.float32),
                "B1": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="ddr_tiled",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "ddr_tiled"}),
        communication_plan=None,
        raw_cfg={},
    )

    # Conv weights are first: W0 has 18 values and B0 has 2 values.
    assert "conv2d_ddr_tiled<4, 4, 1, 2, 2, 2, 3, 1, 0" in source
    assert "weights_mem, 0, 18);" in source
    # Dense starts after Conv W/B payload: 18 + 2 = 20.
    assert "dense_out_in_ddr_tiled<8, 3, 8, 3" in source
    assert "weights_mem, 20, 44);" in source

def test_top_cpp_emits_bram_import_export_runtime_commands() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [Op("dense0", "Dense", ["x", "W0", "B0"], ["y"], {"precision_tag": "op0"})]
            self.tensors = {
                "x": TensorSpec((1, 4)),
                "W0": TensorSpec((3, 4)),
                "B0": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((3, 4), dtype=np.float32),
                "B0": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="ddr",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "bram_import_export_full"}),
        communication_plan=None,
        raw_cfg={},
    )

    assert "ap_uint<32>* weights_mem" in source
    assert "int mode" in source
    assert "#pragma HLS INTERFACE s_axilite port=mode" in source
    assert "FPGAI_MODE_IMPORT_WEIGHTS" in source
    assert "FPGAI_MODE_RUN_INFERENCE" in source
    assert "FPGAI_MODE_EXPORT_WEIGHTS" in source
    assert "if (mode == FPGAI_MODE_IMPORT_WEIGHTS)" in source
    assert "if (mode == FPGAI_MODE_EXPORT_WEIGHTS)" in source
    assert "fpgai_load_ddr_vector<op0_wgt_t, 12>" in source
    assert "fpgai_store_ddr_vector<op0_wgt_t, 12>" in source
    assert "static op0_wgt_t W0[12];" in source
    assert "#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=bram" in source
    assert "Runtime DDR weight load" not in source


def test_top_cpp_emits_uram_import_full_without_export_command() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [Op("dense0", "Dense", ["x", "W0", "B0"], ["y"], {"precision_tag": "op0"})]
            self.tensors = {
                "x": TensorSpec((1, 4)),
                "W0": TensorSpec((3, 4)),
                "B0": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((3, 4), dtype=np.float32),
                "B0": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="uram",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "uram_import_full"}),
        communication_plan=None,
        raw_cfg={},
    )

    assert "FPGAI_MODE_IMPORT_WEIGHTS" in source
    assert "fpgai_load_ddr_vector<op0_wgt_t, 12>" in source
    assert "if (mode == FPGAI_MODE_EXPORT_WEIGHTS)" not in source
    assert "fpgai_store_ddr_vector<op0_wgt_t, 12>" not in source
    assert "static op0_wgt_t W0[12];" in source
    assert "#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=uram" in source
    assert "mode != FPGAI_MODE_RUN_INFERENCE" in source


def test_top_cpp_emits_m_axi_full_input_output_arrays() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
    from fpgai.engine.models import CommunicationEdge, CommunicationPlan

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [Op("dense0", "Dense", ["x", "W0", "B0"], ["y"], {"precision_tag": "op0"})]
            self.tensors = {
                "x": TensorSpec((1, 4)),
                "W0": TensorSpec((3, 4)),
                "B0": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((3, 4), dtype=np.float32),
                "B0": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

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

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "bram_static"}),
        communication_plan=communication_plan,
        raw_cfg={},
    )

    assert "const ap_uint<32>* input_mem" in source
    assert "ap_uint<32>* output_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=input_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=output_mem" in source
    assert "hls::stream<axis_t>& in_stream" not in source
    assert "hls::stream<axis_t>& out_stream" not in source
    assert "input_mem[index].to_uint()" in source
    assert "output_mem[index] = value_to_bits" in source
    assert "FPGAI bram_static weight storage" in source


def test_top_cpp_emits_m_axi_tiled_input_output_arrays() -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import numpy as np

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
    from fpgai.engine.models import CommunicationEdge, CommunicationPlan

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [Op("dense0", "Dense", ["x", "W0", "B0"], ["y"], {"precision_tag": "op0"})]
            self.tensors = {
                "x": TensorSpec((1, 4)),
                "W0": TensorSpec((3, 4)),
                "B0": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((3, 4), dtype=np.float32),
                "B0": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

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

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_weight_semantics": "bram_static"}),
        communication_plan=communication_plan,
        raw_cfg={},
    )

    assert "const ap_uint<32>* input_mem" in source
    assert "ap_uint<32>* output_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=input_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=output_mem" in source
    assert "hls::stream<axis_t>& in_stream" not in source
    assert "hls::stream<axis_t>& out_stream" not in source
    assert "FPGAI_INPUT_TILE_SIZE" in source
    assert "FPGAI_OUTPUT_TILE_SIZE" in source
    assert "input_tile" in source
    assert "output_tile" in source
    assert "m_axi tiled input import" in source
    assert "m_axi tiled output export" in source
    assert "input_mem[tile_base + lane].to_uint()" in source
    assert "output_mem[tile_base + lane] = value_to_bits" in source


def test_manual_activation_storage_drives_memory_plan_region() -> None:
    plan = _compile_plan("bram")
    plan.notes["requested_activation_storage"] = "uram"
    plan.notes["resolved_activation_storage"] = "uram"
    plan.notes["activation_region_preference"] = ["URAM", "BRAM", "DDR"]

    graph = SimpleNamespace(inputs=["x"], outputs=["z"], ops=[])
    mem = make_memory_plan(graph, [_descriptor()], plan)

    activation_regions = {
        placement.region
        for placement in mem.placements
        if placement.kind == "activation"
    }
    assert activation_regions == {"URAM"}


def test_compiler_activation_storage_resolver_accepts_bram_uram_and_rejects_bad_values() -> None:
    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)

    bram = compiler._resolve_activation_storage_semantics({"memory": {"storage": {"activations": "bram"}}})
    assert bram["resolved_activation_storage"] == "bram"
    assert bram["activation_storage_semantics"] == "activation_bram"

    uram = compiler._resolve_activation_storage_semantics({"targets": {"board": "kv260"}, "memory": {"storage": {"activations": "uram"}}})
    assert uram["resolved_activation_storage"] == "uram"
    assert uram["activation_storage_semantics"] == "activation_uram"

    try:
        compiler._resolve_activation_storage_semantics({"memory": {"storage": {"activations": "ddr"}}})
    except ValueError as exc:
        assert "memory.storage.activations must be bram or uram" in str(exc)
    else:
        raise AssertionError("invalid activation storage should reject")


def test_activation_uram_rejects_on_board_without_uram() -> None:
    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)
    try:
        compiler._resolve_activation_storage_semantics({"targets": {"board": "pynq_z2"}, "memory": {"storage": {"activations": "uram"}}})
    except ValueError as exc:
        assert "has 0 URAM blocks" in str(exc)
    else:
        raise AssertionError("URAM activation storage should reject on PYNQ-Z2")


def test_top_cpp_rewrites_activation_buffers_to_uram() -> None:
    from dataclasses import dataclass
    import numpy as np
    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp

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

    class Graph:
        inputs = ["x"]
        outputs = ["y"]

        def __init__(self) -> None:
            self.ops = [Op("dense0", "Dense", ["x", "W0", "B0"], ["y"], {"precision_tag": "op0"})]
            self.tensors = {
                "x": TensorSpec((1, 4)),
                "W0": TensorSpec((3, 4)),
                "B0": TensorSpec((3,)),
                "y": TensorSpec((1, 3)),
            }
            self.constants = {
                "W0": np.ones((3, 4), dtype=np.float32),
                "B0": np.zeros((3,), dtype=np.float32),
            }

        def get_tensor(self, name: str):
            return self.tensors.get(name)

    source = emit_top_cpp(
        graph=Graph(),
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=SimpleNamespace(layer_plans=[]),
        memory_plan=SimpleNamespace(notes={"resolved_activation_storage": "uram"}),
        communication_plan=None,
        raw_cfg={"memory": {"storage": {"activations": "uram"}}},
    )

    assert "FPGAI activation storage: uram" in source
    assert "#pragma HLS BIND_STORAGE variable=layer_in type=ram_1p impl=uram" in source
    assert "#pragma HLS BIND_STORAGE variable=layer_0_out type=ram_1p impl=uram" in source


def test_user_facing_weights_mode_embedded_expands_to_static_bram_and_uram() -> None:
    from fpgai.engine.compiler import Compiler
    from fpgai.engine.planner import _choose_weight_mode

    compiler = object.__new__(Compiler)

    bram_raw = {
        "memory": {"storage": {"weights": "bram"}},
        "weights": {"mode": "embedded"},
    }
    assert _choose_weight_mode(None, bram_raw) == "embedded"
    bram_semantics = compiler._resolve_weight_movement_semantics(bram_raw)
    assert bram_semantics["memory_semantics_mode"] == "bram_static"
    assert bram_semantics["hls_weights_mode"] == "embedded"
    assert bram_semantics["weight_import_interface"] == "compile_time"
    assert bram_semantics["weight_import_policy"] == "static"
    assert bram_semantics["weight_movement_source"] == "weights.mode"

    uram_raw = {
        "memory": {"storage": {"weights": "uram"}},
        "weights": {"mode": "embedded"},
    }
    assert _choose_weight_mode(None, uram_raw) == "embedded"
    uram_semantics = compiler._resolve_weight_movement_semantics(uram_raw)
    assert uram_semantics["memory_semantics_mode"] == "uram_static"
    assert uram_semantics["hls_weights_mode"] == "embedded"


def test_user_facing_weights_mode_import_and_import_export_expand_to_runtime_m_axi() -> None:
    from fpgai.engine.compiler import Compiler
    from fpgai.engine.planner import _choose_weight_mode

    compiler = object.__new__(Compiler)

    bram_import = {
        "memory": {"storage": {"weights": "bram"}},
        "weights": {"mode": "import"},
    }
    assert _choose_weight_mode(None, bram_import) == "ddr"
    import_semantics = compiler._resolve_weight_movement_semantics(bram_import)
    assert import_semantics["memory_semantics_mode"] == "bram_import_full"
    assert import_semantics["hls_weights_mode"] == "ddr"
    assert import_semantics["weight_import_interface"] == "m_axi"
    assert import_semantics["weight_import_transport"] == "ps_runtime"
    assert import_semantics["weight_import_policy"] == "full"
    assert import_semantics["weight_export_interface"] == "none"
    assert "import_weights" in import_semantics["runtime_commands_supported"]
    assert "export_weights" not in import_semantics["runtime_commands_supported"]

    uram_import_export = {
        "memory": {"storage": {"weights": "uram"}},
        "weights": {"mode": "import_export"},
    }
    assert _choose_weight_mode(None, uram_import_export) == "uram"
    export_semantics = compiler._resolve_weight_movement_semantics(uram_import_export)
    assert export_semantics["memory_semantics_mode"] == "uram_import_export_full"
    assert export_semantics["hls_weights_mode"] == "uram"
    assert export_semantics["weight_export_interface"] == "m_axi"
    assert export_semantics["weight_export_policy"] == "full"
    assert "export_weights" in export_semantics["runtime_commands_supported"]


def test_user_facing_weights_mode_tiled_expands_to_ddr_tiled() -> None:
    from fpgai.engine.compiler import Compiler
    from fpgai.engine.planner import _choose_weight_mode

    compiler = object.__new__(Compiler)
    raw = {
        "memory": {"storage": {"weights": "ddr"}},
        "weights": {"mode": "tiled"},
    }

    assert _choose_weight_mode(None, raw) == "ddr"
    semantics = compiler._resolve_weight_movement_semantics(raw)
    assert semantics["memory_semantics_mode"] == "ddr_tiled"
    assert semantics["hls_weights_mode"] == "ddr_tiled"
    assert semantics["weight_import_interface"] == "m_axi"
    assert semantics["weight_import_policy"] == "tiled"
    assert semantics["weight_export_interface"] == "none"
    assert semantics["full_local_weight_replica"] is False
    assert semantics["tile_weight_buffer"] is True


def test_manual_detailed_data_movement_overrides_user_facing_weights_mode() -> None:
    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)
    raw = {
        "memory": {"storage": {"weights": "bram"}},
        "weights": {"mode": "embedded"},
        "data_movement": {
            "weights": {
                "import": {
                    "interface": "m_axi",
                    "transport": "ps_runtime",
                    "policy": "full",
                }
            }
        },
    }

    semantics = compiler._resolve_weight_movement_semantics(raw)
    assert semantics["memory_semantics_mode"] == "bram_import_full"
    assert semantics["weight_movement_source"] == "data_movement"
    assert semantics["weight_import_interface"] == "m_axi"
    assert semantics["weight_import_policy"] == "full"


def test_user_facing_weights_mode_rejects_unsupported_storage_combinations() -> None:
    import pytest
    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)

    with pytest.raises(ValueError, match="weights.mode=embedded.*DDR storage"):
        compiler._resolve_weight_movement_semantics(
            {"memory": {"storage": {"weights": "ddr"}}, "weights": {"mode": "embedded"}}
        )

    with pytest.raises(ValueError, match="weights.mode=tiled requires memory.storage.weights=ddr"):
        compiler._resolve_weight_movement_semantics(
            {"memory": {"storage": {"weights": "bram"}}, "weights": {"mode": "tiled"}}
        )

    with pytest.raises(ValueError, match="Unsupported weights.mode"):
        compiler._resolve_weight_movement_semantics(
            {"memory": {"storage": {"weights": "bram"}}, "weights": {"mode": "magic"}}
        )


def test_training_weight_initialization_mode_expands_to_static_or_import() -> None:
    import pytest
    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)

    compile_time_raw = {
        "pipeline": {"mode": "training_on_device"},
        "memory": {"storage": {"weights": "bram"}},
        "training": {"weight_initialization": {"mode": "compile_time"}},
    }
    compile_time_semantics = compiler._resolve_weight_movement_semantics(compile_time_raw)
    assert compile_time_semantics["memory_semantics_mode"] == "bram_static"
    assert compile_time_semantics["weight_import_interface"] == "compile_time"
    assert compile_time_semantics["weight_import_policy"] == "static"

    import_raw = {
        "pipeline": {"mode": "training_on_device"},
        "memory": {"storage": {"weights": "bram"}},
        "training": {"weight_initialization": {"mode": "import"}},
    }
    import_semantics = compiler._resolve_weight_movement_semantics(import_raw)
    assert import_semantics["memory_semantics_mode"] == "bram_import_full"
    assert import_semantics["weight_import_interface"] == "m_axi"
    assert import_semantics["weight_import_policy"] == "full"

    with pytest.raises(ValueError, match="training.weight_initialization.mode=.*not implemented"):
        compiler._resolve_weight_movement_semantics(
            {
                "pipeline": {"mode": "training_on_device"},
                "memory": {"storage": {"weights": "bram"}},
                "training": {"weight_initialization": {"mode": "xavier"}},
            }
        )
