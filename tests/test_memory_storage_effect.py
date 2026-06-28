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


def test_memory_storage_uram_resolves_to_runtime_uram_mode() -> None:
    from fpgai.engine.planner import _choose_weight_mode

    raw = {
        "memory": {"storage": {"weights": "uram"}},
        "data_movement": {"ps_pl": {"weights": {"mode": "embedded"}}},
    }

    assert _choose_weight_mode(None, raw) == "uram"


def test_memory_storage_ddr_resolves_to_runtime_ddr_mode() -> None:
    from fpgai.engine.planner import _choose_weight_mode

    raw = {
        "memory": {"storage": {"weights": "ddr"}},
        "data_movement": {"ps_pl": {"weights": {"mode": "embedded"}}},
    }

    assert _choose_weight_mode(None, raw) == "ddr"


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
        "data_movement": {
            "weights": {
                "load": {
                    "interface": "ddr",
                    "source": "ps_ddr",
                }
            }
        }
    }
    assert compiler._resolve_hls_weights_mode(raw) == "ddr"

    raw = {
        "memory": {
            "storage": {
                "weights": "uram",
            }
        },
        "data_movement": {
            "weights": {
                "load": {
                    "interface": "embedded",
                }
            }
        },
    }
    assert compiler._resolve_hls_weights_mode(raw) == "uram"


def test_communication_plan_accepts_new_input_output_schema(tmp_path) -> None:
    from types import SimpleNamespace

    from fpgai.engine.communication import make_communication_plan
    from fpgai.engine.memory import MemoryPlan

    cfg = SimpleNamespace(
        raw={
            "data_movement": {
                "input": {
                    "load": {
                        "tensor_name": "x_new",
                        "region": "HOST",
                        "size_words": 11,
                        "interface": "dma",
                    }
                },
                "output": {
                    "store": {
                        "tensor_name": "y_new",
                        "region": "HOST",
                        "size_words": 7,
                        "interface": "dma",
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
