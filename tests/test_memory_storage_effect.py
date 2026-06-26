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
