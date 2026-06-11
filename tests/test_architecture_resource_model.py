from __future__ import annotations

import pytest

from fpgai.analysis.architecture_resource_model import (
    estimate_architecture_layer_resources,
)
from fpgai.analysis.hls_architecture import LayerArchitecture


@pytest.mark.parametrize("op_type", ["MaxPool", "AvgPool"])
def test_pool_resource_estimate_has_zero_dsp(op_type: str) -> None:
    layer = LayerArchitecture(
        name="pool0",
        op_type=op_type,
        dimensions={
            "input_elements": 2704,
            "output_elements": 676,
            "input_channels": 4,
            "input_width": 26,
            "kernel_height": 2,
        },
        pipeline_scope="output",
        pipeline_ii=1,
        unroll={},
        reduction_iterations=4,
        pipeline_overlap=1,
        explicit_lanes=1,
        effective_lanes=1,
        arithmetic={
            "activation_bits": 16,
            "weight_bits": 16,
            "bias_bits": 24,
            "accumulator_bits": 24,
            "output_bits": 16,
            "effective_multiplier_units": 0,
        },
        memory={
            "input_banks": 1,
            "output_banks": 1,
            "weight_banks": 1,
            "weight_mode": "embedded",
            "activation_mode": "buffer",
            "buffering": "single",
        },
    )

    result = estimate_architecture_layer_resources(
        layer,
        {
            "memory": {
                "storage": {
                    "weights": "bram",
                    "activations": "bram",
                }
            }
        },
    )

    assert result["predicted_dsp"] == 0
    assert result["partition_factor"] == 1
