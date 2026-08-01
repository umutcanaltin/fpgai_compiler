from __future__ import annotations

from .implementation_contract import ImplementationContract


def builtin_implementation_contracts() -> tuple[ImplementationContract, ...]:
    """Return descriptive built-in contracts without changing compiler dispatch."""
    return (
        ImplementationContract(
            package_id="fpgai.implementation.dense_hls",
            version="1.0.0",
            operator_id="fpgai.operator.dense",
            language="hls_cpp",
            backend="vitis_hls",
            top="dense",
            sources=("builtin://dense.cpp",),
            inference=True,
            precisions=("fp32", "int16", "int8"),
            validation_level="c_simulation_validated",
            metadata={"descriptive_only": True},
        ),
        ImplementationContract(
            package_id="fpgai.implementation.conv2d_hls",
            version="1.0.0",
            operator_id="fpgai.operator.conv2d",
            language="hls_cpp",
            backend="vitis_hls",
            top="conv2d",
            sources=("builtin://conv.cpp",),
            inference=True,
            precisions=("fp32", "int16", "int8"),
            validation_level="c_simulation_validated",
            metadata={"descriptive_only": True},
        ),
    )
