from __future__ import annotations

from enum import Enum


PACKAGE_SCHEMA_V1 = "fpgai.package/v1"
FPGAI_CONTRACT_VERSION = "1.0"


class AssetType(str, Enum):
    MODEL = "model"
    OPERATOR = "operator"
    IMPLEMENTATION = "implementation"
    ACCELERATOR = "accelerator"
    BOARD = "board"
    BACKEND = "backend"
    OPTIMIZER = "optimizer"
    LOSS = "loss"
    DATASET = "dataset"
    MEMORY_POLICY = "memory_policy"
    TRANSPORT = "transport"
    RUNTIME_REFERENCE = "runtime_reference"
    VALIDATION = "validation"
    REPORTER = "reporter"
    BENCHMARK = "benchmark"
    SYSTEM_BLOCK = "system_block"
    ADAPTER = "adapter"


class LicenseCategory(str, Enum):
    OPEN_SOURCE = "open_source"
    SOURCE_AVAILABLE = "source_available"
    RESEARCH_ONLY = "research_only"
    COMMERCIAL = "commercial"
    PROPRIETARY = "proprietary"
    INTERNAL = "internal"


class ValidationLevel(str, Enum):
    UNVALIDATED = "unvalidated"
    REFERENCE_TESTED = "reference_tested"
    C_SIMULATION_VALIDATED = "c_simulation_validated"
    RTL_SIMULATION_VALIDATED = "rtl_simulation_validated"
    HLS_SYNTHESIZED = "hls_synthesized"
    VIVADO_SYNTHESIZED = "vivado_synthesized"
    IMPLEMENTED = "implemented"
    BITSTREAM_GENERATED = "bitstream_generated"
    HARDWARE_TESTED = "hardware_tested"
    REPRODUCIBLE = "reproducible"
    CERTIFIED_EXTERNAL = "certified_external"


class ImplementationLanguage(str, Enum):
    HLS_CPP = "hls_cpp"
    VHDL = "vhdl"
    VERILOG = "verilog"
    SYSTEMVERILOG = "systemverilog"


class InterfaceProtocol(str, Enum):
    AXI_STREAM = "axi_stream"
    M_AXI = "m_axi"
    AXI_LITE = "axi_lite"
    SCALAR = "scalar"
    MEMORY = "memory"
    SOFTWARE = "software"
