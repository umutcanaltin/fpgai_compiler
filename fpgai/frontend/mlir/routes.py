from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class MLIRIngressRoute:
    framework: str
    producer: str
    preferred_dialect: str
    accepted_by_fpgai: bool
    maturity: str
    accepted_formats: Tuple[str, ...] = ()
    legalization_required: bool = False
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


_ROUTES = {
    "jax": MLIRIngressRoute(
        framework="jax",
        producer="jax.export / OpenXLA",
        preferred_dialect="stablehlo",
        accepted_by_fpgai=True,
        maturity="supported_subset",
        accepted_formats=("stablehlo", "mlir"),
        legalization_required=False,
        notes=("JAX StableHLO textual import is supported for the FPGAI operator subset.",),
    ),
    "pytorch": MLIRIngressRoute(
        framework="pytorch",
        producer="torch.export + PyTorch/XLA StableHLO export (or torch-mlir upstream)",
        preferred_dialect="stablehlo",
        accepted_by_fpgai=True,
        maturity="supported_subset",
        accepted_formats=("stablehlo", "mlir", "onnx"),
        legalization_required=False,
        notes=(
            "PyTorch-origin models are accepted after upstream export/legalization to StableHLO or ONNX.",
            "Raw torch dialect MLIR is not parsed directly by FPGAI.",
        ),
    ),
    "tensorflow": MLIRIngressRoute(
        framework="tensorflow",
        producer="tf.mlir.experimental SavedModel/ConcreteFunction conversion",
        preferred_dialect="tf",
        accepted_by_fpgai=False,
        maturity="requires_upstream_legalization",
        accepted_formats=("stablehlo", "mlir", "onnx"),
        legalization_required=True,
        notes=(
            "TensorFlow-dialect MLIR is not parsed directly by FPGAI yet.",
            "Legalize/export to StableHLO or another FPGAI-supported MLIR dialect first.",
        ),
    ),
    "stablehlo": MLIRIngressRoute(
        framework="stablehlo",
        producer="any StableHLO producer",
        preferred_dialect="stablehlo",
        accepted_by_fpgai=True,
        maturity="supported_subset",
        accepted_formats=("stablehlo", "mlir"),
        legalization_required=False,
    ),
    "fpgai": MLIRIngressRoute(
        framework="fpgai",
        producer="FPGAI MLIR bridge/native dialect",
        preferred_dialect="fpgai",
        accepted_by_fpgai=True,
        maturity="bridge_supported_native_dialect_scaffolded",
        accepted_formats=("mlir",),
        legalization_required=False,
    ),
}


def framework_mlir_routes() -> Dict[str, dict]:
    return {name: route.to_dict() for name, route in _ROUTES.items()}


def framework_mlir_route(framework: str) -> MLIRIngressRoute:
    key = str(framework).strip().lower()
    if key not in _ROUTES:
        raise KeyError(f"MLIRROUTE001: unsupported framework route {framework!r}")
    return _ROUTES[key]


def detect_mlir_dialect(text: str) -> str:
    source = str(text)
    if "fpgai.bridge.payload" in source or 'fpgai.schema = "fpgai.mlir-bridge/v1"' in source:
        return "fpgai_bridge"
    if "stablehlo." in source or '"stablehlo.' in source:
        return "stablehlo"
    if "tf." in source or '"tf.' in source:
        return "tf"
    if "torch." in source or '"torch.' in source:
        return "torch"
    if "fpgai." in source or '"fpgai.' in source:
        return "fpgai_native"
    return "unknown"
