"""Read-only source-only readiness for pinned hls4ml ONNX frontend.

Inspects the installed hls4ml operator registry, not guesses from generic
hls4ml docs. This never invokes model conversion, CSim, HLS or Vivado.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_candidate_heldout import attest_candidate

SCHEMA = "fpgai-research/hls4ml-frontend-readiness/v1"
PINNED_HLS4ML_VERSION = "1.3.0"


def _installed_frontend() -> tuple[str | None, set[str] | None]:
    """Optional external dependency; no model conversion is attempted."""
    try:
        import hls4ml
        layers = hls4ml.converters.get_supported_onnx_layers()
        if not isinstance(layers, (list, tuple, set)):
            return None, None
        if not all(isinstance(op, str) and op for op in layers):
            return None, None
        return str(hls4ml.__version__), set(layers)
    except (ImportError, AttributeError, RuntimeError):
        return None, None


def collect_frontend_readiness(repository: str | Path, candidate_dir: str | Path,
                               kind: str) -> dict:
    """Read the immutable Gemm-free candidate and inspect installed handlers."""
    import onnx

    attested = attest_candidate(repository, candidate_dir, kind)
    model = onnx.load_model_from_string(attested["candidate_bytes"])
    onnx.checker.check_model(model)
    operators = Counter(node.op_type for node in model.graph.node)
    if not operators or "Gemm" in operators:
        raise ValueError("Expected actual nonempty Gemm-free candidate graph")
    versions = sorted((node.domain or "ai.onnx", node.op_type)
                      for node in model.graph.node)
    nonstandard = sorted({
        domain for domain, _ in versions if domain != "ai.onnx"})
    installed_version, supported = _installed_frontend()
    unsupported = (None if supported is None
                   else sorted(set(operators) - supported))
    conv_needs_channel_last = any(
        node.op_type == "Conv" and
        node.domain != "qonnx.custom_op.channels_last"
        for node in model.graph.node)
    blockers = []
    if installed_version is None or supported is None:
        blockers.append("HLS4ML_ONNX_FRONTEND_NOT_INSTALLED_OR_NOT_IMPORTABLE")
    elif installed_version != PINNED_HLS4ML_VERSION:
        blockers.append("HLS4ML_VERSION_DIFFERS_FROM_PINNED_STUDY")
    if unsupported:
        blockers.append("UNSUPPORTED_ONNX_OPERATOR_IN_INSTALLED_FRONTEND")
    if nonstandard:
        blockers.append("NONSTANDARD_ONNX_DOMAINS_REQUIRE_MANUAL_REVIEW")
    if conv_needs_channel_last:
        blockers.append("CONV_REQUIRES_CHANNELS_LAST_QONNX_PREPROCESSING")
    if any(d is None or d <= 0
           for value in model.graph.input
           for d in [
               dim.dim_value if dim.HasField("dim_value") else None
               for dim in value.type.tensor_type.shape.dim
           ]):
        blockers.append("DYNAMIC_OR_NONPOSITIVE_INPUT_GEOMETRY")
    # Even when there are no known preflight blockers, real conversion
    # and software/fixed-point matching must still be checked separately.
    return {
        "schema": SCHEMA,
        "model_kind": kind,
        "candidate_onnx_sha256": attested["candidate_sha256"],
        "source_onnx_sha256": attested["source_sha256"],
        "pinned_hls4ml_version": PINNED_HLS4ML_VERSION,
        "installed_hls4ml_version": installed_version,
        "installed_frontend_registry_available": supported is not None,
        "operator_counts": dict(sorted(operators.items())),
        "unsupported_operator_types": unsupported,
        "nonstandard_operator_domains": nonstandard,
        "conv_channels_last_required": conv_needs_channel_last,
        "requires_input_layout_numerical_review": True,
        "source_only_conversion_attempt_ready": not blockers,
        "blockers": blockers,
        "actual_hls4ml_conversion_executed": False,
        "hls4ml_e1_source_generated": False,
        "numeric_e0_full_candidate_verified_by_this_audit": False,
        "numeric_e2_verified": False,
        "hls4ml_e3_e4_verified": False,
        "external_qor_comparison_eligible": False,
        "native_vendor_tools_executed": False,
    }


def export_frontend_readiness(repository: str | Path, candidate_dir: str | Path,
                              kind: str, output: str | Path) -> dict:
    """Immutable private JSON plus public-safe readable operator/blocker summary."""
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("hls4ml frontend readiness output already exists")
    report = collect_frontend_readiness(repository, candidate_dir, kind)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls4ml-front-", dir=target.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        (stage / "readiness.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        lines = [
            "# hls4ml ONNX frontend — source-only preflight",
            "",
            f'Model: {kind.upper()}. Pinned baseline: hls4ml {PINNED_HLS4ML_VERSION}.',
            f'Installed version: {report["installed_hls4ml_version"] or "unavailable"}.',
            "",
            "Operators in derived candidate: " +
            ", ".join(f"{op} ({count})" for op, count in report["operator_counts"].items()) + ".",
            "",
            "Unsupported operators: " +
            ("not checked (frontend unavailable)" if
             report["unsupported_operator_types"] is None else
             ", ".join(report["unsupported_operator_types"]) or "none detected"),
            "",
            "Source-only conversion *attempt* ready: " +
            ("yes" if report["source_only_conversion_attempt_ready"] else "no"),
            "",
            "Blockers: " + (", ".join(report["blockers"]) or "none detected"),
            "",
            "**No hls4ml conversion has been executed. No E1/E2/E3/E4 or "
            "competitive hardware result is established.**",
            "",
        ]
        (stage / "readiness.md").write_text("\n".join(lines), encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Readiness output appeared during export")
        stage.rename(target)
    return report
