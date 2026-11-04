"""Preserve both successful and failed real hls4ml ONNX parser probes.

Unlike a registry-only preflight, this invokes the pinned hls4ml parser on
an in-memory shape-annotated Gemm-free candidate. It does not generate C++
or run Vitis/Vivado. Raw node names and weights stay out of public reports.
"""
from __future__ import annotations
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_baseline import (
    _attest_derived_e0, annotate_initializer_shapes_for_hls4ml,
)
from fpgai_research.experiments.hls4ml_frontend_readiness import (
    collect_frontend_readiness, PINNED_HLS4ML_VERSION,
)

SCHEMA = "fpgai-research/hls4ml-onnx-parser-probe/v1"


def probe_candidate_parser(repository, candidate_dir, full_e0_dir, model_kind):
    """Read-only parser invocation; a failure is recorded, not paper success."""
    import onnx
    import hls4ml

    src = Path(candidate_dir) / "candidate.onnx"
    if src.is_symlink() or not src.is_file() or src.stat().st_size > 64*1024*1024:
        raise ValueError("Missing or unsafe immutable candidate")
    raw = src.read_bytes()
    sha = sha256(raw).hexdigest()
    lineage = _attest_derived_e0(
        repository=repository, candidate_dir=candidate_dir,
        full_e0_dir=full_e0_dir, kind=model_kind, model_sha256=sha)
    status = collect_frontend_readiness(repository, candidate_dir, model_kind)
    if (str(getattr(hls4ml, "__version__", "unknown")) != PINNED_HLS4ML_VERSION
            or status.get("installed_hls4ml_version") != PINNED_HLS4ML_VERSION):
        raise ValueError("Expected the pinned hls4ml ONNX parser")
    if not status.get("source_only_conversion_attempt_ready"):
        blockers = status.get("blockers", [])
        return {
            "schema": SCHEMA, "model_kind": model_kind,
            "candidate_onnx_sha256": sha,
            "source_lineage_e0_reattested": bool(lineage),
            "hls4ml_version": PINNED_HLS4ML_VERSION,
            "parser_attempted": False, "parser_passed": False,
            "blockers": blockers,
            "exception_class": None, "layer_class_counts": None,
            "hls4ml_source_e1_generated": False,
            "hls4ml_fixed_point_e2_verified": False,
            "external_qor_comparison_eligible": False,
            "native_vendor_tools_executed": False,
        }
    source = onnx.load_model_from_string(raw)
    model, metadata = annotate_initializer_shapes_for_hls4ml(source)
    error_type = None
    classes = None
    # Upstream prints private ONNX names during parsing; suppress them.
    try:
        with redirect_stdout(StringIO()):
            layer_list, inputs, outputs = hls4ml.converters.parse_onnx_model(model)
        if (not isinstance(layer_list, list) or not layer_list
                or not isinstance(inputs, list) or not inputs
                or not isinstance(outputs, list) or not outputs
                or any(not isinstance(row, dict)
                       or not isinstance(row.get("class_name"), str)
                       or not row["class_name"]
                       for row in layer_list)):
            raise ValueError("Incomplete hls4ml parsed topology")
        classes = dict(sorted(Counter(row["class_name"] for row in layer_list).items()))
    except Exception as exc:
        # The exception *message* may include weight or model node names.
        error_type = type(exc).__name__
    return {
        "schema": SCHEMA, "model_kind": model_kind,
        "candidate_onnx_sha256": sha,
        "source_lineage_e0_reattested": bool(lineage),
        "hls4ml_version": PINNED_HLS4ML_VERSION,
        "initializer_shape_metadata_added": metadata["added_initializer_value_info_count"],
        "parser_attempted": True, "parser_passed": error_type is None,
        "blockers": [] if error_type is None else ["HLS4ML_ONNX_PARSER_REJECTED_GRAPH"],
        "exception_class": error_type,
        "layer_class_counts": classes,
        "hls4ml_source_e1_generated": False,
        "hls4ml_fixed_point_e2_verified": False,
        "external_qor_comparison_eligible": False,
        "native_vendor_tools_executed": False,
    }


def export_candidate_parser_probe(repository, candidate_dir, full_e0_dir,
                                  model_kind, output):
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Immutable ONNX parser probe already exists")
    report = probe_candidate_parser(repository, candidate_dir, full_e0_dir, model_kind)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-onnx-parser-probe-", dir=dest.parent) as temp:
        stage = Path(temp)/"probe"
        stage.mkdir()
        (stage/"parser_probe.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n",
            encoding="utf-8")
        (stage/"parser_probe.md").write_text(
            "# hls4ml 1.3.0 ONNX parser compatibility\n\n"
            f'Model: {model_kind.upper()}. Parser passed: {report["parser_passed"]}.\n\n'
            f'Blockers: {", ".join(report["blockers"]) or "none detected"}.\n\n'
            "No source E1, fixed-point E2, native E3/E4 or external advantage is claimed.\n",
            encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("ONNX parser probe output appeared concurrently")
        stage.rename(dest)
    return report
