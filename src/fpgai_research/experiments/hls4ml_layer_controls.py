"""Exact hls4ml 1.3.0 per-layer control census for the E0-attested ONNX.

This interrogates real backend layer attributes via hls4ml's config generator.
It does NOT generate HLS, execute native tools, or interpret unsupported
operator-local controls as proof of an FPGAI hardware advantage.
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
    fixed_precision,
)
from fpgai_research.experiments.hls4ml_frontend_readiness import (
    PINNED_HLS4ML_VERSION, collect_frontend_readiness,
)

SCHEMA = "fpgai-research/hls4ml-layer-control-census/v1"


def inspect_layer_controls(repository, candidate_dir, full_e0_dir, model_kind,
                           *, bits=16, integer=6, reuse_factor=1) -> dict:
    """Read-only; actual per-layer configurable keys, not guessed abilities."""
    import onnx
    import hls4ml

    precision = fixed_precision(bits, integer)
    if type(reuse_factor) is not int or not 1 <= reuse_factor <= 4096:
        raise ValueError("Invalid canonical hls4ml reuse factor")
    path = Path(candidate_dir) / "candidate.onnx"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Missing/unsafe derived ONNX source")
    raw = path.read_bytes()
    identity = sha256(raw).hexdigest()
    binding = _attest_derived_e0(
        repository=repository, candidate_dir=candidate_dir,
        full_e0_dir=full_e0_dir, kind=model_kind, model_sha256=identity)
    preflight = collect_frontend_readiness(repository, candidate_dir, model_kind)
    if (str(getattr(hls4ml, "__version__", "")) != PINNED_HLS4ML_VERSION
            or preflight.get("source_only_conversion_attempt_ready") is not True):
        raise ValueError("Census requires compatible pinned hls4ml ONNX frontend")
    model, adapter = annotate_initializer_shapes_for_hls4ml(
        onnx.load_model_from_string(raw))
    # hls4ml logs original private node names while parsing.
    with redirect_stdout(StringIO()):
        config = hls4ml.utils.config_from_onnx_model(
            model, granularity="name", backend="Vitis",
            default_precision=precision, default_reuse_factor=reuse_factor)
    if (not isinstance(config, dict) or not isinstance(config.get("Model"), dict)
            or not isinstance(config.get("LayerName"), dict)
            or not config["LayerName"]):
        raise ValueError("hls4ml returned no usable per-layer-name config")
    rows = []
    for name, cfg in sorted(config["LayerName"].items()):
        if (not isinstance(name, str) or not name or not isinstance(cfg, dict)
                or not all(isinstance(k, str) and k for k in cfg)):
            raise ValueError("Malformed hls4ml per-layer control definition")
        factor = cfg.get("ReuseFactor")
        if factor is not None and (type(factor) is not int or not 1 <= factor <= 4096):
            raise ValueError("Unsupported hls4ml ReuseFactor representation")
        rows.append({
            "layer_name": name,
            "configurable_keys": sorted(cfg.keys()),
            "configurable_reuse_factor": factor is not None,
            "default_reuse_factor": factor,
        })
    matmuls = sorted(node.name.replace(".", "_") for node in model.graph.node
                     if node.op_type == "MatMul")
    recorded = {r["layer_name"]: r for r in rows}
    if not matmuls or len(set(matmuls)) != len(matmuls):
        raise ValueError("Expected uniquely named MLP/CNN MatMul operators")
    matmul_controls = [
        {"layer_name": name,
         "present_in_hls4ml_config": name in recorded,
         "configurable_reuse_factor":
             recorded[name]["configurable_reuse_factor"] if name in recorded else False}
        for name in matmuls
    ]
    if sha256(path.read_bytes()).hexdigest() != identity:
        raise ValueError("Source ONNX changed during the control census")
    return {
        "schema": SCHEMA,
        "model_kind": model_kind,
        "candidate_onnx_sha256": identity,
        "full_e0_receipt_sha256": binding["full_e0_receipt_sha256"],
        "hls4ml_version": PINNED_HLS4ML_VERSION,
        "precision": precision,
        "default_reuse_factor": reuse_factor,
        "layer_count": len(rows),
        "layers": rows,
        "configurable_reuse_layer_count": sum(r["configurable_reuse_factor"] for r in rows),
        "matmul_layers": matmul_controls,
        "matmul_reuse_configurable_count":
            sum(r["configurable_reuse_factor"] for r in matmul_controls),
        "actual_onnx_parser_and_backend_config_generator_called": True,
        "onnx_adapter_shape_annotations": adapter["added_initializer_value_info_count"],
        "onnx_adapter_constant_aliases": adapter["renamed_initializer_identity_count"],
        "actual_hls4ml_e1_generated": False,
        "achieved_reuse_in_hls_or_rtl_verified": False,
        "external_e2_e3_e4_qualified": False,
        "external_qor_comparison_eligible": False,
        "native_vendor_tools_executed": False,
    }


def export_layer_controls(repository, candidate_dir, full_e0_dir, model_kind,
                          output, *, bits=16, integer=6, reuse_factor=1) -> dict:
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Private layer-control census already exists")
    record = inspect_layer_controls(
        repository, candidate_dir, full_e0_dir, model_kind,
        bits=bits, integer=integer, reuse_factor=reuse_factor)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls4ml-census-", dir=destination.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        (stage / "layer_controls_private.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        # No trainable tensor/node names or fingerprints in public-facing prose.
        (stage / "summary.md").write_text(
            "# hls4ml 1.3.0 per-layer E1 configuration inventory\n\n"
            f"Model: {model_kind.upper()}; observed hls4ml layers: {record['layer_count']}.\n\n"
            f"Layers exposing ReuseFactor: {record['configurable_reuse_layer_count']}.\n\n"
            f"MatMul layers: {len(record['matmul_layers'])}; "
            f"MatMul layers exposing ReuseFactor: {record['matmul_reuse_configurable_count']}.\n\n"
            "Use the **private** layer name/key inventory to select a defensible "
            "tuned E1 variant. A reported key is only a configuration capability, "
            "not evidence of realized reuse, source differences, HLS synthesis "
            "or a hardware performance advantage.\n",
            encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Census destination appeared during write")
        stage.rename(destination)
    return record
