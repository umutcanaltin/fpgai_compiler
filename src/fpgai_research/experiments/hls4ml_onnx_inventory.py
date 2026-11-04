"""Immutable private ONNX graph inventory for a *candidate* hls4ml comparison.

This inspects the unchanged MNIST training/source copies without loading
weights into public artifacts, transforming ONNX, or invoking hls4ml/Vitis.
ONNX checker success is not hls4ml converter support or numerical equivalence.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.mnist_heldout_paper import (
    DEFAULT_SUITE, KINDS, _DEFAULT_TRAINED,
)

SCHEMA = "fpgai-research/hls4ml-onnx-inventory/v1"
_MAX_BYTES = 64 * 1024 * 1024


def _source_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("ONNX source must be a regular, non-linked file")
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("ONNX source exceeds bounded comparison inventory limit")
    return path.read_bytes()


def _shape(value) -> dict:
    if not value.type.HasField("tensor_type"):
        return {"kind": "non_tensor", "shape": None, "element_type": None}
    tensor = value.type.tensor_type
    dims = []
    for d in tensor.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            dims.append(None)
    return {"kind": "tensor", "shape": dims, "element_type": int(tensor.elem_type)}


def _bounded_attributes(node, onnx) -> dict:
    """Only semantic operator controls; never tensor attribute values/names."""
    names = {
        "Gemm": ("transA", "transB", "alpha", "beta"),
        "Conv": ("group", "kernel_shape", "strides", "pads", "dilations", "auto_pad"),
        "MaxPool": ("kernel_shape", "strides", "pads", "ceil_mode", "auto_pad"),
        "Flatten": ("axis",),
        "Transpose": ("perm",),
        "Softmax": ("axis",),
    }.get(node.op_type, ())
    result = {}
    for attr in node.attribute:
        if attr.name not in names:
            continue
        if attr.type == onnx.AttributeProto.INT:
            result[attr.name] = int(attr.i)
        elif attr.type == onnx.AttributeProto.FLOAT:
            result[attr.name] = float(attr.f)
        elif attr.type == onnx.AttributeProto.INTS:
            result[attr.name] = [int(i) for i in attr.ints]
        elif attr.type == onnx.AttributeProto.STRING:
            result[attr.name] = attr.s.decode("ascii", errors="replace")
        else:
            result[attr.name] = "REQUIRES_REVIEW"
    return result


def _inspect_model(path: Path, onnx) -> dict:
    raw = _source_bytes(path)
    # External tensor data is intentionally forbidden: merely inspecting the
    # graph must not follow a relative path into unknown private assets.
    model = onnx.load_model_from_string(raw)
    initializers = list(model.graph.initializer)
    if model.graph.sparse_initializer:
        raise ValueError("Sparse ONNX initializers require separate review")
    if any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data
           for t in initializers):
        raise ValueError("External ONNX tensor data is not accepted")
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS,
                             onnx.AttributeProto.SPARSE_TENSOR,
                             onnx.AttributeProto.SPARSE_TENSORS):
                raise ValueError("Nested graph/sparse tensor attributes require a separate audit")
    onnx.checker.check_model(model)
    counts = Counter(n.op_type for n in model.graph.node)
    nodes = [
        {
            "index": i,
            "domain": n.domain or "ai.onnx",
            "op_type": n.op_type,
            "input_count": len(n.input),
            "output_count": len(n.output),
            "operator_controls": _bounded_attributes(n, onnx),
        }
        for i, n in enumerate(model.graph.node)
    ]
    custom_domains = sorted({
        n.domain for n in model.graph.node if n.domain not in ("", "ai.onnx")
    })
    input_names = {t.name for t in initializers}
    inputs = [_shape(t) for t in model.graph.input if t.name not in input_names]
    outputs = [_shape(t) for t in model.graph.output]
    if not nodes or not inputs or not outputs:
        raise ValueError("Comparison requires a nonempty computational ONNX graph")
    return {
        "onnx_sha256": sha256(raw).hexdigest(),
        "onnx_ir_version": int(model.ir_version),
        "opsets": [{"domain": entry.domain or "ai.onnx", "version": int(entry.version)}
                   for entry in model.opset_import],
        "node_count": len(nodes),
        "operator_counts": dict(sorted(counts.items())),
        "operator_inventory": nodes,
        "inputs": inputs,
        "outputs": outputs,
        "initializer_count": len(initializers),
        "initializer_element_types": dict(sorted(Counter(
            str(int(t.data_type)) for t in initializers).items())),
        "has_nonstandard_node_domains": bool(custom_domains),
        "nonstandard_node_domains": custom_domains,
        "gemm_decomposition_requires_review": "Gemm" in counts,
        "cnn_tensor_layout_requires_review": "Conv" in counts or
            any(t["shape"] is not None and len(t["shape"]) == 4 for t in inputs),
        "graph_checked_by_onnx": True,
        "hls4ml_frontend_conversion_verified": False,
        "hls4ml_numeric_equivalence_verified": False,
    }


def collect_onnx_inventory(repository: str | Path) -> dict:
    """Audit BOTH exact original models against assembled copies, read-only."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Optional ONNX dependency needed for source inspection") from exc
    root = Path(repository)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected regular private research repository")
    rows = []
    for kind in KINDS:
        original = root / _DEFAULT_TRAINED[kind] / "model.onnx"
        assembled = root / DEFAULT_SUITE / "assets" / f"mnist_{kind}" / "model.onnx"
        original_bytes, assembled_bytes = _source_bytes(original), _source_bytes(assembled)
        if original_bytes != assembled_bytes:
            raise ValueError("Trained ONNX differs from original paper-suite ONNX")
        row = _inspect_model(original, onnx)
        if sha256(_source_bytes(original)).hexdigest() != row["onnx_sha256"]:
            raise ValueError("Original ONNX mutated during inspection")
        if sha256(_source_bytes(assembled)).hexdigest() != row["onnx_sha256"]:
            raise ValueError("Assembled ONNX mutated during inspection")
        rows.append({"model": kind, "trained_onnx_matches_assembled": True, **row})
    return {
        "schema": SCHEMA,
        "onnx_version": str(onnx.__version__),
        "model_count": len(rows),
        "models": rows,
        "source_onnx_identical_to_assembled": True,
        "hls4ml_invoked": False,
        "native_vendor_tools_executed": False,
        "external_qor_comparison_eligible": False,
        "status": "ONNX_GRAPH_INSPECTED_FRONTEND_NUMERIC_AND_HARDWARE_REVIEW_REQUIRED",
    }


def export_onnx_inventory(repository: str | Path, output: str | Path) -> dict:
    """Atomic new private receipt; public-facing summary has no private SHA."""
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable ONNX inventory output already exists")
    result = collect_onnx_inventory(repository)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-onnx-inventory-", dir=target.parent) as temp:
        stage = Path(temp) / "inventory"
        stage.mkdir()
        (stage / "inventory.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# hls4ml ONNX source inspection (not a benchmark)",
            "",
            "Both original trained ONNX files match their paper-suite copies.",
            "No hls4ml conversion, numerical match, CSim, HLS or Vivado was executed.",
            "",
            "| Model | Nodes | Operators | Gemm review | CNN layout review |",
            "| --- | ---: | --- | --- | --- |",
        ]
        for row in result["models"]:
            ops = ", ".join(f"{op} ({count})" for op, count in row["operator_counts"].items())
            lines.append(
                f'| {row["model"].upper()} | {row["node_count"]} | {ops} | '
                f'{"yes" if row["gemm_decomposition_requires_review"] else "no"} | '
                f'{"yes" if row["cnn_tensor_layout_requires_review"] else "no"} |'
            )
        lines.extend([
            "", "**No comparison eligibility is granted by this inspection.**",
            "Check the installed hls4ml ONNX operator registry, graph transformations, "
            "weight/layout identity, per-layer numeric policy and native source-to-E4 lineage.",
            "",
        ])
        (stage / "inventory.md").write_text("\n".join(lines), encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("ONNX inventory appeared during export")
        stage.rename(target)
    return result
