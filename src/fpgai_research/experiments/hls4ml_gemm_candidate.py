"""Source-attested private Gemm normalization for an hls4ml ONNX candidate.

This is NOT a backend benchmark. It preserves trained source models and
requires actual ONNX Runtime agreement on the original selected inputs
before exporting a derived ONNX graph. Only bounded float32 constant-RHS
Gemm with alpha=beta=1, transA=0, transB=0/1 is accepted. No silent
fallback or unvalidated algebraic lowering is allowed.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.hls4ml_baseline import _attest_private_inventory
from fpgai_research.experiments.hls4ml_onnx_inventory import (
    KINDS, _DEFAULT_TRAINED, DEFAULT_SUITE, _source_bytes,
)

SCHEMA = "fpgai-research/hls4ml-gemm-candidate/v1"


def _unique(base: str, used: set[str]) -> str:
    name = base
    counter = 0
    while not name or name in used:
        counter += 1
        name = f"{base}_{counter}"
    used.add(name)
    return name


def decompose_constant_gemm(model):
    """Return model and number of changed Gemm nodes; do not alter input.

    The preserved Gemm result tensor is produced by Add when a bias exists.
    FP arithmetic reassociation is accepted ONLY after independent ORT checks.
    """
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    transformed = onnx.ModelProto()
    transformed.CopyFrom(model)
    graph = transformed.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    used = {item.name for item in graph.input}
    used.update(item.name for item in graph.output)
    used.update(item.name for item in graph.value_info)
    used.update(initializers)
    used.update(node.name for node in graph.node if node.name)
    used.update(name for node in graph.node for name in [*node.input, *node.output] if name)

    nodes = []
    added = []
    count = 0
    for index, node in enumerate(graph.node):
        if node.op_type != "Gemm":
            copy = onnx.NodeProto()
            copy.CopyFrom(node)
            nodes.append(copy)
            continue
        if node.domain not in ("", "ai.onnx") or len(node.output) != 1:
            raise ValueError("Only standard one-output Gemm can be normalized")
        inputs = list(node.input)
        if len(inputs) not in (2, 3) or not inputs[0] or not inputs[1]:
            raise ValueError("Gemm requires two nonempty inputs and optional bias")
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if (set(attrs) - {"alpha", "beta", "transA", "transB"}
                or float(attrs.get("alpha", 1.0)) != 1.0
                or float(attrs.get("beta", 1.0)) != 1.0
                or int(attrs.get("transA", 0)) != 0
                or int(attrs.get("transB", 0)) not in (0, 1)):
            raise ValueError("Gemm alpha/beta/transA/transB contract unsupported")
        weight = initializers.get(inputs[1])
        if weight is None or weight.data_type != TensorProto.FLOAT or len(weight.dims) != 2:
            raise ValueError("Gemm RHS must be a static rank-two float32 initializer")
        bias = inputs[2] if len(inputs) == 3 else None
        if bias:
            tensor = initializers.get(bias)
            if tensor is None or tensor.data_type != TensorProto.FLOAT:
                raise ValueError("Gemm bias must be a static float32 initializer")
            if len(tensor.dims) not in (1, 2):
                raise ValueError("Gemm bias must be a rank-one or rank-two tensor")
        rhs = inputs[1]
        if int(attrs.get("transB", 0)) == 1:
            values = numpy_helper.to_array(weight)
            if values.dtype != np.float32 or values.ndim != 2:
                raise ValueError("Gemm RHS initializer is not float32 rank two")
            rhs = _unique(f"fpgai_gemm_rhs_{index}", used)
            added.append(numpy_helper.from_array(
                np.ascontiguousarray(values.T), name=rhs))
        intermediate = (_unique(f"fpgai_gemm_product_{index}", used)
                        if bias else node.output[0])
        nodes.append(helper.make_node(
            "MatMul", [inputs[0], rhs], [intermediate],
            name=_unique(f"fpgai_matmul_{index}", used)))
        if bias:
            nodes.append(helper.make_node(
                "Add", [intermediate, bias], [node.output[0]],
                name=_unique(f"fpgai_bias_add_{index}", used)))
        count += 1
    if count == 0:
        raise ValueError("No Gemm operators found; refusing a no-op candidate")
    del graph.node[:]
    graph.node.extend(nodes)
    graph.initializer.extend(added)
    # Remove an original transposed RHS initializer only if no longer used.
    # Retain all other original constant tensors byte-for-byte.
    referenced = {name for node in graph.node for name in node.input if name}
    referenced.update(item.name for item in graph.output)
    kept = [t for t in graph.initializer if t.name in referenced]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(transformed)
    transformed = onnx.shape_inference.infer_shapes(transformed)
    onnx.checker.check_model(transformed)
    if any(n.op_type == "Gemm" for n in transformed.graph.node):
        raise ValueError("Gemm survived normalization")
    return transformed, count


def _selected_input_agreement(source_bytes: bytes, candidate_bytes: bytes,
                              samples: np.ndarray) -> dict:
    """Exact same selected float32 inputs, both ONNX Runtime CPU sessions."""
    import onnxruntime as ort

    original = ort.InferenceSession(source_bytes, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(candidate_bytes, providers=["CPUExecutionProvider"])
    if len(original.get_inputs()) != 1 or len(candidate.get_inputs()) != 1:
        raise ValueError("Comparison requires exactly one runtime input")
    if len(original.get_outputs()) != 1 or len(candidate.get_outputs()) != 1:
        raise ValueError("Comparison requires exactly one runtime output")
    input_shape = original.get_inputs()[0].shape
    if input_shape != candidate.get_inputs()[0].shape:
        raise ValueError("Normalized candidate changed runtime input shape")
    if samples.dtype != np.float32 or samples.ndim != len(input_shape):
        raise ValueError("Expected source-identical selected float32 test inputs")
    if not (1 <= samples.shape[0] <= 256 and samples.shape[0] >= 1):
        raise ValueError("Invalid bounded selected test cohort")
    if input_shape[0] != 1 or any(type(d) is not int or d < 1 for d in input_shape):
        raise ValueError("Only static original batch-one input geometry allowed")
    if list(samples.shape[1:]) != input_shape[1:]:
        raise ValueError("Selected test data shape differs from original model")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Nonfinite selected test data")
    max_abs = 0.0
    total_abs = 0.0
    count = 0
    disagreements = 0
    for row in samples:
        sample = np.ascontiguousarray(row[np.newaxis, ...])
        lhs = original.run(None, {original.get_inputs()[0].name: sample})[0]
        rhs = candidate.run(None, {candidate.get_inputs()[0].name: sample})[0]
        if (lhs.dtype != np.float32 or rhs.dtype != np.float32
                or lhs.shape != rhs.shape or lhs.ndim != 2
                or lhs.shape[0] != 1 or lhs.shape[1] < 2
                or not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs))):
            raise ValueError("Normalized candidate produced incompatible output")
        if not np.allclose(lhs, rhs, rtol=2e-5, atol=2e-5):
            raise ValueError("Normalized Gemm candidate fails selected ORT numeric tolerance")
        disagreements += int(np.argmax(lhs) != np.argmax(rhs))
        diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
        max_abs = max(max_abs, float(np.max(diff)))
        total_abs += float(np.sum(diff))
        count += diff.size
    if disagreements:
        raise ValueError("Normalized candidate changed selected top-one predictions")
    return {
        "selected_samples_verified": int(samples.shape[0]),
        "selected_top1_disagreements": disagreements,
        "max_absolute_difference": max_abs,
        "mean_absolute_difference": total_abs / count,
        "absolute_tolerance": 2e-5,
        "relative_tolerance": 2e-5,
        "onnxruntime_version": str(ort.__version__),
        "full_heldout_10000_transformed_model_verified": False,
        "hls4ml_numerical_e2_verified": False,
    }


def export_gemm_candidate(repository: str | Path, inventory_dir: str | Path,
                          model_kind: str, output: str | Path) -> dict:
    """Write a new *private* immutable candidate after selected E0 agreement."""
    import onnx

    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable hls4ml candidate output already exists")
    if model_kind not in KINDS:
        raise ValueError("Only original MNIST MLP/CNN study models are supported")
    root = Path(repository)
    original = root / _DEFAULT_TRAINED[model_kind] / "model.onnx"
    assembled = root / DEFAULT_SUITE / "assets" / f"mnist_{model_kind}" / "model.onnx"
    raw = _source_bytes(original)
    if raw != _source_bytes(assembled):
        raise ValueError("Original model does not match immutable paper-suite copy")
    source_hash = sha256(raw).hexdigest()
    inventory_hash = _attest_private_inventory(
        repository=root, inventory_dir=inventory_dir,
        model_kind=model_kind, source=original.resolve(strict=True),
        model_sha256=source_hash)
    transformed, gemm_count = decompose_constant_gemm(onnx.load_model_from_string(raw))
    candidate_bytes = transformed.SerializeToString()
    # The selected original samples are distinct from the 10k-image E0 study.
    trained_samples = root / _DEFAULT_TRAINED[model_kind] / "inputs.npy"
    assembled_samples = root / DEFAULT_SUITE / "assets" / f"mnist_{model_kind}" / "inputs.npy"
    # Do not depend on ONNX extension for sample checks; enforce regular
    # non-symlink file and matched original paper-suite fingerprint.
    if (trained_samples.is_symlink() or assembled_samples.is_symlink()
            or not trained_samples.is_file() or not assembled_samples.is_file()
            or trained_samples.stat().st_size > 8 * 1024 * 1024
            or assembled_samples.stat().st_size > 8 * 1024 * 1024):
        raise ValueError("Missing/unsafe selected original MNIST input arrays")
    raw_samples = trained_samples.read_bytes()
    if raw_samples != assembled_samples.read_bytes():
        raise ValueError("Original selected inputs do not match paper-suite copy")
    samples = np.load(trained_samples, allow_pickle=False)
    agreement = _selected_input_agreement(raw, candidate_bytes, samples)
    # Recheck source and selected inputs after the numerical comparison.
    if (sha256(_source_bytes(original)).hexdigest() != source_hash
            or sha256(_source_bytes(assembled)).hexdigest() != source_hash
            or sha256(trained_samples.read_bytes()).hexdigest() != sha256(raw_samples).hexdigest()
            or sha256(assembled_samples.read_bytes()).hexdigest() != sha256(raw_samples).hexdigest()):
        raise ValueError("Original study source or selected data mutated during comparison")
    report = {
        "schema": SCHEMA,
        "evidence": "E0-selected-onnxruntime-numeric-acceptance",
        "model_kind": model_kind,
        "source_onnx_sha256": source_hash,
        "source_inventory_sha256": inventory_hash,
        "candidate_onnx_sha256": sha256(candidate_bytes).hexdigest(),
        "selected_inputs_sha256": sha256(raw_samples).hexdigest(),
        "gemm_nodes_decomposed": gemm_count,
        "generated_matmul_nodes": gemm_count,
        "original_trained_onnx_and_suite_unchanged": True,
        "selected_sample_numeric_agreement": agreement,
        "candidate_hls4ml_frontend_conversion_verified": False,
        "candidate_hls4ml_source_e1_verified": False,
        "candidate_native_e2_e3_e4_verified": False,
        "original_trained_onnx_to_native_fpga_weight_identity_verified": False,
        "external_qor_comparison_eligible": False,
        "hls4ml_invoked": False,
        "native_vendor_tools_executed": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-gemm-candidate-", dir=target.parent) as temp:
        stage = Path(temp) / "candidate"
        stage.mkdir()
        (stage / "candidate.onnx").write_bytes(candidate_bytes)
        (stage / "normalization.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        (stage / "summary.md").write_text(
            "# hls4ml Gemm normalization candidate (NOT a benchmark)\n\n"
            f'Model: {model_kind.upper()}; Gemm replaced: {gemm_count}; '
            f'selected inputs numerically checked: {agreement["selected_samples_verified"]}; '
            f'top-one disagreements: {agreement["selected_top1_disagreements"]}.\n\n'
            "This is selected E0 ONNX Runtime software agreement only. "
            "Full 10,000-image candidate equivalence and all external hls4ml "
            "E1/E2/E3/E4 evidence remain unverified.\n",
            encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Candidate output appeared during export")
        stage.rename(target)
    return report
