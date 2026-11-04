"""Full official MNIST E0 comparison: original ONNX vs derived Gemm-free ONNX.

No hls4ml conversion, native CSim, hardware accuracy or FPGA performance
is inferred. Original trained/suite files and immutable candidate stay read-only.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.hls4ml_gemm_candidate import (
    SCHEMA as CANDIDATE_SCHEMA, decompose_constant_gemm,
)
from fpgai_research.experiments.hls4ml_onnx_inventory import (
    _source_bytes, _DEFAULT_TRAINED, DEFAULT_SUITE, KINDS,
)
from fpgai_research.experiments.mnist_heldout_paper import (
    collect_heldout_paper, _DEFAULT_RECEIPTS,
)

SCHEMA = "fpgai-research/hls4ml-candidate-heldout-e0/v1"
COUNT = 10000
ATOL = 2e-5
RTOL = 2e-5


def _private_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Missing or unsafe private evidence JSON")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected private evidence object")
    return value


def attest_candidate(repository: str | Path, candidate_dir: str | Path,
                     kind: str) -> dict:
    """Recompute normalized graph and source fingerprints before any evaluation."""
    import onnx

    if kind not in KINDS:
        raise ValueError("Unknown trained MNIST comparison model")
    root = Path(repository)
    stage = Path(candidate_dir)
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("Expected regular immutable candidate directory")
    src = root / _DEFAULT_TRAINED[kind] / "model.onnx"
    assembled = root / DEFAULT_SUITE / "assets" / f"mnist_{kind}" / "model.onnx"
    original = _source_bytes(src)
    if original != _source_bytes(assembled):
        raise ValueError("Original and assembled paper-suite ONNX disagree")
    candidate_path = stage / "candidate.onnx"
    converted = _source_bytes(candidate_path)
    receipt = _private_json(stage / "normalization.json")
    if (receipt.get("schema") != CANDIDATE_SCHEMA
            or receipt.get("model_kind") != kind
            or receipt.get("source_onnx_sha256") != sha256(original).hexdigest()
            or receipt.get("candidate_onnx_sha256") != sha256(converted).hexdigest()
            or receipt.get("evidence") != "E0-selected-onnxruntime-numeric-acceptance"
            or receipt.get("original_trained_onnx_and_suite_unchanged") is not True
            or receipt.get("external_qor_comparison_eligible") is not False
            or receipt.get("candidate_native_e2_e3_e4_verified") is not False
            or receipt.get("hls4ml_invoked") is not False
            or receipt.get("native_vendor_tools_executed") is not False):
        raise ValueError("Unqualified or altered derived ONNX candidate")
    selection = receipt.get("selected_sample_numeric_agreement")
    if (not isinstance(selection, dict)
            or selection.get("selected_samples_verified") != 16
            or selection.get("selected_top1_disagreements") != 0
            or selection.get("full_heldout_10000_transformed_model_verified") is not False):
        raise ValueError("Derived candidate lacks the canonical 16-sample acceptance")
    expected_model, count = decompose_constant_gemm(
        onnx.load_model_from_string(original))
    if (count != receipt.get("gemm_nodes_decomposed")
            or expected_model.SerializeToString() != converted):
        raise ValueError("Derived candidate differs from reproducible Gemm normalization")
    return {
        "original_bytes": original,
        "candidate_bytes": converted,
        "source_sha256": sha256(original).hexdigest(),
        "candidate_sha256": sha256(converted).hexdigest(),
        "source_inventory_sha256": receipt.get("source_inventory_sha256"),
    }


def compare_probabilities(source_session, candidate_session, data: np.ndarray,
                          labels: np.ndarray, expected_correct: int,
                          *, limit: int = COUNT) -> dict:
    """Strict float32, one-image-at-a-time ORT comparison, no output samples."""
    if (type(limit) is not int or limit < 1 or type(expected_correct) is not int
            or expected_correct < 0 or expected_correct > limit
            or data.shape != (limit, 1, 28, 28) or data.dtype != np.float32
            or labels.shape != (limit,) or labels.dtype.kind not in ("i", "u")
            or np.any(labels < 0) or np.any(labels > 9) or not np.all(np.isfinite(data))):
        raise ValueError("Invalid official MNIST batch or labels")
    inputs = (source_session.get_inputs(), candidate_session.get_inputs())
    outputs = (source_session.get_outputs(), candidate_session.get_outputs())
    if (any(len(items) != 1 for items in (*inputs, *outputs))
            or any(tuple(items[0].shape) != (1, 1, 28, 28) for items in inputs)
            or any(tuple(items[0].shape) != (1, 10) for items in outputs)):
        raise ValueError("Original and candidate runtime I/O contracts differ")
    correct_source = correct_candidate = disagreements = 0
    maximum = total_abs = 0.0
    for index in range(limit):
        x = np.ascontiguousarray(data[index:index + 1])
        lhs = source_session.run(None, {inputs[0][0].name: x})
        rhs = candidate_session.run(None, {inputs[1][0].name: x})
        if len(lhs) != 1 or len(rhs) != 1:
            raise ValueError("Invalid ONNX Runtime output inventory")
        left, right = lhs[0], rhs[0]
        if (not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray)
                or left.shape != (1, 10) or right.shape != (1, 10)
                or left.dtype != np.float32 or right.dtype != np.float32
                or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
                or np.any(left < 0) or np.any(right < 0)
                or not np.allclose(np.sum(left, axis=1), 1.0, atol=1e-4, rtol=0)
                or not np.allclose(np.sum(right, axis=1), 1.0, atol=1e-4, rtol=0)):
            raise ValueError("Invalid candidate probability output")
        if not np.allclose(left, right, atol=ATOL, rtol=RTOL):
            raise ValueError("Full held-out candidate diverged from original ONNX")
        lp, rp = int(np.argmax(left)), int(np.argmax(right))
        correct_source += int(lp == labels[index])
        correct_candidate += int(rp == labels[index])
        disagreements += int(lp != rp)
        diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
        maximum = max(maximum, float(diff.max()))
        total_abs += float(diff.sum())
    if disagreements or correct_source != expected_correct or correct_candidate != expected_correct:
        raise ValueError("Candidate top-one accuracy differs from qualified original E0")
    return {
        "official_test_images": limit,
        "original_correct_top1": correct_source,
        "candidate_correct_top1": correct_candidate,
        "top1_prediction_disagreements": disagreements,
        "max_probability_absolute_difference": maximum,
        "mean_probability_absolute_difference": total_abs / (limit * 10),
        "absolute_tolerance": ATOL,
        "relative_tolerance": RTOL,
        "candidate_onnx_software_e0_qualified": limit == COUNT,
        "native_fpga_full_test_accuracy_measured": False,
    }


def export_candidate_heldout(repository: str | Path, candidate_dir: str | Path,
                             dataset_dir: str | Path, model_kind: str,
                             output: str | Path) -> dict:
    """Re-audit original E0, run 10,000 real MNIST images, publish immutably."""
    import onnxruntime as ort
    import torch
    from torchvision.datasets import MNIST

    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable full candidate E0 output already exists")
    root = Path(repository)
    qualified = collect_heldout_paper(root)
    source = attest_candidate(root, candidate_dir, model_kind)
    matched_row = next((r for r in qualified["models"] if r["model"] == model_kind), None)
    if matched_row is None or matched_row.get("source_identical_to_assembled_suite") is not True:
        raise ValueError("Requested model not qualified by the original E0 study")
    original_receipt = _private_json(
        root / _DEFAULT_RECEIPTS[model_kind] / "mnist_full_heldout_accuracy.json")
    expected_digests = original_receipt["input_and_model_hashes"]
    dataset_root = Path(dataset_dir)
    if dataset_root.is_symlink():
        raise ValueError("Official MNIST dataset directory must not be linked")
    # Never download, retrain or replace source examples while evaluating.
    ds = MNIST(str(dataset_root), train=False, download=False)
    if (len(ds) != COUNT or ds.data.shape != (COUNT, 28, 28)
            or ds.data.dtype != torch.uint8 or ds.targets.shape != (COUNT,)):
        raise ValueError("Expected official 10,000-image MNIST test dataset")
    pixels = ds.data.numpy()
    labels = ds.targets.numpy()
    if (labels.dtype.kind not in ("i", "u")
            or sha256(pixels.tobytes()).hexdigest() != expected_digests["mnist_test_pixels"]
            or sha256(labels.tobytes()).hexdigest() != expected_digests["mnist_test_labels"]):
        raise ValueError("Held-out images/labels do not match original qualified E0 data")
    data = np.expand_dims(pixels.astype(np.float32) / 255.0, axis=1)
    data = np.ascontiguousarray(data)
    original = ort.InferenceSession(source["original_bytes"], providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(source["candidate_bytes"], providers=["CPUExecutionProvider"])
    result = compare_probabilities(
        original, candidate, data, labels, matched_row["correct_top1_onnxruntime"])
    if (sha256(pixels.tobytes()).hexdigest() != expected_digests["mnist_test_pixels"]
            or sha256(labels.tobytes()).hexdigest() != expected_digests["mnist_test_labels"]
            or attest_candidate(root, candidate_dir, model_kind)["candidate_sha256"] !=
            source["candidate_sha256"]):
        raise ValueError("Candidate sources or dataset changed during full evaluation")
    report = {
        "schema": SCHEMA,
        "model_kind": model_kind,
        "evidence": "E0-full-official-MNIST-original-vs-derived-ONNX",
        "source_onnx_sha256": source["source_sha256"],
        "candidate_onnx_sha256": source["candidate_sha256"],
        "mnist_test_pixels_sha256": expected_digests["mnist_test_pixels"],
        "mnist_test_labels_sha256": expected_digests["mnist_test_labels"],
        "onnxruntime_version": str(ort.__version__),
        "numeric_comparison": result,
        "hls4ml_e1_e2_e3_e4_qualified": False,
        "external_qor_comparison_eligible": False,
        "native_vendor_tools_executed": False,
        "original_trained_to_native_hls_weight_lineage_verified": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls4ml-full-e0-", dir=target.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        (stage / "heldout.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        (stage / "heldout.md").write_text(
            "# Gemm-normalized hls4ml candidate full MNIST E0 comparison\n\n"
            f'Model: {model_kind.upper()} — {result["official_test_images"]} images\n\n'
            f'Original correct: {result["original_correct_top1"]}; '
            f'candidate correct: {result["candidate_correct_top1"]}; '
            f'top-one disagreements: {result["top1_prediction_disagreements"]}.\n\n'
            f'Max absolute probability difference: {result["max_probability_absolute_difference"]:.9g}; '
            f'mean: {result["mean_probability_absolute_difference"]:.9g}.\n\n'
            "**E0 software only. No hls4ml source conversion, fixed-point CSim, "
            "synthesis, routing, board runtime or advantage measured.**\n",
            encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Held-out candidate output appeared during publication")
        stage.rename(target)
    return report
