"""Stage source-identical hls4ml E1 as a Vitis CSim-only experiment.

Does not run Vitis, CSim, HLS synth, GCC, cosim, or Vivado. Original E1
and qualified original/normalized ONNX stay unchanged.
"""
from __future__ import annotations
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.hls4ml_baseline import _attest_derived_e0
from fpgai_research.experiments.hls4ml_e1_reaudit import collect_e1_source_reaudit
from fpgai_research.experiments.hls4ml_gemm_candidate import _selected_input_agreement
from fpgai_research.experiments.hls4ml_onnx_inventory import (
    _DEFAULT_TRAINED, DEFAULT_SUITE, KINDS, _source_bytes,
)

SCHEMA = "fpgai-research/hls4ml-selected-e2-csim-stage/v1"
CSIM_ONLY = {
    "reset": 0, "csim": 1, "synth": 0, "cosim": 0,
    "validation": 0, "export": 0, "vsynth": 0, "fifo_opt": 0,
}


def safe_project(source: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Expected ordinary immutable E1 source")
    count = total = 0
    for p in source.rglob("*"):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            raise ValueError("Linked or unsafe E1 project file")
        count += 1
        if count > 3000:
            raise ValueError("Exceeded bounded E1 file count")
        if p.is_file():
            size = p.stat().st_size
            total += size
            if size > 128 * 1024 * 1024 or total > 512 * 1024 * 1024:
                raise ValueError("Oversized E1 project")


def csim_only_options(content: str) -> str:
    if not isinstance(content, str) or not content.strip().startswith("array set opt {"):
        raise ValueError("Unknown Vitis build_opt Tcl format")
    entries = re.findall(r"(?m)^\s*([a-z_]+)\s+([01])\s*$", content)
    if (len(entries) != len(CSIM_ONLY)
            or {key for key, _ in entries} != set(CSIM_ONLY)
            or len({key for key, _ in entries}) != len(entries)
            or not content.strip().endswith("}")):
        raise ValueError("Unknown, duplicated or incomplete build options")
    return "array set opt {\n" + "".join(
        f"    {key:<10} {value}\n" for key, value in CSIM_ONLY.items()) + "}\n"


def format_tb(data: np.ndarray, features: int) -> str:
    if (not isinstance(data, np.ndarray) or data.dtype != np.float32
            or data.shape != (16, features) or not np.all(np.isfinite(data))):
        raise ValueError("Expected 16 finite float32 samples of fixed geometry")
    lines = [" ".join(format(float(v), ".9g") for v in row) for row in data]
    check = np.asarray([[float(x) for x in line.split()] for line in lines],
                       dtype=np.float32)
    if not np.array_equal(check, data):
        raise ValueError("Float32 changed in HLS testbench text")
    return "\n".join(lines) + "\n"


def stage_selected_csim(repository, candidate_dir, full_e0_dir,
                        model_kind, project_dir, output) -> dict:
    import onnxruntime as ort

    if model_kind not in KINDS:
        raise ValueError("Expected study MLP or CNN")
    dest, root, project = Path(output), Path(repository), Path(project_dir)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Immutable CSim output already exists")
    safe_project(project)
    audit = collect_e1_source_reaudit(
        root, candidate_dir, full_e0_dir, model_kind, project)
    cand = _source_bytes(Path(candidate_dir) / "candidate.onnx")
    raw = _source_bytes(root / _DEFAULT_TRAINED[model_kind] / "model.onnx")
    binding = _attest_derived_e0(
        repository=root, candidate_dir=candidate_dir,
        full_e0_dir=full_e0_dir, kind=model_kind,
        model_sha256=sha256(cand).hexdigest())
    if (audit["candidate_onnx_sha256"] != sha256(cand).hexdigest()
            or audit["full_e0_receipt_sha256"] != binding["full_e0_receipt_sha256"]):
        raise ValueError("Source or E0 lineage differs from audited E1")

    trained = root / _DEFAULT_TRAINED[model_kind]
    suite = root / DEFAULT_SUITE / "assets" / f"mnist_{model_kind}"
    inp_raw, labels_raw = _source_bytes(trained / "inputs.npy"), _source_bytes(trained / "labels.npy")
    if (inp_raw != _source_bytes(suite / "inputs.npy")
            or labels_raw != _source_bytes(suite / "labels.npy")):
        raise ValueError("Selected original and suite MNIST inputs differ")
    examples = np.load(BytesIO(inp_raw), allow_pickle=False)
    labels = np.load(BytesIO(labels_raw), allow_pickle=False)
    if (examples.dtype != np.float32 or examples.shape != (16, 1, 28, 28)
            or labels.shape != (16,) or labels.dtype.kind not in ("i", "u")
            or np.any(labels > 9) or np.any(labels < 0)
            or not np.all(np.isfinite(examples))):
        raise ValueError("Expected canonical selected 16-image MNIST cohort")
    agreement = _selected_input_agreement(raw, cand, examples)
    if agreement["selected_samples_verified"] != 16 or agreement["selected_top1_disagreements"]:
        raise ValueError("Failed original vs derived selected numerical agreement")
    session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Unexpected original ONNX input/output contract")
    reference = []
    for x in examples:
        y = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(x[None])})
        if len(y) != 1 or y[0].shape != (1, 10) or y[0].dtype != np.float32:
            raise ValueError("Unexpected original ONNX output geometry")
        reference.append(y[0][0].copy())
    reference = np.ascontiguousarray(np.stack(reference))
    if (not np.all(np.isfinite(reference))
            or not np.allclose(reference.sum(axis=1), 1.0, atol=1e-4, rtol=0)):
        raise ValueError("Original output probabilities invalid")
    input_text = format_tb(np.ascontiguousarray(examples.reshape(16, 784)), 784)
    reference_text = format_tb(reference, 10)

    options = project / "build_opt.tcl"
    if options.is_symlink() or not options.is_file():
        raise ValueError("Generated E1 build_opt Tcl missing or linked")
    csim_only = csim_only_options(options.read_text(encoding="utf-8"))
    for name in ("build_prj.tcl", "project.tcl", "fpgairesearch_baseline_test.cpp"):
        if not (project / name).is_file() or (project / name).is_symlink():
            raise ValueError("Generated E1 CSim Tcl/testbench source missing")
    if sha256(_source_bytes(Path(candidate_dir) / "candidate.onnx")).hexdigest() != audit["candidate_onnx_sha256"]:
        raise ValueError("Candidate changed during staging")

    record = {
        "schema": SCHEMA,
        "evidence": "E2-CSIM-STAGED-NOT-EXECUTED",
        "model_kind": model_kind,
        "hls4ml_version": audit["hls4ml_version"],
        "original_onnx_sha256": sha256(raw).hexdigest(),
        "candidate_onnx_sha256": audit["candidate_onnx_sha256"],
        "full_e0_receipt_sha256": audit["full_e0_receipt_sha256"],
        "e1_project_receipt_sha256": sha256((project / "research_baseline.json").read_bytes()).hexdigest(),
        "generated_cpp_sha256": audit["generated_cpp_sha256"],
        "selected_inputs_sha256": sha256(inp_raw).hexdigest(),
        "selected_labels_sha256": sha256(labels_raw).hexdigest(),
        "tb_input_sha256": sha256(input_text.encode()).hexdigest(),
        "tb_reference_sha256": sha256(reference_text.encode()).hexdigest(),
        "test_samples": 16,
        "onnx_reference_correct": int(np.sum(reference.argmax(axis=1) == labels)),
        "selected_original_to_derived_onnx_parity": agreement,
        "build_options": CSIM_ONLY,
        "cloned_generated_cpp_unmodified": True,
        "csim_executed": False,
        "csim_numerical_e2_verified": False,
        "full_10000_fpga_accuracy_measured": False,
        "synthesis_e3_executed": False,
        "physical_e4_executed": False,
        "external_qor_comparison_eligible": False,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls4ml-csim-", dir=dest.parent) as temp:
        stage = Path(temp) / "staged"
        stage.mkdir()
        clone = stage / "project"
        shutil.copytree(project, clone)
        (clone / "build_opt.tcl").write_text(csim_only, encoding="utf-8")
        tb = clone / "tb_data"
        tb.mkdir(exist_ok=True)
        (tb / "tb_input_features.dat").write_text(input_text, encoding="utf-8")
        (tb / "tb_output_predictions.dat").write_text(reference_text, encoding="utf-8")
        for path, digest in audit["generated_cpp_sha256"].items():
            if sha256((clone / path).read_bytes()).hexdigest() != digest:
                raise ValueError("Qualified E1 source changed during private clone")
        config = clone / "hls4ml_config.yml"
        old = str(project.absolute())
        yaml = config.read_text(encoding="utf-8")
        if old not in yaml:
            raise ValueError("Cloned YAML did not identify audited original E1 project")
        config.write_text(yaml.replace(old, str((dest / "project").absolute())),
                          encoding="utf-8")
        (stage / "stage.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        (stage / "README.md").write_text(
            "# Private Vitis CSim-only MNIST E2 staging\n\n"
            "16 selected original E0 inputs and ONNX Runtime reference probabilities. "
            "No native tool was executed. The cloned HLS C++ hashes match E1, "
            "and cloned build_opt.tcl enables only csim. Manually execute "
            "Vitis HLS 2023.2 from the staged project directory with "
            "vitis_hls -f build_prj.tcl only after reviewing the staged controls. "
            "Record tool output and compare it numerically before claiming E2. "
            "This is not E3, E4 or a hardware superiority result.\n",
            encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Stage appeared before immutable publication")
        stage.rename(dest)
    return record
