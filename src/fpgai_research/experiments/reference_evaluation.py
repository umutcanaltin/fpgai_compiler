"""Reproducible float32 Model IR reference; NOT Vitis or FPGA evidence."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
from fpgai_research.validation.reference import evaluate
from fpgai_research.experiments.immutable_evidence import immutable_evidence_directory

def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def evaluate_reference(*, model, inputs_path, output_dir, model_path=None,
                       labels_path=None, crosscheck_onnxruntime=False,
                       atol=1e-5, rtol=1e-4):
    import numpy as np
    model.validate()
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("Reference evaluator requires one input and output")
    if not (0 <= atol < float("inf") and 0 <= rtol < float("inf")):
        raise ValueError("Invalid comparison tolerance")
    if crosscheck_onnxruntime and model_path is None:
        raise ValueError("An actual ONNX file is needed for ONNX Runtime cross-check")
    source=Path(inputs_path)
    source_files={"inputs":source}
    if model_path is not None:
        source_files["model"]=Path(model_path)
    if labels_path is not None:
        source_files["labels"]=Path(labels_path)
    for role,path in source_files.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Reference {role} must be a regular non-linked file: {path}")
    original_hashes={role:_hash(path) for role,path in source_files.items()}
    samples=np.load(source,allow_pickle=False)
    expected_shape=model.tensor(model.inputs[0]).shape
    if samples.dtype != np.float32:
        raise ValueError("Reference inputs must be float32, not silently cast")
    from fpgai_research.validation.samples import normalize_samples
    samples=normalize_samples(samples,expected_shape)
    output_name=model.outputs[0]
    reference=np.stack([evaluate(model,{model.inputs[0]:item})[output_name]
                        for item in samples],axis=0)
    if not np.all(np.isfinite(reference)):
        raise ValueError("Nonfinite reference output")
    labels=None
    predictions=None
    if labels_path is not None:
        labels=np.load(labels_path,allow_pickle=False)
        if labels.ndim!=1 or len(labels)!=len(samples) or labels.dtype.kind not in ("i","u"):
            raise ValueError("Labels must be a 1-D integer array with one label per sample")
        output_shape=model.tensor(output_name).shape
        if len(output_shape)!=2 or output_shape[0]!=1:
            raise ValueError("Classification accuracy requires one output row per sample")
        if np.any(labels<0) or np.any(labels>=output_shape[1]):
            raise ValueError("Class labels outside model output range")
        predictions=reference.reshape(len(samples),output_shape[1]).argmax(axis=1)
    ort_result=None
    if crosscheck_onnxruntime:
        if model_path is None:
            raise ValueError("An actual ONNX file is needed for ONNX Runtime cross-check")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime to cross-check the model") from exc
        session=ort.InferenceSession(str(model_path),providers=["CPUExecutionProvider"])
        if len(session.get_inputs())!=1 or len(session.get_outputs())!=1:
            raise ValueError("Expected single-input/single-output ONNX graph")
        observed=np.stack([session.run(None,{session.get_inputs()[0].name:item})[0]
                           for item in samples],axis=0)
        if observed.shape!=reference.shape:
            raise ValueError("ONNX Runtime and Model IR output shapes disagree")
        deviation=np.abs(observed.astype(np.float64)-reference.astype(np.float64))
        passed=bool(np.all(np.isfinite(observed)) and
                    np.all(deviation<=atol+rtol*np.abs(observed.astype(np.float64))))
        ort_result={"passed":passed,"max_absolute_error":float(np.max(deviation)),
                    "atol":atol,"rtol":rtol,"onnxruntime_version":ort.__version__}
        if not passed:
            raise ValueError("ONNX Runtime cross-check failed: Model IR is not equivalent")
    # Inputs are checked again after evaluation; a concurrent replacement
    # invalidates the software reference rather than silently changing its
    # declared model, labels or numerical input provenance.
    if any(_hash(path)!=original_hashes[role] for role,path in source_files.items()):
        raise ValueError("Model, samples or labels changed during reference evaluation")
    dest=Path(output_dir)
    with immutable_evidence_directory(dest) as staging:
        outputs=staging/"reference_outputs.npy"
        np.save(outputs,reference)
        result={
           "schema":"fpgai-research/reference-evaluation/v1",
           "evidence":"E0-NumPy-reference-only",
           "model_sha256":original_hashes.get("model"),
           "inputs_sha256":original_hashes["inputs"],
           "labels_sha256":original_hashes.get("labels"),
           "outputs_sha256":_hash(outputs),
           "samples":int(len(samples)),
           "sample_shape":list(expected_shape),
           "output_shape":list(model.tensor(output_name).shape),
           "numpy_version":np.__version__,
           "accuracy_top1":float(np.mean(predictions==labels)) if labels is not None else None,
           "correct_top1":int(np.count_nonzero(predictions==labels)) if labels is not None else None,
           "labels_present":labels is not None,
           "onnxruntime_crosscheck":ort_result,
           "disclaimer":"Software reference; not compiled fixed-point, Vitis, or FPGA evidence",
        }
        (staging/"reference_evaluation.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result
