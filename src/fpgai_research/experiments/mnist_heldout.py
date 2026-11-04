"""Full MNIST held-out inference agreement for NEWLY trained academic models.

Independent E0 software-only re-evaluation of all 10,000 official MNIST
test samples with trained PyTorch weights and the identical ONNX model.
Never infer FPGA fixed-point accuracy from this result or the 16 CSim cases.
No sample logits, images or trained weights are exported in its JSON.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import numpy as np

from fpgai_research.experiments.immutable_evidence import immutable_evidence_directory
from fpgai_research.experiments.mnist_assets import _validate as validate_trained

SCHEMA="fpgai-research/mnist-full-heldout-onnx-pytorch/v1"
COUNT=10000
CLASS_COUNT=10


def build_mnist_network(kind):
    """Reuse the exact training module architecture, never load foreign code."""
    from torch import nn
    if kind=="mlp":
        return nn.Sequential(nn.Flatten(),nn.Linear(784,128),
                             nn.ReLU(),nn.Linear(128,10))
    if kind=="cnn":
        return nn.Sequential(nn.Conv2d(1,4,3),nn.ReLU(),
                             nn.MaxPool2d(2),nn.Flatten(),nn.Linear(676,10))
    raise ValueError("Only canonical paper MLP/CNN are supported")


def summarize_full_heldout(kind,labels,pytorch_probabilities,
                           onnx_probabilities,*,metadata,
                           hashes:dict,versions:dict)->dict:
    """Fail closed on sample counts/types/metadata; no synthetic accuracy."""
    if kind not in ("mlp","cnn"):
        raise ValueError("Unknown trained paper model")
    y=np.asarray(labels)
    torch_p=np.asarray(pytorch_probabilities)
    onnx_p=np.asarray(onnx_probabilities)
    if (y.shape!=(COUNT,) or y.dtype.kind not in ("i","u")
            or np.any(y<0) or np.any(y>=CLASS_COUNT)
            or torch_p.shape!=(COUNT,CLASS_COUNT)
            or onnx_p.shape!=torch_p.shape
            or torch_p.dtype!=np.float32 or onnx_p.dtype!=np.float32
            or not np.all(np.isfinite(torch_p))
            or not np.all(np.isfinite(onnx_p))):
        raise ValueError("Expected the full 10000-sample, finite float32 top-10 study")
    if (np.any(torch_p<0) or np.any(onnx_p<0)
            or not np.allclose(torch_p.sum(axis=1),1.,atol=1e-4,rtol=0.)
            or not np.allclose(onnx_p.sum(axis=1),1.,atol=1e-4,rtol=0.)):
        raise ValueError("Expected normalized classification probabilities")
    if (not isinstance(metadata,dict)
            or metadata.get("schema")!="fpgai-research/trained-mnist/v1"
            or metadata.get("kind")!=kind
            or metadata.get("test_samples")!=COUNT
            or metadata.get("historical_original_weights") is not False
            or type(metadata.get("full_test_top1_accuracy")) is not float
            or not isfinite(metadata["full_test_top1_accuracy"])
            or not 0<=metadata["full_test_top1_accuracy"]<=1):
        raise ValueError("Missing original, bounded trained-model full-test claim")
    if (not isinstance(hashes,dict)
            or not {"model_onnx","weights_pt","mnist_test_pixels","mnist_test_labels"}.issubset(hashes)
            or any(not isinstance(v,str) or len(v)!=64
                   or any(ch not in "0123456789abcdef" for ch in v)
                   for v in hashes.values())
            or not isinstance(versions,dict)
            or not {"torch","torchvision","onnxruntime","numpy"}.issubset(versions)
            or any(not isinstance(v,str) or not v for v in versions.values())):
        raise ValueError("Full-test data/model/tool fingerprint incomplete")
    torch_pred=np.argmax(torch_p,axis=1)
    onnx_pred=np.argmax(onnx_p,axis=1)
    torch_correct=int(np.count_nonzero(torch_pred==y))
    onnx_correct=int(np.count_nonzero(onnx_pred==y))
    disagreements=int(np.count_nonzero(torch_pred!=onnx_pred))
    source_accuracy=metadata["full_test_top1_accuracy"]
    training_count_matches=abs(source_accuracy-torch_correct/COUNT)<=1e-12
    deviation=np.abs(torch_p.astype(np.float64)-onnx_p.astype(np.float64))
    return {
        "schema":SCHEMA,"evidence":"E0-independent-full-heldout-software",
        "model_kind":kind,"test_set":"torchvision MNIST official test split",
        "test_samples":COUNT,"num_classes":CLASS_COUNT,
        "pytorch_correct_top1":torch_correct,
        "onnxruntime_correct_top1":onnx_correct,
        "pytorch_top1_accuracy":torch_correct/COUNT,
        "onnxruntime_top1_accuracy":onnx_correct/COUNT,
        "top1_prediction_disagreement_count":disagreements,
        "top1_prediction_agreement_rate":1-disagreements/COUNT,
        "max_probability_absolute_difference":float(np.max(deviation)),
        "mean_probability_absolute_difference":float(np.mean(deviation)),
        "recorded_training_full_test_top1_accuracy":source_accuracy,
        "recorded_training_accuracy_reverified":training_count_matches,
        "onnx_and_pytorch_top1_predictions_identical":disagreements==0,
        "input_and_model_hashes":dict(sorted(hashes.items())),
        "tool_versions":dict(sorted(versions.items())),
        "model_ir_full_test_qualified":False,
        "vitis_fixed_point_full_test_qualified":False,
        "fpga_inference_latency_measured":False,
        "native_fpga_tools_executed":False,
        "historical_original_weights":False,
        "raw_weights_images_logits_or_predictions_exported":False,
        "limitations":(
            "This is the original held-out MNIST 10000-image software test set "
            "for newly trained weights. It verifies PyTorch and ONNX Runtime "
            "classification predictions with distinct CPU execution engines. "
            "It does not verify full-set FPGA fixed-point CSim, HLS numerical "
            "output, Model IR full-set semantics, native E4 correctness, "
            "board inference runtime or historical trained model accuracy. "
            "Small 16-sample native CSim and E4 routed QoR are separate evidence."
        ),
    }


def evaluate_mnist_full_heldout(trained_dir, dataset_dir, output,
                                *,download=False,batch_size=128):
    """Read-only model + fixed official test split; opt-in MNIST download."""
    if type(download) is not bool or type(batch_size) is not int or not 1<=batch_size<=1024:
        raise ValueError("Invalid full-test evaluation parameters")
    root=Path(trained_dir)
    if root.is_symlink():
        raise ValueError("Trained MNIST inputs must be a regular private directory")
    if not root.is_dir():
        raise FileNotFoundError(
            f"Trained model directory does not exist: {root}. "
            "The README example path is not guaranteed to exist. "
            "Discover original model.onnx/weights.pt sources with "
            "python experiments/locate_mnist_heldout_assets.py --search-root build; "
            "do not retrain unrelated weights and attribute them to prior E4 data.")
    meta_path=root/"metadata.json"
    if meta_path.is_symlink() or not meta_path.is_file():
        raise ValueError("Missing regular trained-model metadata")
    metadata_bytes=meta_path.read_bytes()
    metadata_sha=sha256(metadata_bytes).hexdigest()
    metadata=json.loads(metadata_bytes.decode("utf-8"))
    kind=metadata.get("kind")
    if kind not in ("mlp","cnn"):
        raise ValueError("Unknown trained model")
    validate_trained(root,kind,check_onnx=True)
    weights=root/"weights.pt"
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("Missing/linked trained PyTorch weights")
    src_sha=sha256(weights.read_bytes()).hexdigest()
    if metadata.get("sha256",{}).get("weights.pt")!=src_sha:
        raise ValueError("Trained weights differ from original training manifest")
    onnx_path=root/"model.onnx"
    orig_model=sha256(onnx_path.read_bytes()).hexdigest()
    dataset_root=Path(dataset_dir)
    if dataset_root.is_symlink():
        raise ValueError("Do not evaluate a linked official test directory")

    try:
        import torch
        import torchvision
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install torch, torchvision and onnxruntime for software accuracy") from exc
    # The original trainer uses torchvision's canonical ToTensor pixel scale.
    test=torchvision.datasets.MNIST(str(dataset_root),train=False,download=download)
    if len(test)!=COUNT:
        raise ValueError("Expected all 10000 official MNIST test images")
    pixel_tensor=test.data
    label_tensor=test.targets
    if (tuple(pixel_tensor.shape)!=(COUNT,28,28)
            or pixel_tensor.dtype!=torch.uint8
            or tuple(label_tensor.shape)!=(COUNT,)):
        raise ValueError("Unrecognized MNIST test-set layout")
    raw_pixels=pixel_tensor.numpy()
    labels=label_tensor.numpy()
    if labels.dtype.kind not in ("i","u"):
        raise ValueError("Expected integral MNIST test labels")
    inputs=np.expand_dims(raw_pixels.astype(np.float32)/255.0,axis=1)
    inputs=np.ascontiguousarray(inputs,dtype=np.float32)
    network=build_mnist_network(kind)
    try:
        state=torch.load(str(weights),map_location="cpu",weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Safe weights_only PyTorch loading is required") from exc
    network.load_state_dict(state,strict=True)
    network.eval()
    torch_outputs=[]
    with torch.inference_mode():
        for start in range(0,COUNT,batch_size):
            sample=torch.from_numpy(inputs[start:start+batch_size])
            scores=network(sample)
            if tuple(scores.shape)!=(len(sample),CLASS_COUNT):
                raise ValueError("Unexpected trained Torch classifier outputs")
            probability=torch.softmax(scores,dim=1)
            torch_outputs.append(probability.cpu().numpy())
    expected=np.concatenate(torch_outputs,axis=0)
    session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"])
    if (len(session.get_inputs())!=1 or len(session.get_outputs())!=1
            or tuple(session.get_inputs()[0].shape)!=(1,1,28,28)
            or tuple(session.get_outputs()[0].shape)!=(1,CLASS_COUNT)):
        raise ValueError("Expected static ONNX batch-one MNIST model with ten outputs")
    name=session.get_inputs()[0].name
    observed=[]
    for index in range(COUNT):
        out=session.run(None,{name:inputs[index:index+1]})
        if (len(out)!=1 or not isinstance(out[0],np.ndarray)
                or out[0].dtype!=np.float32
                or out[0].shape!=(1,CLASS_COUNT)):
            raise ValueError("Invalid ONNX Runtime full-set output")
        observed.append(out[0][0])
    actual=np.stack(observed,axis=0)
    hashes={
        "model_onnx":orig_model,"weights_pt":src_sha,
        "mnist_test_pixels":sha256(raw_pixels.tobytes()).hexdigest(),
        "mnist_test_labels":sha256(labels.tobytes()).hexdigest(),
    }
    versions={
        "torch":str(torch.__version__),
        "torchvision":str(torchvision.__version__),
        "onnxruntime":str(ort.__version__),
        "numpy":str(np.__version__),
    }
    result=summarize_full_heldout(
        kind,labels,expected,actual,metadata=metadata,
        hashes=hashes,versions=versions)
    if (sha256(weights.read_bytes()).hexdigest()!=src_sha
            or sha256(onnx_path.read_bytes()).hexdigest()!=orig_model
            or sha256(meta_path.read_bytes()).hexdigest()!=metadata_sha
            or sha256(raw_pixels.tobytes()).hexdigest()!=hashes["mnist_test_pixels"]
            or sha256(labels.tobytes()).hexdigest()!=hashes["mnist_test_labels"]):
        raise ValueError("Trained model or held-out MNIST data mutated during run")
    # Selected E0 conformance samples and checksum policy must also remain
    # source-attested throughout the full held-out run.
    validate_trained(root,kind,check_onnx=False)
    with immutable_evidence_directory(output) as staging:
        (staging/"mnist_full_heldout_accuracy.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
    return result
