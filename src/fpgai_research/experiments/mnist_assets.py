"""Package newly trained MNIST MLP/CNN models as a private provenance-checked suite.

Never claim historical paper weights or native FPGA measurements were reproduced.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import numpy as np

FILES=("model.onnx","inputs.npy","labels.npy","metadata.json")

def _sha(path:Path)->str:
    h=sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(1048576),b""):
            h.update(data)
    return h.hexdigest()

def _validate(folder:Path,kind:str,*,check_onnx:bool)->dict:
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"Missing or linked trained {kind} directory")
    for name in FILES:
        path=folder/name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked trained {kind}/{name}")
    metadata=json.loads((folder/"metadata.json").read_text())
    if (metadata.get("schema")!="fpgai-research/trained-mnist/v1" or
        metadata.get("kind")!=kind or metadata.get("historical_original_weights") is not False):
        raise ValueError(f"Incorrect or unsupported trained {kind} metadata")
    hashes=metadata.get("sha256")
    if not isinstance(hashes,dict):
        raise ValueError("Training metadata needs source hashes")
    actual={name:_sha(folder/name) for name in FILES[:3]}
    if any(hashes.get(name)!=digest for name,digest in actual.items()):
        raise ValueError(f"Trained {kind} source checksum mismatch")
    samples=np.load(folder/"inputs.npy",allow_pickle=False)
    labels=np.load(folder/"labels.npy",allow_pickle=False)
    if (samples.dtype!=np.float32 or samples.ndim not in (4,5)
        or tuple(samples.shape[-3:])!=(1,28,28)
        or (samples.ndim==5 and samples.shape[1]!=1)
        or not 1<=len(samples)<=256 or not np.all(np.isfinite(samples))):
        raise ValueError("MNIST evaluation samples require finite float32 N x 1 x 28 x 28")
    if (labels.ndim!=1 or labels.dtype.kind not in ("i","u")
        or len(labels)!=len(samples) or np.any(labels<0) or np.any(labels>=10)):
        raise ValueError("MNIST labels must be integer classes 0..9 aligned to samples")
    if (metadata.get("evaluation_indices")!=list(range(len(samples))) or
        metadata.get("test_samples",0)<len(samples)):
        raise ValueError("Selected evaluation indices disagree with training metadata")
    conformance=metadata.get("onnx_conformance")
    if (not isinstance(conformance,dict) or
        conformance.get("schema")!="fpgai-research/trained-onnx-conformance/v1" or
        conformance.get("evidence")!="E0-PyTorch-ONNXRuntime-ModelIR" or
        conformance.get("model_sha256")!=actual["model.onnx"] or
        conformance.get("evaluated_samples")!=len(samples) or
        conformance.get("model_ir_compared") is not True or
        conformance.get("exact_float32_inputs") is not True or
        conformance.get("class_predictions_matched") is not True or
        conformance.get("native_vitis_executed") is not False or
        conformance.get("vivado_executed") is not False):
        raise ValueError("Trained model lacks attributed PyTorch/ONNXRuntime/Model IR E0 conformance")
    verified=False
    if check_onnx:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError("Install onnx to verify trained research models") from exc
        onnx.checker.check_model(str(folder/"model.onnx"))
        verified=True
    return {"files_sha256":actual,"sample_count":len(samples),
            "onnx_checker_passed":verified,"onnx_e0_conformance_recorded":True,
            "historical_original_weights":False}

def assemble_mnist_suite(mlp_dir:str|Path,cnn_dir:str|Path,destination:str|Path,
                         *,check_onnx:bool=True)->dict:
    if type(check_onnx) is not bool:
        raise ValueError("ONNX checker policy must be boolean")
    source={"mlp":Path(mlp_dir),"cnn":Path(cnn_dir)}
    checked={key:_validate(path,key,check_onnx=check_onnx)
             for key,path in source.items()}
    if checked["mlp"]["files_sha256"]["model.onnx"]==checked["cnn"]["files_sha256"]["model.onnx"]:
        raise ValueError("MLP and CNN must contain distinct trained models")
    target=Path(destination)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing research suite")
    with TemporaryDirectory(dir=target.parent,prefix=".mnist-suite-") as staging:
        root=Path(staging)
        for kind,path in source.items():
            folder=root/"assets"/f"mnist_{kind}"
            folder.mkdir(parents=True)
            for filename in FILES:
                shutil.copyfile(path/filename,folder/filename)
            for name,digest in checked[kind]["files_sha256"].items():
                if _sha(folder/name)!=digest:
                    raise ValueError("Training input changed during suite assembly")
        rows=[{"id":f"mnist_{kind}","family":kind,
               "onnx":f"assets/mnist_{kind}/model.onnx",
               "input":f"assets/mnist_{kind}/inputs.npy",
               "labels":f"assets/mnist_{kind}/labels.npy"} for kind in ("mlp","cnn")]
        suite={"schema":"fpgai-research/full-paper-suite/v1",
               "notes":"Newly trained MNIST models only; not original paper weights",
               "experiments":rows}
        (root/"suite.json").write_text(json.dumps(suite,indent=2,sort_keys=True)+"\n")
        report={"schema":"fpgai-research/mnist-training-suite/v1",
                "model_count":2,"original_historical_weights":False,
                "native_vendor_evidence":False,
                "private_model_weights_included":True,"models":checked}
        (root/"provenance.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        for kind,path in source.items():
            if any(_sha(path/name)!=checked[kind]["files_sha256"][name]
                   for name in FILES[:3]):
                raise ValueError("Training input changed during suite assembly")
        root.rename(target)
    return report
