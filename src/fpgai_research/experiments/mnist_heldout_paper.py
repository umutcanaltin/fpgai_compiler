"""Source-bound E0 MNIST held-out paper table; NEVER native FPGA accuracy.

Private trained files and the selected inputs from the original assembled
paper suite must revalidate before a 10000-image PyTorch/ORT receipt enters
the manuscript. This rechecks receipts and exact source hashes; it does NOT
reexecute inference and does NOT assert that E0 accuracy was observed at E4.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.mnist_assets import _validate as validate_trained

SCHEMA="fpgai-research/mnist-heldout-paper-e0/v1"
EVAL_SCHEMA="fpgai-research/mnist-full-heldout-onnx-pytorch/v1"
KINDS=("mlp","cnn")
_SOURCE_FILES=("model.onnx","inputs.npy","labels.npy","metadata.json","weights.pt")
_SUITE_FILES=("model.onnx","inputs.npy","labels.npy","metadata.json")
_DEFAULT_TRAINED={
    "mlp":"build/trained_mlp_20261024",
    "cnn":"build/trained_cnn_20261024",
}
_DEFAULT_RECEIPTS={
    "mlp":"build/mnist_mlp_heldout_20261026",
    "cnn":"build/mnist_cnn_heldout_20261026",
}
DEFAULT_SUITE="build/mnist_suite_20261024"


def _regular(path:Path)->bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>64*1024*1024:
        raise ValueError("Source-attested E0 evidence file missing, linked or oversized")
    return path.read_bytes()


def _sha(path:Path)->str:
    return sha256(_regular(path)).hexdigest()


def _read_json(path:Path)->dict:
    data=_regular(path)
    if len(data)>1048576:
        raise ValueError("Oversized E0 paper receipt")
    value=json.loads(data.decode("utf-8"))
    if not isinstance(value,dict):
        raise ValueError("E0 evidence receipt must be JSON object")
    return value


def _probability(v)->bool:
    return type(v) in (float,int) and isfinite(v) and 0<=v<=1


def collect_heldout_paper(root)->dict:
    """Recheck both private source models, full-set receipts and suite copies."""
    base=Path(root)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("Missing regular private manuscript repository")
    suite=base/DEFAULT_SUITE
    if suite.is_symlink() or not suite.is_dir():
        raise ValueError("Missing regular original private assembled MNIST suite")
    suite_record=_read_json(suite/"suite.json")
    if suite_record.get("schema")!="fpgai-research/full-paper-suite/v1":
        raise ValueError("Unexpected assembled paper suite schema")
    entries=suite_record.get("experiments")
    if (not isinstance(entries,list) or len(entries)!=2 or
            {r.get("id") for r in entries if isinstance(r,dict)}
            !={"mnist_mlp","mnist_cnn"}):
        raise ValueError("Incorrect two-model paper suite inventory")
    records=[]
    shared_test=None
    onnx_digests=set()
    for kind in KINDS:
        trained=base/_DEFAULT_TRAINED[kind]
        receipt_dir=base/_DEFAULT_RECEIPTS[kind]
        copy=suite/"assets"/f"mnist_{kind}"
        if (trained.is_symlink() or not trained.is_dir() or
                receipt_dir.is_symlink() or not receipt_dir.is_dir() or
                copy.is_symlink() or not copy.is_dir()):
            raise ValueError("Missing regular private trained/suite/evaluation directory")
        checked=validate_trained(trained,kind,check_onnx=False)
        if (checked.get("onnx_e0_conformance_recorded") is not True or
                checked.get("historical_original_weights") is not False):
            raise ValueError("Original trained ONNX conformance is not qualified")
        meta=_read_json(trained/"metadata.json")
        original_hashes=meta.get("sha256")
        if (not isinstance(original_hashes,dict) or
                any(original_hashes.get(name)!=_sha(trained/name)
                    for name in _SOURCE_FILES if name!="metadata.json")):
            raise ValueError("Trained source does not match original training manifest")
        for name in _SUITE_FILES:
            if _sha(trained/name)!=_sha(copy/name):
                raise ValueError("Trained model differs from the assembled paper suite")
        designated=next(row for row in entries if row["id"]==f"mnist_{kind}")
        if (designated.get("onnx")!=f"assets/mnist_{kind}/model.onnx"
                or designated.get("input")!=f"assets/mnist_{kind}/inputs.npy"
                or designated.get("labels")!=f"assets/mnist_{kind}/labels.npy"):
            raise ValueError("Suite manifest points at different or unsafe study inputs")
        receipt=_read_json(receipt_dir/"mnist_full_heldout_accuracy.json")
        digests=receipt.get("input_and_model_hashes")
        if (not isinstance(digests,dict) or
                digests.get("model_onnx")!=_sha(trained/"model.onnx")
                or digests.get("weights_pt")!=_sha(trained/"weights.pt")):
            raise ValueError("Full-set accuracy receipt is from different trained weights")
        pair=(digests.get("mnist_test_pixels"),digests.get("mnist_test_labels"))
        if any(not isinstance(v,str) or len(v)!=64 or
               any(ch not in "0123456789abcdef" for ch in v) for v in pair):
            raise ValueError("Invalid official test dataset fingerprints")
        if shared_test is not None and pair!=shared_test:
            raise ValueError("CNN and MLP evaluated on different held-out MNIST datasets")
        shared_test=pair
        if (receipt.get("schema")!=EVAL_SCHEMA
                or receipt.get("evidence")!="E0-independent-full-heldout-software"
                or receipt.get("model_kind")!=kind
                or receipt.get("test_set")!="torchvision MNIST official test split"
                or receipt.get("test_samples")!=10000
                or receipt.get("num_classes")!=10
                or receipt.get("recorded_training_accuracy_reverified") is not True
                or receipt.get("onnx_and_pytorch_top1_predictions_identical") is not True
                or receipt.get("top1_prediction_disagreement_count")!=0
                or receipt.get("top1_prediction_agreement_rate")!=1.0
                or receipt.get("native_fpga_tools_executed") is not False
                or receipt.get("vitis_fixed_point_full_test_qualified") is not False
                or receipt.get("model_ir_full_test_qualified") is not False
                or receipt.get("fpga_inference_latency_measured") is not False
                or receipt.get("historical_original_weights") is not False
                or receipt.get("raw_weights_images_logits_or_predictions_exported") is not False):
            raise ValueError("Full MNIST E0 evidence is unqualified or misattributed")
        correct=receipt.get("pytorch_correct_top1")
        onnx_correct=receipt.get("onnxruntime_correct_top1")
        accuracy=receipt.get("pytorch_top1_accuracy")
        if (type(correct) is not int or type(onnx_correct) is not int
                or not 0<=correct<=10000 or correct!=onnx_correct
                or not _probability(accuracy)
                or accuracy!=correct/10000
                or receipt.get("onnxruntime_top1_accuracy")!=accuracy
                or receipt.get("recorded_training_full_test_top1_accuracy")!=
                   meta.get("full_test_top1_accuracy")
                or accuracy!=meta.get("full_test_top1_accuracy")):
            raise ValueError("Trained metadata and independent held-out accuracy disagree")
        maximum=receipt.get("max_probability_absolute_difference")
        average=receipt.get("mean_probability_absolute_difference")
        if (type(maximum) not in (int,float) or not isfinite(maximum)
                or type(average) not in (int,float) or not isfinite(average)
                or not 0<=average<=maximum<=1):
            raise ValueError("Nonfinite or inconsistent PyTorch/ONNX numerical discrepancy")
        if digests["model_onnx"] in onnx_digests:
            raise ValueError("CNN/MLP cannot share identical trained ONNX")
        onnx_digests.add(digests["model_onnx"])
        records.append({
            "model":kind,
            "test_samples":10000,
            "correct_top1_pytorch":correct,
            "correct_top1_onnxruntime":onnx_correct,
            "top1_accuracy_pytorch":accuracy,
            "top1_accuracy_onnxruntime":accuracy,
            "prediction_disagreements":0,
            "max_probability_absolute_difference":maximum,
            "mean_probability_absolute_difference":average,
            "original_training_metadata_accuracy_reverified":True,
            "source_identical_to_assembled_suite":True,
            "same_official_heldout_set_as_other_model":True,
            "native_fpga_full_set_accuracy_measured":False,
        })
    return {
        "schema":SCHEMA,"evidence":"E0-PyTorch-ONNXRuntime-software-only",
        "model_count":2,"heldout_sample_count_per_model":10000,
        "both_original_training_accuracies_reverified":True,
        "both_source_identical_to_assembled_paper_suite":True,
        "both_evaluated_on_same_official_mnist_test_set":True,
        "all_predictions_pytorch_onnx_agree":True,
        "models":records,"vendor_tools_executed_by_reaudit":False,
        "native_e3_e4_weight_identity_verified":False,
        "fixed_point_full_test_accuracy_measured":False,
        "fpga_board_accuracy_or_latency_measured":False,
        "private_hashes_weights_or_predictions_exported":False,
        "limitations":(
            "Reaudits published held-out E0 receipts against the original trained "
            "model/weights and assembled paper-suite ONNX copies. This is a "
            "source-hash and receipt check, NOT a rerun of 10000 CPU inferences "
            "or native FPGA classification. The physical E3/E4 workspace uses "
            "generated source; a direct identity proof from trained ONNX to "
            "the native implementation remains separate. No E2/E4 full-test "
            "accuracy, board latency or hls4ml comparison follows."
        ),
    }


def render_heldout_latex(data:dict)->str:
    if (data.get("schema")!=SCHEMA or data.get("model_count")!=2
            or data.get("native_e3_e4_weight_identity_verified") is not False
            or data.get("fixed_point_full_test_accuracy_measured") is not False
            or data.get("private_hashes_weights_or_predictions_exported") is not False):
        raise ValueError("Only bounded E0 software accuracy can enter the paper")
    rows=data.get("models")
    if not isinstance(rows,list) or [r.get("model") for r in rows]!=list(KINDS):
        raise ValueError("Incomplete or reordered MNIST held-out software data")
    lines=[
        "% Independent, receipt-verified E0 software classification; never E4 accuracy.",
        r"\begin{table}[t]",r"\centering\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Model & Test images & Correct & Top-1 (\%) & Disagreements \\",
        r"\midrule",
    ]
    for record in rows:
        if (record.get("test_samples")!=10000
                or record.get("correct_top1_pytorch")!=record.get("correct_top1_onnxruntime")
                or record.get("prediction_disagreements")!=0
                or record.get("source_identical_to_assembled_suite") is not True):
            raise ValueError("Unqualified full-set software accuracy row")
        correct=record["correct_top1_pytorch"]
        if (type(correct) is not int or not 0<=correct<=10000 or
                record.get("top1_accuracy_pytorch")!=correct/10000):
            raise ValueError("Accuracy is not supported by labeled counts")
        label={"mlp":"MLP","cnn":"CNN"}[record["model"]]
        lines.append(f"{label} & 10000 & {correct} & {100*correct/10000:.2f} & 0"+r" \\")
    lines.extend([
        r"\bottomrule",r"\end{tabular}",
        r"\caption{Independent full-test software accuracy on the official "
        r"10,000-image MNIST test split. PyTorch and ONNX Runtime top-1 "
        r"predictions agree for all samples; no fixed-point FPGA "
        r"classification accuracy or device latency is implied.}",
        r"\label{tab:fpgai-mnist-heldout-e0}",r"\end{table}","",
    ])
    return "\n".join(lines)


def export_heldout_paper(root,output)->dict:
    result=collect_heldout_paper(root)
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Immutable full MNIST E0 paper export already exists")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-mnist-heldout-paper-",dir=dest.parent) as temp:
        stage=Path(temp)
        (stage/"heldout_accuracy.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        (stage/"heldout_accuracy.tex").write_text(render_heldout_latex(result))
        lines=["# MNIST full held-out E0 software accuracy","",
               "This is **not** Vitis fixed-point or FPGA accuracy.","",
               "| Model | Samples | Correct | Top-1 | PyTorch/ONNX disagreements |",
               "| --- | ---: | ---: | ---: | ---: |"]
        for row in result["models"]:
            lines.append(
                f"| {row['model'].upper()} | 10000 | {row['correct_top1_pytorch']} "
                f"| {row['top1_accuracy_pytorch']*100:.2f}% | 0 |")
        lines.extend(["",result["limitations"],""])
        (stage/"heldout_accuracy.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Concurrent held-out evidence publication")
        stage.rename(dest)
    return result
