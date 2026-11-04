"""Prepare a genuinely ordered numerical cohort from private source files.

Reexecutes the original trained ONNX graph in ONNX Runtime on EVERY selected
input; refuses stored reference arrays that disagree. Copies numeric outputs
to a fresh private directory with source SHA-256 provenance. Does not attest
that Vitis actually produced either alleged native-output file: those must be
independently tied to source-backed E2 receipts before publication.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.matched_compiler_numerics import (
    SCHEMA, compare_arrays, _load_numpy,
)

MAX=128*1024*1024


def _read(path, suffixes):
    file=Path(path)
    if (file.is_symlink() or not file.is_file() or
            file.suffix not in suffixes or
            not 0<file.stat().st_size<=MAX):
        raise ValueError("Missing safe private comparator input")
    return file


def _npy(path):
    file=_read(path,{".npy"})
    x=np.load(file,allow_pickle=False)
    if x.dtype.kind not in "fiu" or not np.all(np.isfinite(x)):
        raise ValueError("Invalid source numeric array")
    return x


def _predictions(path,n):
    file=_read(path,{".npy",".dat",".txt",".log"})
    if file.suffix==".npy":
        result=_npy(file)
    else:
        # No log messages, headers or incomplete rows may become predictions.
        lines=file.read_text(encoding="utf-8").strip().splitlines()
        if len(lines)!=n or n>10000:
            raise ValueError("Mismatched native-output row count")
        rows=[]
        for line in lines:
            cells=line.split()
            if len(cells)!=10:
                raise ValueError("Native output row must contain exactly 10 values")
            rows.append([float(x) for x in cells])
        result=np.asarray(rows,dtype=np.float64)
    if result.shape!=(n,10) or not np.all(np.isfinite(result)):
        raise ValueError("Invalid native numerical output matrix")
    return np.asarray(result,dtype=np.float64)


def _verify_onnx(source,inputs,reference,atol):
    import onnxruntime as ort
    session=ort.InferenceSession(source.read_bytes(),providers=["CPUExecutionProvider"])
    if len(session.get_inputs())!=1 or len(session.get_outputs())!=1:
        raise ValueError("Original ONNX comparator must have one input/output")
    entry=session.get_inputs()[0]
    errors=[]
    for i in range(len(inputs)):
        values=np.asarray(inputs[i:i+1],dtype=np.float32)
        got=session.run(None,{entry.name:values})
        if len(got)!=1 or np.asarray(got[0]).shape!=(1,10):
            raise ValueError("Original ONNX Runtime output shape unexpected")
        actual=np.asarray(got[0],dtype=np.float64)[0]
        if not np.all(np.isfinite(actual)):
            raise ValueError("Nonfinite original trained ONNX prediction")
        errors.append(float(np.max(np.abs(actual-reference[i]))))
    maximum=max(errors)
    if maximum>atol:
        raise ValueError("Stored ONNX reference mismatches original trained source")
    return maximum


def stage_matched_numeric(model, original_onnx, ordered_inputs, labels,
                          reference_outputs, fpgai_outputs, hls4ml_outputs,
                          output,*,reference_max_error=2e-5):
    if model not in ("mlp","cnn"):
        raise ValueError("Only trained MNIST MLP or CNN are supported")
    if (type(reference_max_error) not in (float,int)
            or not np.isfinite(reference_max_error)
            or reference_max_error<=0 or reference_max_error>1e-3):
        raise ValueError("Invalid independent ONNX verification tolerance")
    model_file=_read(original_onnx,{".onnx"})
    inputs=np.asarray(_npy(ordered_inputs))
    labels_array=np.asarray(_npy(labels))
    ref=np.asarray(_npy(reference_outputs),dtype=np.float64)
    if (inputs.dtype.kind!="f" or inputs.dtype!=np.float32
            or inputs.ndim<2 or not 16<=len(inputs)<=10000
            or labels_array.shape!=(len(inputs),)
            or labels_array.dtype.kind not in "iu"
            or np.any(labels_array<0) or np.any(labels_array>9)
            or ref.shape!=(len(inputs),10)):
        raise ValueError("Unmatched input/reference/label shape or precision")
    fp=_predictions(fpgai_outputs,len(inputs))
    hl=_predictions(hls4ml_outputs,len(inputs))
    summary=compare_arrays(ref,fp,hl,labels_array)
    max_e0_error=_verify_onnx(
        model_file,inputs,ref,float(reference_max_error))
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Matched numeric bundle must use a new destination")
    hashes={
        "original_onnx":sha256(model_file.read_bytes()).hexdigest(),
        "ordered_inputs":sha256(_read(ordered_inputs,{".npy"}).read_bytes()).hexdigest(),
        "source_labels":sha256(_read(labels,{".npy"}).read_bytes()).hexdigest(),
        "source_reference":sha256(_read(reference_outputs,{".npy"}).read_bytes()).hexdigest(),
        "source_fpgai":sha256(_read(fpgai_outputs,{".npy",".dat",".txt",".log"}).read_bytes()).hexdigest(),
        "source_hls4ml":sha256(_read(hls4ml_outputs,{".npy",".dat",".txt",".log"}).read_bytes()).hexdigest(),
    }
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".matched-paired-onnx-",dir=dest.parent) as tmp:
        root=Path(tmp)/"bundle"
        root.mkdir()
        arrs={"reference":ref,"fpgai":fp,"hls4ml":hl,
              "labels":labels_array.astype(np.int64)}
        model_copy=root/"original_model.onnx"
        model_copy.write_bytes(model_file.read_bytes())
        ordered_copy=root/"ordered_inputs.npy"
        np.save(ordered_copy,inputs,allow_pickle=False)
        specs={}
        for name,array in arrs.items():
            file=root/(name+".npy")
            np.save(file,array,allow_pickle=False)
            specs[name]={"file":file.name,
                         "sha256":sha256(file.read_bytes()).hexdigest()}
        manifest={
            "schema":SCHEMA,"model_kind":model,
            "original_onnx_sha256":hashes["original_onnx"],
            "original_onnx_file":"original_model.onnx",
            "ordered_input_cohort_sha256":
                sha256(ordered_copy.read_bytes()).hexdigest(),
            "ordered_input_cohort":{
                "file":ordered_copy.name,
                "sha256":sha256(ordered_copy.read_bytes()).hexdigest()},
            "output_semantics":"post_softmax_probability",
            "native_execution_independently_attested":False,
            "external_hardware_qor_claimed":False,
            "arrays":specs,
        }
        (root/"manifest.json").write_text(
            json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        provenance={
            "schema":"fpgai-research/matched-numeric-staging/v1",
            "source_sha256":hashes,
            "selected_samples":len(inputs),
            "onnx_reference_recomputed_all_samples":True,
            "max_original_onnx_vs_stored_probability_error":max_e0_error,
            "matched_array_shapes_and_labels_validated":True,
            "paired_source_files_attested_to_native_executions":False,
            "independent_original_weight_lineage_to_both_native_compilers":False,
            "external_e2_e3_e4_qualified":False,
            "observed_selected_cohort_strict_numeric_dominance":
                summary["selected_cohort_strict_fpgai_numeric_dominance_observed"],
        }
        (root/"staging_provenance.json").write_text(
            json.dumps(provenance,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Numeric staging appeared concurrently")
        root.rename(dest)
    return provenance
