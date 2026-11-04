"""Bounded, read-only discovery of private trained MNIST evaluation assets.

Use this before full 10000-image held-out evaluation. The native HLS E3/E4
workspace holds generated source and reports, NOT necessarily original
PyTorch weights.pt. Never mistake an unrelated freshly trained model for the
one used in an earlier physical study.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA="fpgai-research/mnist-trained-asset-discovery/v1"
_MAX_JSON_BYTES=256*1024
_REQUIRED=("model.onnx","inputs.npy","labels.npy","weights.pt")
_IGNORED={"vivado","vitis","_vitis","_vivado","csim","syn","impl",
          "node_modules",".git",".venv","venv","reports","logs","hls"}
_MODES=("mlp","cnn")


def discover_trained_mnist(root,*,max_depth=9,max_directories=15000)->dict:
    """Search only an explicit root without following links or loading models."""
    if (type(max_depth) is not int or not 0<=max_depth<=30
            or type(max_directories) is not int or not 1<=max_directories<=100000):
        raise ValueError("Unsafe model discovery bounds")
    base=Path(root)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("Search root must be an existing regular directory")
    candidates=[]
    scanned=0
    truncated=False
    for parent,dirs,files in os.walk(base,topdown=True,followlinks=False):
        folder=Path(parent)
        scanned+=1
        if scanned>max_directories:
            truncated=True
            break
        depth=len(folder.relative_to(base).parts)
        if depth>=max_depth:
            dirs[:]=[]
        else:
            dirs[:]=sorted(
                name for name in dirs
                if name not in _IGNORED
                and not (folder/name).is_symlink()
            )
        if "metadata.json" not in files:
            continue
        path=folder/"metadata.json"
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size>_MAX_JSON_BYTES):
            continue
        try:
            meta=json.loads(path.read_text(encoding="utf-8"))
        except (ValueError,UnicodeDecodeError,OSError):
            continue
        if (not isinstance(meta,dict)
                or meta.get("schema")!="fpgai-research/trained-mnist/v1"
                or meta.get("kind") not in _MODES):
            continue
        status={}
        for name in _REQUIRED:
            f=folder/name
            status[name]="REGULAR" if f.is_file() and not f.is_symlink() else (
                "LINKED_UNSAFE" if f.is_symlink() else "MISSING")
        complete=all(v=="REGULAR" for v in status.values())
        candidates.append({
            "kind":meta["kind"],
            "directory":str(folder),
            "status":("POTENTIAL_FULL_HELDOUT_INPUTS" if complete
                      else "INCOMPLETE_TRAINED_ASSETS"),
            "files":status,
            "historical_original_weights":meta.get("historical_original_weights"),
            "source_attested_to_e3_e4":False,
            "model_and_weights_hashes_reverified":False,
        })
    candidates.sort(key=lambda row:(row["kind"],row["directory"]))
    return {
        "schema":SCHEMA,"searched_root":str(base),
        "scanned_directories":min(scanned,max_directories),
        "scan_truncated":truncated,"candidate_count":len(candidates),
        "potentially_complete_count":sum(
            r["status"]=="POTENTIAL_FULL_HELDOUT_INPUTS" for r in candidates),
        "candidates":candidates,"native_tools_executed":False,
        "model_files_read":False,"raw_weights_exported":False,
        "guidance":(
            "Only POTENTIAL_FULL_HELDOUT_INPUTS directories with original "
            "weights.pt, model.onnx, selected inputs and metadata may be used "
            "for independent 10000-image PyTorch/ONNX Runtime evaluation. "
            "The evaluator will still check hashes and numerical provenance. "
            "This locator does not prove that a candidate was used by the "
            "original native E3/E4 campaign. Absent original weights, full "
            "PyTorch-vs-ONNX comparison of that exact model is unavailable; "
            "new training is a DIFFERENT study and must not be assigned to "
            "prior routed implementation measurements."
        ),
    }
