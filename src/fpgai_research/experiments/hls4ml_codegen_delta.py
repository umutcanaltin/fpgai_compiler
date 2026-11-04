"""Read-only fingerprint comparison of hls4ml generated model C++ AND headers.

Previous E1 comparisons checked generated *.cpp inventories only. Reuse
and precision semantics can live in firmware/parameters.h or defines.h,
so a 0-C++ diff cannot prove code generation was unchanged. This audit
never prints model types, source text, weight identities or private paths.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json

SCHEMA = "fpgai-research/hls4ml-generated-firmware-delta/v1"
MAX_FILE = 16*1024*1024
ALLOWED = frozenset({".cpp", ".h", ".hpp", ".cc", ".cxx"})


def _model_files(project):
    root=Path(project)
    receipt=root/"research_baseline.json"
    firmware=root/"firmware"
    if (root.is_symlink() or not root.is_dir()
            or receipt.is_symlink() or not receipt.is_file()
            or firmware.is_symlink() or not firmware.is_dir()):
        raise ValueError("Expected ordinary attested E1 project")
    meta=json.loads(receipt.read_text(encoding="utf-8"))
    if (meta.get("schema")!="fpgai-research/hls4ml-baseline/v2"
            or meta.get("evidence")!="E1-source-only"):
        raise ValueError("Not a source-attested E1 project")
    files={}
    # Direct firmware/ files are written for this model; never recurse into
    # bundled vendor headers, architecture libraries or trained weights.
    for path in sorted(firmware.iterdir()):
        if path.is_symlink():
            raise ValueError("Generated firmware contains a linked entry")
        if not path.is_file() or path.suffix.lower() not in ALLOWED:
            continue
        if path.stat().st_size > MAX_FILE:
            raise ValueError("Oversized generated firmware file")
        files[path.name]=sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError("No model-owned generated firmware files")
    return meta,files


def collect_firmware_delta(global_project, name_rf1_project, name_rf2_project):
    trio=[_model_files(p) for p in (
        global_project,name_rf1_project,name_rf2_project)]
    g,a,b=[i[0] for i in trio]
    must_match=("model_sha256","part","clock_ns","precision","reuse_factor",
                "backend","io_type","hls4ml_version")
    if any(any(item.get(key)!=g.get(key) for key in must_match)
           for item in (a,b)):
        raise ValueError("Generated firmware comparison has incompatible model/target")
    inventories=[i[1] for i in trio]
    if not set(inventories[0])==set(inventories[1])==set(inventories[2]):
        raise ValueError("Different generated model-firmware file inventories")
    def changed(left,right):
        keys=sorted(k for k in left if left[k]!=right[k])
        return {
            "changed_model_source_files":len(keys),
            "changed_cpp_implementation_files":
                sum(Path(k).suffix.lower() in {".cpp",".cc",".cxx"} for k in keys),
            "changed_model_header_files":
                sum(Path(k).suffix.lower() in {".h",".hpp"} for k in keys),
            # Only file types, not model identifiers or filenames, in stdout.
            "any_model_source_change":bool(keys),
        }
    return {
        "schema":SCHEMA,
        "same_original_trained_candidate_and_hardware_contract":True,
        "generated_firmware_file_count":len(inventories[0]),
        "global_vs_name_rf1":changed(inventories[0],inventories[1]),
        "name_rf1_vs_rf2":changed(inventories[1],inventories[2]),
        "original_model_source_modified":False,
        "trained_weight_bytes_rewritten":False,
        "header_difference_causes_numerical_error_proven":False,
        "native_vitis_or_vivado_executed":False,
        "external_hardware_qor_eligible":False,
    }


def export_firmware_delta(global_project, name_rf1_project,name_rf2_project,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Firmware delta output already exists")
    audit=collect_firmware_delta(global_project,name_rf1_project,name_rf2_project)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".hls4ml-firmware-delta-",dir=dest.parent) as tmp:
        stage=Path(tmp)/"audit"
        stage.mkdir()
        (stage/"firmware_delta_private.json").write_text(
            json.dumps(audit,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Firmware delta output appeared")
        stage.rename(dest)
    return audit
