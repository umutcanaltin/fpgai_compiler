"""R2z prospective full Conv2D -> multi-Dense neural source + HLS packages.

3 prespecified network topologies x 2 per-operator Phi policies
= 6 *complete, original, source-pinned* C++/Vitis HLS packages.
Two deterministic inputs per package = 12 ACTUAL GCC host E1 checks,
covering EVERY intermediate Conv/Pool/Flatten/Dense/activation tensor.

No vendor Vitis/Vivado calls. The generated Vitis packages are PREPARED
only; CSim/CSynth/physical outcomes remain UNKNOWN. Original R2i-R2y
studies and frozen RL are never touched.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from fpgai_research.compiler.neural_chain import compile_neural_chain
from fpgai_research.experiments.cst_r2z_neural_network_fixtures import (
    KINDS, network_architecture, network_fixture,
)
from fpgai_research.toolchains.vitis_hls.package import (
    make_testbench, make_vitis_tcl, write_vitis_package,
)
from fpgai_research.validation.neural_chain_host import (
    verify_neural_chain_gcc,
)

SCHEMA = "fpgai-research/cst-R2z-conv2d-dense-host-E1-original-study/v1"
AUDIT_SCHEMA = "fpgai-research/cst-R2z-read-only-package-source-audit/v1"
SOURCE_CASES = len(KINDS) * 2
HOST_INPUTS_PER_CASE = 2
PART = "xck26-sfvc784-2LV-c"
CLOCK_NS = 10.0
UNCERTAINTY_NS = 1.0
CSIM_ATOL = 0.01
HOST_ATOL = 0.0001
HOST_RTOL = 0.001
SEEDS = (1221, 4559)
CASE_FIELDS = ("case","network","independent_phi","operators",
               "source_cpp_sha256","functional_sha256",
               "neural_contract_sha256","original_input_batch_sha256",
               "source_package_sha256","prepared_vitis_tcl_sha256",
               "prepared_vitis_testbench_sha256","host_probes",
               "host_all_stages_qualified","host_max_absolute_error")


def _sha(data:bytes)->str:
    return sha256(data).hexdigest()


def _rev()->str:
    repo=Path(__file__).resolve().parents[3]
    clean=subprocess.run(
        ["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=10,
    ).stdout
    if clean.strip():
        raise RuntimeError("R2z original source requires clean tracked Git checkout")
    rev=subprocess.run(
        ["git","-C",str(repo),"rev-parse","HEAD"],
        capture_output=True,text=True,check=True,timeout=10,
    ).stdout.strip()
    if re.fullmatch(r"[a-f0-9]{40}",rev) is None:
        raise RuntimeError("R2z clean full git revision unavailable")
    return rev


def _input_batch(network:str)->np.ndarray:
    model=network_fixture(network)
    shape=model.tensor(model.inputs[0]).shape
    return np.stack([
        np.random.default_rng(seed).uniform(
            -.45,.45,size=shape).astype(np.float32)
        for seed in SEEDS
    ],axis=0)


def _records(*,run_host:bool)->list[tuple[dict,object,object,np.ndarray]]:
    records=[]
    for network in KINDS:
        model=network_fixture(network)
        batch=_input_batch(network)
        for independent_phi in (False,True):
            arch=network_architecture(network,independent_phi=independent_phi)
            compilation=compile_neural_chain(arch)
            emitted=compilation.emission
            probe=[]
            if run_host:
                for input_sample in batch:
                    actual=verify_neural_chain_gcc(
                        model,emitted,input_sample,
                        atol=HOST_ATOL,rtol=HOST_RTOL)
                    if actual["status"]!=(
                        "R2Z_ALL_ACTUAL_GCC_NN_STAGE_VALUES_QUALIFIED_E1"):
                        raise ValueError(
                            "R2z generated CNN/Dense source failed independent GCC stage oracle")
                    probe.append({
                        "input_sha256":actual["input_sha256"],
                        "per_stage_numerical_qualified":actual[
                            "all_stages_accepted"],
                        "final_output_qualified":actual[
                            "final_output_accepted"],
                        "intermediate_stage_count":actual["stage_count"],
                        "checked_stage_elements":actual[
                            "checked_stage_elements"],
                        "maximum_stage_absolute_error":actual[
                            "maximum_stage_absolute_error"],
                        "final_max_absolute_error":actual[
                            "final_max_absolute_error"],
                        "source_cpp_sha256":actual["source_cpp_sha256"],
                    })
            tb=make_testbench(
                model,batch,emitted.manifest["precision_bits"],
                emitted.manifest["integer_bits"],tolerance=CSIM_ATOL)
            tcl=make_vitis_tcl(
                part=PART,clock_ns=CLOCK_NS,
                clock_uncertainty_ns=UNCERTAINTY_NS)
            row={
                "case":f"case_{len(records):03d}",
                "network":network,"independent_phi":independent_phi,
                "operators":[op.kind for op in model.operators],
                "source_cpp_sha256":emitted.manifest["cpp_sha256"],
                "functional_sha256":emitted.manifest["functional_sha256"],
                "neural_contract_sha256":compilation.evidence["contract_sha256"],
                "original_input_batch_sha256":_sha(batch.tobytes()),
                "source_package_sha256":None,
                "prepared_vitis_tcl_sha256":_sha(tcl.encode()),
                "prepared_vitis_testbench_sha256":_sha(tb.encode()),
                "host_probes":probe,
                "host_all_stages_qualified":len(probe)==HOST_INPUTS_PER_CASE and
                    all(p["per_stage_numerical_qualified"] and
                        p["final_output_qualified"] for p in probe),
                "host_max_absolute_error":max(
                    (p["maximum_stage_absolute_error"] for p in probe),
                    default=None,
                ),
            }
            records.append((row,model,emitted,batch))
    if len(records)!=SOURCE_CASES:
        raise AssertionError("R2z original source population incomplete")
    if len({row["source_cpp_sha256"] for row,*_ in records}) != SOURCE_CASES:
        raise AssertionError("R2z factorial source/architecture differences missing")
    return records


def run(*,output:str|Path)->dict:
    destination=Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("R2z complete original neural study strictly create-only")
    revision=_rev()
    # Run ALL actual emitted C++ source checks and precompute every CSim
    # package BEFORE creating an immutable original study directory.
    all_records=_records(run_host=True)
    destination.mkdir(parents=True)
    final=[]
    for row,model,emitted,batch in all_records:
        folder=destination/row["case"]
        package=write_vitis_package(
            str(folder),model,emitted,batch,
            part=PART,clock_ns=CLOCK_NS,clock_uncertainty_ns=UNCERTAINTY_NS,
            tolerance=CSIM_ATOL,
        )
        if (package["cpp_sha256"]!=row["source_cpp_sha256"]
                or package["testbench_sha256"]!=row[
                    "prepared_vitis_testbench_sha256"]
                or _sha((folder/"run_hls.tcl").read_bytes())!=row[
                    "prepared_vitis_tcl_sha256"]):
            raise AssertionError("R2z prepared native package differs from source preflight")
        row["source_package_sha256"]=_sha(
            (folder/"manifest.json").read_bytes())
        final.append(row)
    summary={
        "schema":SCHEMA,
        "source_revision":revision,
        "network_topologies":list(KINDS),
        "source_cases":SOURCE_CASES,
        "independently_mapped_source_cases":len(KINDS),
        "actual_GCC_input_runs":sum(len(r["host_probes"]) for r in final),
        "actual_GCC_qualified_runs":sum(
            p["per_stage_numerical_qualified"] and p["final_output_qualified"]
            for r in final for p in r["host_probes"]),
        "complete_neural_networks_all_stages_qualified":sum(
            r["host_all_stages_qualified"] for r in final),
        "prepared_vitis_packages":len(final),
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "clock_ns":CLOCK_NS,"clock_uncertainty_ns":UNCERTAINTY_NS,
        "csim_absolute_tolerance":CSIM_ATOL,
        "host_absolute_tolerance":HOST_ATOL,
        "host_relative_tolerance":HOST_RTOL,
        "physical_BRAM":None,
        "limits":[
            "Deterministic nontrained small CNN/MLP samples, not broad model accuracy",
            "GCC host float E1, not ap_fixed bit-exact Vitis CSim or HLS resource proof",
            "HLS source packages prepared, native Vitis never invoked",
            "Full intermediate arrays; no dataflow streaming or source RAM reuse",
            "One sequential runtime input/output, no branching graphs",
        ],
    }
    (destination/"cases.json").write_text(
        json.dumps(final,sort_keys=True,indent=2)+"\n")
    (destination/"summary.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def audit(*,study:str|Path)->dict:
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R2z original neural study missing or symlink")
    files=[root/"summary.json",root/"cases.json"]
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("R2z original summary/case receipt absent")
    summary=json.loads(files[0].read_text())
    rows=json.loads(files[1].read_text())
    if (summary.get("schema")!=SCHEMA
            or summary.get("source_cases")!=SOURCE_CASES
            or summary.get("original_Vitis_calls")!=0
            or summary.get("original_Vivado_calls")!=0
            or summary.get("clock_ns")!=CLOCK_NS
            or summary.get("clock_uncertainty_ns")!=UNCERTAINTY_NS
            or summary.get("csim_absolute_tolerance")!=CSIM_ATOL
            or not isinstance(summary.get("source_revision"),str)
            or re.fullmatch(r"[a-f0-9]{40}",summary["source_revision"]) is None
            or not isinstance(rows,list) or len(rows)!=SOURCE_CASES):
        raise ValueError("R2z original source population/protocol mismatch")
    # SOURCE re-derivation on the current compiler checkout. The full
    # original source SHA remains pinned in summary even when only docs
    # advance main. This is NOT a new GCC run or numerical replay.
    expected=_records(run_host=False)
    for i,(saved,source) in enumerate(zip(rows,expected)):
        generated,model,emitted,batch=source
        folder=root/f"case_{i:03d}"
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("R2z original source package folder missing")
        for key in (
            "case","network","independent_phi","operators",
            "source_cpp_sha256","functional_sha256",
            "neural_contract_sha256","original_input_batch_sha256",
            "prepared_vitis_tcl_sha256","prepared_vitis_testbench_sha256",
        ):
            if saved.get(key)!=generated[key]:
                raise ValueError("R2z saved neural source differs from current compiler")
        for filename,expected_sha in (
            ("model.cpp",saved["source_cpp_sha256"]),
            ("tb.cpp",saved["prepared_vitis_testbench_sha256"]),
            ("run_hls.tcl",saved["prepared_vitis_tcl_sha256"]),
            ("manifest.json",saved["source_package_sha256"]),
        ):
            path=folder/filename
            if path.is_symlink() or not path.is_file() or _sha(
                    path.read_bytes())!=expected_sha:
                raise ValueError("R2z original C++/CSim/Tcl/package source SHA changed")
        if (folder/"model.cpp").read_text()!=emitted.cpp:
            raise ValueError("R2z original source not equal to canonical emitter")
        manifest=json.loads((folder/"manifest.json").read_text())
        if (manifest.get("cpp_sha256")!=saved["source_cpp_sha256"]
                or manifest.get("functional_sha256")!=saved["functional_sha256"]
                or manifest.get("part")!=PART
                or manifest.get("clock_ns")!=CLOCK_NS
                or manifest.get("explicit_clock_uncertainty_ns")!=UNCERTAINTY_NS
                or manifest.get("numerical_tolerance")!=CSIM_ATOL
                or manifest.get("testbench_sha256")!=
                    saved["prepared_vitis_testbench_sha256"]
                or manifest.get("csim_sample_count")!=HOST_INPUTS_PER_CASE
                or manifest.get("csim_outputs_per_sample")!=
                    int(np.prod(model.tensor(model.outputs[0]).shape))):
            raise ValueError("R2z original Vitis source, numeric or timing controls altered")
        if (len(saved.get("host_probes",[]))!=HOST_INPUTS_PER_CASE
                or saved.get("host_all_stages_qualified") is not True):
            raise ValueError("R2z original actual host stage checks incomplete")
        for j,receipt in enumerate(saved["host_probes"]):
            if (receipt.get("input_sha256")!=
                    _sha(batch[j].tobytes())
                    or receipt.get("source_cpp_sha256")!=saved[
                        "source_cpp_sha256"]
                    or receipt.get("per_stage_numerical_qualified") is not True
                    or receipt.get("final_output_qualified") is not True
                    or receipt.get("intermediate_stage_count") !=
                        len(model.operators)):
                raise ValueError("R2z original host witness altered")
    if (summary["actual_GCC_input_runs"]!=SOURCE_CASES*HOST_INPUTS_PER_CASE
            or summary["actual_GCC_qualified_runs"]!=
                SOURCE_CASES*HOST_INPUTS_PER_CASE
            or summary["complete_neural_networks_all_stages_qualified"]!=
                SOURCE_CASES
            or summary["prepared_vitis_packages"]!=SOURCE_CASES):
        raise ValueError("R2z original full original denominator inconsistent")
    return {
        "schema":AUDIT_SCHEMA,
        "source_revision":summary["source_revision"],
        "original_source_cases":SOURCE_CASES,
        "original_host_GCC_qualified_input_receipts":
            summary["actual_GCC_qualified_runs"],
        "original_source_packages_SHA_attested":SOURCE_CASES,
        "current_compiler_generated_cpp_identical":True,
        "new_original_GCC_calls":0,
        "new_original_Vitis_calls":0,
        "new_original_Vivado_calls":0,
        "note":"Read-only same-source SHA/rederivation and receipt audit; not independent GCC replay or HLS CSim",
    }


def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--study")
    args=parser.parse_args(argv)
    outcome=run(output=args.output) if args.output else audit(study=args.study)
    print(json.dumps(outcome,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
