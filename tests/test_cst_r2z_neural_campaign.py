"""R2z full mixed CNN+Dense original source campaign and source SHA audit.

Unit tests stage real GCC E1 sources but NO Vitis/Vivado and NO
filesystem mutation of the frozen R2i-R2y native evidence cohorts.
"""
from __future__ import annotations

import json
import shutil

import pytest

from fpgai_research.experiments import cst_r2z_conv_dense_network_campaign as study


def test_exact_prespecified_original_network_population_and_source_sampling():
    assert study.SOURCE_CASES==6
    assert study.HOST_INPUTS_PER_CASE==2
    assert study.CLOCK_NS==10.0
    assert study.UNCERTAINTY_NS==1.0
    assert study.CSIM_ATOL==0.01
    assert [x["source_cpp_sha256"] for x,*_ in study._records(
        run_host=False)] == [
        x["source_cpp_sha256"] for x,*_ in study._records(run_host=False)
    ]
    rows=[r for r,*_ in study._records(run_host=False)]
    assert len(rows)==6
    assert len(set(r["source_cpp_sha256"] for r in rows))==6
    assert [r["independent_phi"] for r in rows]==[False,True]*3
    assert all("Conv" in r["operators"] and
               any(x in ("Gemm","MatMul") for x in r["operators"])
               for r in rows)
    assert all(r["host_probes"]==[] for r in rows)


@pytest.fixture(scope="module")
def original_package(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("R2z original C++ source campaign needs host GNU C++")
    folder=tmp_path_factory.mktemp("r2z_nn_source")/"original"
    # The full source campaign is create-only in production; tests
    # should not fail simply because a developer has uncommitted docs.
    original=study._rev
    study._rev=lambda: "a"*40
    try:
        recorded=study.run(output=folder)
    finally:
        study._rev=original
    return folder,recorded


def test_all_actual_generated_source_GCC_receipts_and_6_vitis_packages(
        original_package):
    folder,result=original_package
    assert result["source_cases"]==6
    assert result["actual_GCC_input_runs"]==12
    assert result["actual_GCC_qualified_runs"]==12
    assert result["complete_neural_networks_all_stages_qualified"]==6
    assert result["prepared_vitis_packages"]==6
    assert result["original_Vitis_calls"]==0
    assert result["original_Vivado_calls"]==0
    rows=json.loads((folder/"cases.json").read_text())
    assert len(rows)==6
    assert all(len(r["host_probes"])==2 for r in rows)
    assert all(r["host_all_stages_qualified"] for r in rows)
    assert all((folder/r["case"]/"model.cpp").exists() for r in rows)
    assert all((folder/r["case"]/"run_hls.tcl").exists() for r in rows)
    assert not any((folder/r["case"]/"fpgai_hls_project").exists()
                   for r in rows)
    before={str(p.relative_to(folder)):p.read_bytes()
            for p in folder.rglob("*") if p.is_file()}
    audited=study.audit(study=folder)
    assert audited["original_source_cases"]==6
    assert audited["original_host_GCC_qualified_input_receipts"]==12
    assert audited["original_source_packages_SHA_attested"]==6
    assert audited["new_original_GCC_calls"]==0
    assert audited["new_original_Vitis_calls"]==0
    assert before=={str(p.relative_to(folder)):p.read_bytes()
                    for p in folder.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError,match="create-only"):
        study.run(output=folder)


def test_tampered_hls_model_cpp_and_testbench_rejected(original_package,tmp_path):
    root,_=original_package
    folder=tmp_path/"tampered"
    shutil.copytree(root,folder)
    original=(folder/"case_000"/"model.cpp").read_bytes()
    (folder/"case_000"/"model.cpp").write_bytes(original+b"\n// tamper\n")
    with pytest.raises(ValueError,match="source SHA"):
        study.audit(study=folder)
    (folder/"case_000"/"model.cpp").write_bytes(original)
    p=folder/"case_005"/"tb.cpp"
    p.write_bytes(p.read_bytes()+b"\n// changed CSim targets\n")
    with pytest.raises(ValueError,match="source SHA"):
        study.audit(study=folder)


def test_tampered_host_receipt_rejected(original_package,tmp_path):
    root,_=original_package
    folder=tmp_path/"fake"
    shutil.copytree(root,folder)
    path=folder/"cases.json"
    rows=json.loads(path.read_text())
    rows[3]["host_probes"][0]["final_output_qualified"]=False
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="host witness altered"):
        study.audit(study=folder)
