"""R2z actual mixed Conv2D -> Flatten -> multi-Dense source coverage.

Real GCC/NumPy tests check EVERY generated intermediate tensor,
not only final classification. Existing emit_hls and canonical
compile_architecture are reused byte-for-byte. Never run Vitis/Vivado.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import re
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve, ArchitectureError
from fpgai_research.codegen.hls.emitter import Emission, emit_hls
from fpgai_research.compiler.neural_chain import compile_neural_chain
from fpgai_research.experiments.cst_r2z_neural_network_fixtures import (
    KINDS, network_fixture, network_architecture,
)
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.neural_chain import (
    resolve_neural_chain, neural_chain_receipt,
)
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.neural_chain_host import verify_neural_chain_gcc


@pytest.mark.parametrize("family",KINDS)
@pytest.mark.parametrize("independent_phi",(False,True))
def test_cnn_dense_real_compiler_reuses_canonical_conv_and_matmul_emitters(
        family,independent_phi):
    model=network_fixture(family)
    arch=network_architecture(family,independent_phi=independent_phi)
    result=compile_neural_chain(arch)
    assert result.ir.architectural==arch
    assert result.ir.has_conv and result.ir.has_dense
    assert result.ir.transition_operator=="flatten"
    assert result.ir.total_declared_intermediate_elements>0
    assert len(result.ir.boundaries)==len(model.operators)-1
    assert result.evidence["operators_preserved"]==len(model.operators)
    assert result.evidence["native_vitis_calls"]==0
    assert result.evidence["physical_BRAM"] is None
    assert result.evidence["actual_GCC_execution"] is False
    assert result.evidence["per_operator_lowering_preserved"] is True
    assert result.evidence["original_existing_conv_and_dense_emitters_reused"] is True
    assert result.emission==emit_hls(lower(arch))
    assert result.evidence["canonical_hls_source_sha256"] == \
        sha256(result.emission.cpp.encode()).hexdigest()
    actual=[(op["operator"],op["kind"])
            for op in result.emission.manifest["operators"]]
    assert actual==[(op.name,op.kind) for op in model.operators]
    assert "for(int yh=0;" in result.emission.cpp
    assert "#pragma HLS ARRAY_PARTITION variable=partial complete" in (
        result.emission.cpp)
    assert "using data_t = ap_fixed<16, 6, AP_RND, AP_SAT>;" in result.emission.cpp
    assert result.emission.manifest["loop_codegen_version"]=="static-conv-trip-v3"
    certificate=neural_chain_receipt(result.ir)
    assert certificate["contract_sha256"]==result.evidence["contract_sha256"]
    assert [c["tensor"] for c in certificate["boundaries"]]==[
        op.outputs[0] for op in model.operators[:-1]]


@pytest.mark.skipif(shutil.which("g++") is None,reason="Real GNU GCC is required")
@pytest.mark.parametrize("family",KINDS)
@pytest.mark.parametrize("independent_phi",(False,True))
def test_real_gcc_checks_every_conv_pool_flatten_dense_and_activation_tensor(
        family,independent_phi):
    model=network_fixture(family)
    arch=network_architecture(family,independent_phi=independent_phi)
    emission=compile_neural_chain(arch).emission
    rng=np.random.default_rng(1000+len(model.operators))
    x=rng.uniform(-.45,.45,size=model.tensor("x").shape).astype(np.float32)
    report=verify_neural_chain_gcc(model,emission,x,atol=1e-4,rtol=1e-3)
    assert report["status"]=="R2Z_ALL_ACTUAL_GCC_NN_STAGE_VALUES_QUALIFIED_E1",report
    assert report["all_stages_accepted"] is True
    assert report["final_output_accepted"] is True
    assert report["stage_count"]==len(model.operators)
    assert report["checked_stage_elements"]==sum(
        np.prod(model.tensor(op.outputs[0]).shape) for op in model.operators
    )
    assert report["actual_generated_GCC_executions"]==1
    assert report["new_original_Vitis_calls"]==0
    assert report["new_original_Vivado_calls"]==0
    assert [c["kind"] for c in report["stages"]]==[
        op.kind for op in model.operators]
    # Independent final-output-only oracle is an additional control:
    host=validate_host(model,emission,x,rtol=1e-3,atol=1e-4)
    assert host["accepted"] is True,host


@pytest.mark.skipif(shutil.which("g++") is None,reason="Real GNU GCC is required")
def test_wrong_real_conv_weight_changes_actual_stage_values_and_is_detected():
    model=network_fixture("cnn_pool_dense")
    arch=network_architecture("cnn_pool_dense")
    source=compile_neural_chain(arch).emission
    cpp,count=re.subn(
        r"(static const data_t tensor_\d+\[\d+\] = \{)data_t\([^)]+\)",
        lambda match:match.group(1)+"data_t(2.5)",
        source.cpp,count=1
    )
    assert count==1 and cpp!=source.cpp
    changed=Emission(cpp,{**source.manifest,
        "cpp_sha256":sha256(cpp.encode()).hexdigest()})
    x=np.linspace(-.4,.4,np.prod(model.tensor("x").shape),
                  dtype=np.float32).reshape(model.tensor("x").shape)
    rejected=verify_neural_chain_gcc(
        model,changed,x,rtol=1e-3,atol=1e-4)
    assert rejected["status"]=="R2Z_ACTUAL_GENERATED_NN_SOURCE_NUMERIC_REJECTED_E1"
    assert rejected["all_stages_accepted"] is False
    assert rejected["stages"][0]["accepted"] is False
    assert rejected["maximum_stage_absolute_error"]>1e-3


def test_model_input_and_output_geometry_and_missing_flatten_fail_closed():
    graph=network_fixture("cnn_pool_dense")
    arch=network_architecture("cnn_pool_dense")
    assert len(arch.points)==len(graph.operators)
    with pytest.raises(ValueError,match="bounded single-input/output"):
        resolve_neural_chain(arch,max_operators=3)
    with pytest.raises(ValueError,match="finite positive"):
        resolve_neural_chain(arch,max_constant_elements=0)
    with pytest.raises(ValueError,match="budget exceeded"):
        resolve_neural_chain(arch,max_model_elements=4)
    with pytest.raises(ValueError,match="budget exceeded"):
        resolve_neural_chain(arch,max_constant_elements=4)
    no_flat=FunctionalModel(
        tensors=graph.tensors,
        operators=tuple(op for op in graph.operators
                        if op.kind!="Flatten"),
        inputs=graph.inputs,outputs=graph.outputs,
    )
    with pytest.raises(ValueError,match="Missing dependency|non-topological"):
        resolve(no_flat,ArchitectureRequest())


def test_rejects_fanout_and_unsupported_nn_without_implicit_lowering():
    graph=network_fixture("cnn_pool_dense")
    ops=list(graph.operators)
    # Legal DAG where both later operators read c0, not a single chain.
    ops[2]=replace(ops[2],inputs=("c0",))
    altered=replace(graph,operators=tuple(ops))
    arch=resolve(altered,ArchitectureRequest())
    with pytest.raises(ValueError,match="sequential activation chain"):
        compile_neural_chain(arch)
    # Although the general emitter supports Add, this strictly bounded
    # sequential CNN/Dense compiler does not claim arbitrary DAG mixing.
    extra_tensor=Tensor("other",graph.tensor("r0").shape)
    add_graph=replace(graph,tensors=graph.tensors+(extra_tensor,))
    with pytest.raises(ValueError,match="no input, initializer or producer"):
        add_graph.validate()


def test_mixed_numeric_implementation_and_nonconstant_weights_refused():
    graph=network_fixture("cnn_pool_dense")
    base=ArchitectureRequest()
    arch=resolve(graph,base,{
        "dense0":replace(base,precision_bits=12,integer_bits=4)
    })
    with pytest.raises(ValueError,match="mixed fixed-point"):
        compile_neural_chain(arch)
    conv=graph.operators[0]
    illegal=replace(conv,attributes=(("group",2),))
    invalid=replace(graph,operators=(illegal,)+graph.operators[1:])
    with pytest.raises(ArchitectureError,match="Grouped/dilated"):
        resolve(invalid,base)
    weight=graph.tensor("w0")
    tensors=tuple(replace(t,data=None) if t.name=="w0" else t
                  for t in graph.tensors)
    with pytest.raises(ValueError,match="Missing dependency|non-topological"):
        resolve(replace(graph,tensors=tensors),base)


def test_manifest_source_sha_provenance_mismatch_is_rejected_before_GCC():
    model=network_fixture("cnn_strided_dense")
    arch=network_architecture("cnn_strided_dense")
    source=compile_neural_chain(arch).emission
    mismatched=Emission(source.cpp,{**source.manifest,
                                   "cpp_sha256":"0"*64})
    x=np.zeros(model.tensor("x").shape,dtype=np.float32)
    with pytest.raises(ValueError,match="source identity"):
        verify_neural_chain_gcc(model,mismatched,x)


def test_existing_on_device_vitis_package_writer_accepts_real_cnn_dense_source(
        tmp_path):
    from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
    model=network_fixture("cnn_strided_dense")
    arch=network_architecture("cnn_strided_dense")
    emitted=compile_neural_chain(arch).emission
    x=np.zeros(model.tensor("x").shape,dtype=np.float32)
    folder=tmp_path/"r2z_hls_package"
    package=write_vitis_package(
        str(folder),model,emitted,x,
        clock_ns=10.0,clock_uncertainty_ns=1.0,tolerance=0.01,
    )
    assert package["cpp_sha256"]==emitted.manifest["cpp_sha256"]
    assert package["csim_sample_count"]==1
    assert package["csim_outputs_per_sample"]==6
    assert package["explicit_clock_uncertainty_ns"]==1.0
    assert "csim_design" in (folder/"run_hls.tcl").read_text()
    assert (folder/"model.cpp").read_text()==emitted.cpp
    assert (folder/"tb.cpp").is_file()
    assert not (folder/"fpgai_hls_project").exists()
    with pytest.raises(FileExistsError,match="overwrite"):
        write_vitis_package(str(folder),model,emitted,x)


def test_unknown_presets_and_invalid_runtime_input_are_rejected():
    with pytest.raises(ValueError,match="three prespecified"):
        network_fixture("unknown")
    model=network_fixture("cnn_pool_dense")
    arch=network_architecture("cnn_pool_dense")
    compiled=compile_neural_chain(arch)
    with pytest.raises(ValueError,match="shape"):
        verify_neural_chain_gcc(
            model,compiled.emission,
            np.zeros((1,1,3,3),dtype=np.float32))


@pytest.mark.skipif(shutil.which("g++") is None,reason="Real GNU GCC is required")
def test_real_onnx_conv_relu_flatten_gemm_network_imports_and_executes(tmp_path):
    onnx=pytest.importorskip("onnx")
    from onnx import helper, TensorProto, numpy_helper
    from fpgai_research.frontend.onnx import load_onnx

    wc=np.full((2,1,3,3),0.03125,dtype=np.float32)
    wd=np.full((8,3),0.025,dtype=np.float32)
    bias=np.asarray((0.01,-0.03,0.02),dtype=np.float32)
    graph=helper.make_graph(
        [
            helper.make_node("Conv",["x","wc"],["conv"],name="conv0"),
            helper.make_node("Relu",["conv"],["relu"],name="relu0"),
            helper.make_node("Flatten",["relu"],["flat"],name="flatten",axis=1),
            helper.make_node("Gemm",["flat","wd","bias"],["out"],name="dense0"),
        ],
        "r2z_actual_onnx_conv_dense",
        [helper.make_tensor_value_info(
            "x",TensorProto.FLOAT,[1,1,4,4])],
        [helper.make_tensor_value_info(
            "out",TensorProto.FLOAT,[1,3])],
        [
            numpy_helper.from_array(wc,name="wc"),
            numpy_helper.from_array(wd,name="wd"),
            numpy_helper.from_array(bias,name="bias"),
        ],
    )
    path=tmp_path/"tiny_cnn_dense.onnx"
    onnx.save(helper.make_model(
        graph,opset_imports=[helper.make_opsetid("",13)]),str(path))
    model=load_onnx(str(path))
    assert [op.kind for op in model.operators]==[
        "Conv","Relu","Flatten","Gemm"]
    compiled=compile_neural_chain(resolve(model,ArchitectureRequest()))
    from fpgai_research.compiler import compile_onnx_neural_chain
    from_onnx=compile_onnx_neural_chain(str(path))
    assert from_onnx.emission.cpp == compiled.emission.cpp
    assert from_onnx.evidence["functional_sha256"]==compiled.evidence["functional_sha256"]
    x=np.linspace(-.3,.3,16,dtype=np.float32).reshape(1,1,4,4)
    actual=verify_neural_chain_gcc(model,compiled.emission,x)
    assert actual["status"]=="R2Z_ALL_ACTUAL_GCC_NN_STAGE_VALUES_QUALIFIED_E1"
    assert actual["stage_count"]==4
    assert actual["actual_generated_GCC_executions"]==1
    assert actual["new_original_Vitis_calls"]==0
