"""Public CNN example gallery: real ONNX import and canonical HLS source."""
from __future__ import annotations

from hashlib import sha256
import json
import shutil

import pytest

from examples import make_neural_network_onnx as generator
from examples import compile_neural_network as gallery


@pytest.fixture(scope="module", autouse=True)
def need_onnx():
    pytest.importorskip("onnx")


@pytest.mark.parametrize("name", generator.NETWORKS)
def test_synthetic_networks_are_deterministic_and_checked_onnx(name, tmp_path):
    import onnx
    from fpgai_research.frontend.onnx import load_onnx

    original = generator.make_model(name)
    again = generator.make_model(name)
    assert original.SerializeToString() == again.SerializeToString()
    onnx.checker.check_model(original)
    destination = tmp_path / name / "example.onnx"
    generator.save_model(name, destination)
    assert destination.is_file()
    parsed = load_onnx(str(destination))
    assert parsed.tensor(parsed.inputs[0]).shape == generator.EXPECTED_INPUTS[name]
    assert parsed.tensor(parsed.outputs[0]).shape == generator.EXPECTED_OUTPUTS[name]
    assert any(op.kind == "Conv" for op in parsed.operators)
    assert any(op.kind in ("MatMul", "Gemm") for op in parsed.operators)
    assert "Flatten" in [op.kind for op in parsed.operators]
    assert len(parsed.operators) >= 6
    with pytest.raises(FileExistsError, match="overwrite"):
        generator.save_model(name, destination)


@pytest.mark.parametrize("name", generator.NETWORKS)
@pytest.mark.parametrize("mapping", ("scalar", "independent"))
def test_public_nn_example_compiles_every_stage_and_matches_canonical_emitter(
    name, mapping, tmp_path
):
    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.compiler.neural_chain import compile_neural_chain
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.model_ir.contracts import ArchitectureRequest

    model_path = generator.save_model(name, tmp_path / "source.onnx")
    model = load_onnx(str(model_path))
    arch = resolve(model, ArchitectureRequest(),
                   gallery.operator_mapping(model, mapping=mapping))
    canonical = compile_neural_chain(arch)
    output = tmp_path / "compiled"
    receipt = gallery.compile_example(model_path, output, mapping=mapping)
    assert receipt["mapping"] == mapping
    assert receipt["source_generated"] is True
    assert receipt["host_verification"] == "NOT_REQUESTED"
    assert receipt["native_vitis_or_vivado_calls"] == 0
    assert receipt["physical_timing_or_resource_claim"] is False
    assert receipt["all_stages_lowered"] is True
    assert receipt["operator_kinds"] == [
        op.kind for op in model.operators
    ]
    assert receipt["source_tensor_boundaries"] == len(model.operators)-1
    assert receipt["input_shape"] == list(generator.EXPECTED_INPUTS[name])
    assert receipt["output_shape"] == list(generator.EXPECTED_OUTPUTS[name])
    assert (output/"model.cpp").read_text() == canonical.emission.cpp
    assert receipt["canonical_cpp_sha256"] == sha256(
        (output/"model.cpp").read_bytes()).hexdigest()
    assert json.loads((output/"manifest.json").read_text())[
        "cpp_sha256"] == receipt["canonical_cpp_sha256"]
    assert (output/"network_contract.json").is_file()
    assert json.loads((output/"example_summary.json").read_text()) == receipt
    assert not (output/"fpgai_hls_project").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        gallery.compile_example(model_path, output, mapping=mapping)


@pytest.mark.skipif(shutil.which("g++") is None, reason="GNU C++ host tool not installed")
@pytest.mark.parametrize("name", generator.NETWORKS)
def test_public_example_can_verify_all_actual_generated_intermediate_tensors(
    name, tmp_path
):
    source = generator.save_model(name, tmp_path / "model.onnx")
    receipt = gallery.compile_example(
        source, tmp_path / "verified",
        mapping="independent", verify_host=True,
    )
    assert receipt["host_verification"] == "ALL_INTERMEDIATE_TENSORS_AGREED"
    assert receipt["host_gcc_executions"] == 1
    assert receipt["verified_intermediate_stages"] == len(
        receipt["operator_kinds"])
    assert receipt["native_vitis_or_vivado_calls"] == 0
    host = json.loads(
        (tmp_path/"verified"/"host_reference_verification.json").read_text())
    assert host["all_stages_accepted"] is True
    assert host["final_output_accepted"] is True


def test_example_invalid_network_and_mapping_fail_closed_without_outputs(
    tmp_path
):
    with pytest.raises(ValueError, match="Unknown network"):
        generator.make_model("proprietary")
    path = generator.save_model("cnn_pool_dense", tmp_path / "safe.onnx")
    from fpgai_research.frontend.onnx import load_onnx
    model = load_onnx(str(path))
    with pytest.raises(ValueError, match="scalar or independent"):
        gallery.operator_mapping(model, mapping="unverified")
    output = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match="scalar or independent"):
        gallery.compile_example(path, output, mapping="invented")
    assert not output.exists()


def test_public_example_is_not_a_vendor_or_private_fixture_import():
    import inspect
    source = inspect.getsource(generator) + inspect.getsource(gallery)
    assert "vitis_hls" not in source.lower()
    assert "vivado" not in generator.__doc__.lower()
    assert "cst_r2z_" not in source
    assert "rl_environment" not in source


def test_public_two_layer_mlp_onnx_uses_general_compiler_path(tmp_path):
    import onnx
    from examples import make_mlp_onnx
    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.compiler import compile_architecture
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.model_ir.contracts import ArchitectureRequest

    original=make_mlp_onnx.make_model()
    assert original.SerializeToString()==make_mlp_onnx.make_model().SerializeToString()
    onnx.checker.check_model(original)
    path=make_mlp_onnx.save_model(tmp_path/"tiny_mlp.onnx")
    model=load_onnx(str(path))
    assert model.tensor(model.inputs[0]).shape==(1,4)
    assert model.tensor(model.outputs[0]).shape==(1,3)
    assert [operator.kind for operator in model.operators]==[
        "Gemm","Relu","Gemm","Softmax"]
    compiled=compile_architecture(resolve(model,ArchitectureRequest()))
    assert [op["kind"] for op in compiled.emission.manifest["operators"]]==[
        "Gemm","Relu","Gemm","Softmax"]
    assert "fpgai_model" in compiled.emission.cpp
    assert compiled.emission.manifest["cpp_sha256"]==sha256(
        compiled.emission.cpp.encode()).hexdigest()
    with pytest.raises(FileExistsError,match="overwrite"):
        make_mlp_onnx.save_model(path)
