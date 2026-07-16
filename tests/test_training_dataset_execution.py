from pathlib import Path

import numpy as np

from fpgai.engine.compiler import Compiler
from fpgai.backends.hls.testbench_train import emit_tb_train_cpp


class _Tensor:
    def __init__(self, shape):
        self.shape = shape


class _Graph:
    outputs = ["output"]

    def get_tensor(self, name):
        assert name == "output"
        return _Tensor((1, 10))


def test_training_dataset_labels_are_lowered_to_one_hot_targets(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "validation" / "dataset"
    dataset_dir.mkdir(parents=True)
    labels_path = dataset_dir / "labels.npy"
    np.save(labels_path, np.asarray([2, 9, 0], dtype=np.int64))

    compiler = Compiler.__new__(Compiler)
    target_path = compiler._emit_training_dataset_target(
        tmp_path,
        _Graph(),
        {},
        {
            "sample_count": 3,
            "labels_path": str(labels_path),
            "targets_path": None,
        },
    )

    targets = np.fromfile(target_path, dtype=np.float32).reshape(3, 10)
    assert targets.shape == (3, 10)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert targets[0, 2] == 1.0
    assert targets[1, 9] == 1.0
    assert targets[2, 0] == 1.0


def test_training_testbench_reports_dataset_record_counts(tmp_path: Path) -> None:
    emit_tb_train_cpp(
        tmp_path,
        graph=_Graph(),
        top_name="deeplearn",
        in_words=784,
        out_words=10,
        weights_mode="embedded",
        weight_words=10,
        preload_weights=[],
        training_cfg={
            "execution": {
                "train_steps": 1,
                "batch_size": 10,
                "batch_mode": "accumulated",
            },
            "optimizer": {"type": "sgd", "learning_rate": 0.01},
        },
        raw_cfg={},
    )
    text = (tmp_path / "tb.cpp").read_text()
    assert r'\"dataset_input_records\"' in text
    assert r'\"dataset_target_records\"' in text
    assert r'\"dataset_records_consumed\"' in text
    assert "input_words_per_record = 784" in text
    assert "target_words_per_record = 10" in text


def test_dataset_artifact_contract_reports_input_words_per_sample(tmp_path: Path) -> None:
    from fpgai.validation.dataset import emit_dataset_artifacts

    dataset_path = tmp_path / "samples.npz"
    np.savez(
        dataset_path,
        inputs=np.zeros((3, 784), dtype=np.float32),
        labels=np.asarray([0, 1, 2], dtype=np.int64),
    )
    result = emit_dataset_artifacts(
        tmp_path / "build",
        raw_config={
            "validation": {
                "dataset": {
                    "source": "npz",
                    "path": str(dataset_path),
                    "inputs_key": "inputs",
                    "labels_key": "labels",
                }
            }
        },
    )
    assert result["status"] == "available"
    assert result["input_shape"] == [784]
    assert result["input_words_per_sample"] == 784


def test_dataset_training_reference_reports_full_batch_loss_and_update(tmp_path: Path) -> None:
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference

    graph = Graph("tiny_dataset_training")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 2))
    graph.add_tensor("dense_out", (1, 2))
    graph.add_tensor("output", (1, 2))
    graph.constants["W"] = np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    graph.constants["B"] = np.zeros((2,), dtype=np.float32)
    graph.add_op(
        "Dense",
        ["input", "W", "B"],
        ["dense_out"],
        name="dense",
        attrs={"in_features": 2, "out_features": 2},
    )
    graph.add_op("Softmax", ["dense_out"], ["output"], name="softmax")

    result = run_training_dataset_reference(
        graph=graph,
        raw_cfg={
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.1},
                "loss": {"type": "cross_entropy"},
                "execution": {
                    "train_steps": 1,
                    "batch_size": 2,
                    "batch_mode": "accumulated",
                },
            }
        },
        out_dir=tmp_path,
        inputs=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        targets=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    payload = __import__("json").loads(result.summary_json.read_text(encoding="utf-8"))
    assert payload["reference_scope"] == "full_dataset_accumulated_update"
    assert payload["sample_count"] == 2
    assert payload["optimizer_updates"] == 1
    assert np.isfinite(payload["initial_dataset_loss"])
    assert np.isfinite(payload["final_dataset_loss"])
    assert payload["gradient_l2_norm"] > 0.0
    assert payload["weight_update_l2_norm"] > 0.0
    assert result.grads_flat_path.exists()
    assert result.weights_after_flat_path.exists()


def test_dataset_training_reference_supports_dense_parameters_stored_in_attrs(tmp_path: Path) -> None:
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference

    graph = Graph("tiny_attr_dataset_training")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 2))
    graph.add_tensor("dense_out", (1, 2))
    graph.add_tensor("output", (1, 2))
    graph.add_op(
        "Dense",
        ["input"],
        ["dense_out"],
        name="dense0",
        attrs={
            "in_features": 2,
            "out_features": 2,
            "weights": np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32),
            "bias": np.zeros((2,), dtype=np.float32),
        },
    )
    graph.add_op("Softmax", ["dense_out"], ["output"], name="softmax")

    result = run_training_dataset_reference(
        graph=graph,
        raw_cfg={
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.1},
                "loss": {"type": "cross_entropy"},
                "execution": {
                    "train_steps": 1,
                    "batch_size": 2,
                    "batch_mode": "accumulated",
                },
            }
        },
        out_dir=tmp_path,
        inputs=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        targets=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    payload = __import__("json").loads(result.summary_json.read_text(encoding="utf-8"))
    assert payload["status"] == "available"
    assert payload["sample_count"] == 2
    assert payload["optimizer_updates"] == 1
    assert np.isfinite(payload["final_dataset_loss"])


def test_dataset_training_comparison_requires_real_complete_metrics(tmp_path: Path) -> None:
    import json
    from types import SimpleNamespace
    from fpgai.benchmark.training_compare import build_dataset_training_comparison

    raw_results = {
        "grads": {
            "mae": 1e-5, "max_abs": 1e-4, "cosine": 0.9999,
            "l2": 1e-4, "relative_l2": 1e-3, "ref_norm": 0.1,
            "got_norm": 0.1, "max_ref_abs": 0.01, "max_got_abs": 0.01,
            "low_energy_ref": False, "low_energy_got": False,
            "cosine_reliable": True, "count": 4, "ref_count": 4, "got_count": 4,
        },
        "weights_before": {
            "mae": 1e-6, "max_abs": 1e-5, "cosine": 1.0,
            "l2": 1e-5, "relative_l2": 1e-5, "ref_norm": 1.0,
            "got_norm": 1.0, "max_ref_abs": 0.5, "max_got_abs": 0.5,
            "low_energy_ref": False, "low_energy_got": False,
            "cosine_reliable": True, "count": 4, "ref_count": 4, "got_count": 4,
        },
        "weight_delta": {
            "mae": 1e-6, "max_abs": 1e-5, "cosine": 0.9999,
            "l2": 1e-5, "relative_l2": 1e-3, "ref_norm": 0.01,
            "got_norm": 0.01, "max_ref_abs": 0.001, "max_got_abs": 0.001,
            "low_energy_ref": False, "low_energy_got": False,
            "cosine_reliable": True, "count": 4, "ref_count": 4, "got_count": 4,
        },
        "weights_after": {
            "mae": 1e-6, "max_abs": 1e-5, "cosine": 1.0,
            "l2": 1e-5, "relative_l2": 1e-5, "ref_norm": 1.0,
            "got_norm": 1.0, "max_ref_abs": 0.5, "max_got_abs": 0.5,
            "low_energy_ref": False, "low_energy_got": False,
            "cosine_reliable": True, "count": 4, "ref_count": 4, "got_count": 4,
        },
    }
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps(raw_results), encoding="utf-8")
    summary_txt = tmp_path / "summary.txt"
    summary_txt.write_text("ok", encoding="utf-8")
    result = SimpleNamespace(results_json=results_json, summary_txt=summary_txt)

    payload = build_dataset_training_comparison(
        training_compare_result=result,
        execution_payload={
            "sample_count_requested": 10,
            "sample_count_executed": 10,
            "optimizer_update_calls": 1,
        },
        reference_payload={"sample_count": 10, "optimizer_updates": 1},
    )
    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["gradient_comparison"]["passed"] is True
    assert payload["initial_weight_comparison"]["passed"] is True
    assert payload["weight_delta_comparison"]["passed"] is True
    assert payload["final_weight_comparison"]["passed"] is True
    assert payload["execution_comparison"]["passed"] is True


def test_dataset_training_comparison_is_pending_without_artifacts() -> None:
    from fpgai.benchmark.training_compare import build_dataset_training_comparison

    payload = build_dataset_training_comparison(
        training_compare_result=None,
        execution_payload=None,
        reference_payload=None,
    )
    assert payload["status"] == "pending_comparison"
    assert payload["passed"] is False


def test_dataset_training_reference_emits_hardware_domain_artifacts(tmp_path: Path) -> None:
    import json
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference

    graph = Graph("tiny_hw_domain_training")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 2))
    graph.add_tensor("dense_out", (1, 2))
    graph.add_tensor("output", (1, 2))
    graph.constants["W"] = np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    graph.constants["B"] = np.zeros((2,), dtype=np.float32)
    graph.add_op("Dense", ["input", "W", "B"], ["dense_out"], name="dense", attrs={"in_features": 2, "out_features": 2})
    graph.add_op("Softmax", ["dense_out"], ["output"], name="softmax")

    result = run_training_dataset_reference(
        graph=graph,
        raw_cfg={
            "numerics": {
                "defaults": {
                    "weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                    "bias": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                    "accum": {"type": "ap_fixed", "total_bits": 18, "int_bits": 6},
                },
                "training": {
                    "grad_weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                    "grad_bias": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                    "update_accum": {"type": "ap_fixed", "total_bits": 18, "int_bits": 6},
                },
            },
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.1},
                "loss": {"type": "cross_entropy"},
                "execution": {"train_steps": 1, "batch_size": 2, "batch_mode": "accumulated"},
            },
        },
        out_dir=tmp_path,
        inputs=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        targets=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
    hw = payload["hardware_domain_reference"]
    assert hw["status"] == "available"
    assert Path(hw["grads_ref_bin"]).exists()
    assert Path(hw["weights_before_ref_bin"]).exists()
    assert Path(hw["weights_after_ref_bin"]).exists()
    assert hw["gradient_reduction"] == "quantize_each_sample_accumulate_then_mean"
    assert hw["reference_method"] == "operation_level_fixed_point"
    assert hw["fallback_reason"] is None


def test_dataset_training_comparison_uses_hardware_domain_for_decision(tmp_path: Path) -> None:
    import json
    from types import SimpleNamespace
    from fpgai.benchmark.training_compare import build_dataset_training_comparison

    def metric():
        return {
            "mae": 0.0, "max_abs": 0.0, "cosine": 1.0,
            "l2": 0.0, "relative_l2": 0.0, "ref_norm": 1.0,
            "got_norm": 1.0, "max_ref_abs": 0.5, "max_got_abs": 0.5,
            "low_energy_ref": False, "low_energy_got": False,
            "cosine_reliable": True, "count": 4, "ref_count": 4, "got_count": 4,
        }
    hw_path = tmp_path / "hw.json"
    hw_path.write_text(json.dumps({"grads": metric(), "weights_before": metric(), "weight_delta": metric(), "weights_after": metric()}), encoding="utf-8")
    float_path = tmp_path / "float.json"
    bad = metric(); bad["cosine"] = 0.2; bad["relative_l2"] = 2.0
    float_path.write_text(json.dumps({"grads": bad, "weights_before": metric(), "weight_delta": bad, "weights_after": metric()}), encoding="utf-8")
    summary = tmp_path / "summary.txt"; summary.write_text("ok", encoding="utf-8")

    payload = build_dataset_training_comparison(
        training_compare_result=SimpleNamespace(results_json=hw_path, summary_txt=summary),
        float_training_compare_result=SimpleNamespace(results_json=float_path, summary_txt=summary),
        execution_payload={"sample_count_requested": 2, "sample_count_executed": 2, "optimizer_update_calls": 1},
        reference_payload={"sample_count": 2, "optimizer_updates": 1},
    )
    assert payload["status"] == "passed"
    assert payload["decision_reference_domain"] == "hardware_fixed_point"
    assert payload["float_reference_diagnostics"]["grads"]["cosine"] == 0.2


def test_training_semantic_trace_report_identifies_first_gradient_divergence(tmp_path: Path) -> None:
    from fpgai.benchmark.training_compare import build_training_semantic_trace_report

    ref_sum = tmp_path / "ref_sum.bin"
    hls_sum = tmp_path / "hls_sum.bin"
    ref_reduced = tmp_path / "ref_reduced.bin"
    hls_reduced = tmp_path / "hls_reduced.bin"
    before = tmp_path / "before.bin"
    after = tmp_path / "after.bin"

    np.asarray([1.0, 2.0], dtype=np.float32).tofile(ref_sum)
    np.asarray([1.0, 2.0], dtype=np.float32).tofile(hls_sum)
    np.asarray([0.5, 1.0], dtype=np.float32).tofile(ref_reduced)
    np.asarray([1.0, 0.0], dtype=np.float32).tofile(hls_reduced)
    np.asarray([0.2, 0.4], dtype=np.float32).tofile(before)
    np.asarray([0.1, 0.3], dtype=np.float32).tofile(after)

    payload = build_training_semantic_trace_report(
        hls_gradient_accumulated=hls_sum,
        hls_gradient_reduced=hls_reduced,
        ref_gradient_accumulated=ref_sum,
        ref_gradient_reduced=ref_reduced,
        hls_weights_before=before,
        hls_weights_after=after,
    )

    assert payload["status"] == "available"
    assert payload["first_divergence_stage"] == "gradient_reduced_export"
    assert payload["stages"]["gradient_accumulated_pre_reduce"]["relative_l2"] == 0.0
    assert payload["stages"]["weight_update_precast"]["status"] == "not_observable"


def test_hardware_domain_reference_emits_gradient_semantic_stages(tmp_path: Path) -> None:
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference

    graph = Graph("tiny_semantic_trace")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 2))
    graph.add_tensor("dense_out", (1, 2))
    graph.add_tensor("output", (1, 2))
    graph.add_op(
        "Dense", ["input"], ["dense_out"], name="dense0",
        attrs={
            "in_features": 2, "out_features": 2,
            "weights": np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32),
            "bias": np.zeros((2,), dtype=np.float32),
        },
    )
    graph.add_op("Softmax", ["dense_out"], ["output"], name="softmax")

    result = run_training_dataset_reference(
        graph=graph,
        raw_cfg={
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.1},
                "loss": {"type": "cross_entropy"},
                "execution": {"train_steps": 1, "batch_size": 2, "batch_mode": "accumulated"},
            }
        },
        out_dir=tmp_path,
        inputs=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        targets=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    payload = __import__("json").loads(result.summary_json.read_text(encoding="utf-8"))
    hardware = payload["hardware_domain_reference"]
    assert Path(hardware["gradient_accumulated_pre_reduce_ref_bin"]).exists()
    assert Path(hardware["gradient_reduced_ref_bin"]).exists()


def test_hardware_domain_reference_emits_per_sample_trace_and_layer_map(tmp_path: Path) -> None:
    import json
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference

    graph = Graph("tiny_per_sample_trace")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 2))
    graph.add_tensor("dense_out", (1, 2))
    graph.add_tensor("output", (1, 2))
    graph.add_op(
        "Dense", ["input"], ["dense_out"], name="dense0",
        attrs={
            "in_features": 2, "out_features": 2,
            "weights": np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32),
            "bias": np.zeros((2,), dtype=np.float32),
        },
    )
    graph.add_op("Softmax", ["dense_out"], ["output"], name="softmax")

    result = run_training_dataset_reference(
        graph=graph,
        raw_cfg={
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.1},
                "loss": {"type": "cross_entropy"},
                "execution": {"train_steps": 1, "batch_size": 2, "batch_mode": "accumulated"},
            }
        },
        out_dir=tmp_path,
        inputs=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        targets=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
    hardware = payload["hardware_domain_reference"]
    assert len(hardware["per_sample_gradient_ref_bins"]) == 2
    assert len(hardware["accumulator_after_ref_bins"]) == 2
    assert all(Path(value).exists() for value in hardware["per_sample_gradient_ref_bins"])
    layer_map = json.loads(Path(hardware["parameter_layer_map_json"]).read_text(encoding="utf-8"))
    assert layer_map["entries"][0]["layer"] == "dense0"
    assert layer_map["entries"][0]["role"] == "weight"


def test_per_sample_gradient_trace_reports_first_sample_and_layer(tmp_path: Path) -> None:
    import json
    from fpgai.benchmark.training_compare import build_training_per_sample_gradient_trace_report

    hls = tmp_path / "hls"
    ref = tmp_path / "ref"
    hls.mkdir(); ref.mkdir()
    np.asarray([1.0, 2.0, 0.0], dtype=np.float32).tofile(hls / "per_sample_gradient_0000.bin")
    np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tofile(ref / "per_sample_gradient_0000_ref.bin")
    np.asarray([1.0, 2.0, 0.0], dtype=np.float32).tofile(hls / "accumulator_after_0000.bin")
    np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tofile(ref / "accumulator_after_0000_ref.bin")
    layer_map = tmp_path / "layer_map.json"
    layer_map.write_text(json.dumps({"entries": [
        {"layer": "dense0", "role": "weight", "offset": 0, "count": 2},
        {"layer": "dense0", "role": "bias", "offset": 2, "count": 1},
    ]}), encoding="utf-8")

    payload = build_training_per_sample_gradient_trace_report(
        hls_trace_root=hls,
        ref_per_sample_paths=[ref / "per_sample_gradient_0000_ref.bin"],
        ref_accumulator_paths=[ref / "accumulator_after_0000_ref.bin"],
        parameter_layer_map_path=layer_map,
    )
    assert payload["status"] == "available"
    assert payload["first_divergent_sample"] == 0
    assert payload["first_divergent_parameter_index"] == 1
    assert payload["first_divergent_layer"] == "dense0"
    assert payload["first_divergent_role"] == "weight"


def test_gradient_layer_role_reports_isolate_bias_and_scale(tmp_path: Path) -> None:
    import json
    from fpgai.benchmark.training_compare import build_training_gradient_layer_role_reports

    hls = tmp_path / "hls"; ref = tmp_path / "ref"
    hls.mkdir(); ref.mkdir()
    # two weights then one bias; HLS bias is batch-scaled by 2
    np.asarray([1.0, 2.0, 0.4], dtype=np.float32).tofile(hls / "per_sample_gradient_0000.bin")
    np.asarray([1.0, 2.0, 0.2], dtype=np.float32).tofile(ref / "per_sample_gradient_0000_ref.bin")
    np.asarray([1.0, 2.0, 0.4], dtype=np.float32).tofile(hls / "accumulator_after_0000.bin")
    np.asarray([1.0, 2.0, 0.2], dtype=np.float32).tofile(ref / "accumulator_after_0000_ref.bin")
    layer_map = tmp_path / "layer_map.json"
    layer_map.write_text(json.dumps({"entries": [
        {"layer": "dense0", "role": "weight", "offset": 0, "count": 2},
        {"layer": "dense0", "role": "bias", "offset": 2, "count": 1},
    ]}), encoding="utf-8")

    by_layer, by_role = build_training_gradient_layer_role_reports(
        hls_trace_root=hls,
        ref_per_sample_paths=[ref / "per_sample_gradient_0000_ref.bin"],
        ref_accumulator_paths=[ref / "accumulator_after_0000_ref.bin"],
        parameter_layer_map_path=layer_map,
        batch_size=2,
    )
    assert by_layer["status"] == "available"
    assert by_layer["layers"]["dense0"]["weight"]["mae"] == 0.0
    assert by_layer["layers"]["dense0"]["bias"]["mae"] > 0.0
    assert by_role["roles"]["bias"]["all"]["got_norm"] > by_role["roles"]["bias"]["all"]["ref_norm"]
    assert by_role["bias_trace"]["samples"][0]["best_matching_scale"] == "batch_size"


def test_compile_training_bias_report_uses_resolved_raw_config() -> None:
    """Guard the full compiler integration path against out-of-scope config names."""
    import ast
    import inspect
    import textwrap

    from fpgai.engine.compiler import Compiler

    tree = ast.parse(textwrap.dedent(inspect.getsource(Compiler._compile_training)))
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    assert "raw_cfg" not in loaded_names
    assert "raw" in loaded_names


def test_hardware_domain_dense_reference_emulates_hls_softmax_and_casts() -> None:
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import (
        _run_hls_numeric_training_sample,
        _trainable_layout,
    )

    graph = Graph("tiny_hls_numeric_reference")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 2))
    graph.add_tensor("hidden", (1, 2))
    graph.add_tensor("relu", (1, 2))
    graph.add_tensor("logits", (1, 2))
    graph.add_tensor("output", (1, 2))
    graph.constants["W0"] = np.asarray([[0.25, -0.5], [0.125, 0.375]], dtype=np.float32)
    graph.constants["B0"] = np.asarray([0.01, -0.02], dtype=np.float32)
    graph.constants["W1"] = np.asarray([[0.5, -0.25], [-0.125, 0.75]], dtype=np.float32)
    graph.constants["B1"] = np.asarray([0.0, 0.0], dtype=np.float32)
    graph.add_op("Dense", ["input", "W0", "B0"], ["hidden"], name="dense0", attrs={"in_features": 2, "out_features": 2})
    graph.add_op("Relu", ["hidden"], ["relu"], name="relu0")
    graph.add_op("Dense", ["relu", "W1", "B1"], ["logits"], name="dense1", attrs={"in_features": 2, "out_features": 2})
    graph.add_op("Softmax", ["logits"], ["output"], name="softmax")

    raw = {
        "numerics": {
            "defaults": {
                "activation": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                "weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                "bias": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                "accum": {"type": "ap_fixed", "total_bits": 18, "int_bits": 6},
            },
            "training": {
                "grad_activation": {"type": "ap_fixed", "total_bits": 14, "int_bits": 5},
                "grad_weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                "grad_bias": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
                "update_accum": {"type": "ap_fixed", "total_bits": 18, "int_bits": 6},
            },
        },
        "training": {"loss": {"type": "cross_entropy"}},
    }
    layout = _trainable_layout(graph)
    gradient, loss = _run_hls_numeric_training_sample(
        graph=graph,
        raw_cfg=raw,
        x_input=np.asarray([0.75, -0.25], dtype=np.float32),
        target=np.asarray([1.0, 0.0], dtype=np.float32),
        layout=layout,
    )

    assert gradient.shape == (12,)
    assert np.all(np.isfinite(gradient))
    assert np.isfinite(loss)
    assert np.linalg.norm(gradient) > 0.0
    # Every exported parameter gradient uses ap_fixed<12,4>, so values must
    # lie exactly on the 2^-8 storage grid after operation-level emulation.
    scaled = gradient * np.float32(256.0)
    assert np.allclose(scaled, np.trunc(scaled), atol=0.0, rtol=0.0)


def test_ap_fixed_default_emulation_uses_floor_and_wrap() -> None:
    from fpgai.numerics.fixed_emulation import quantize_ap_fixed_array

    spec = {"type": "ap_fixed", "total_bits": 4, "int_bits": 2}
    values = quantize_ap_fixed_array(
        np.asarray([-0.1, 0.1, 2.25, -2.25], dtype=np.float32),
        spec,
    )

    # AP_TRN removes low bits, which floors negative two's-complement values.
    # AP_WRAP wraps overflow in the signed four-bit storage domain.
    assert np.array_equal(
        values,
        np.asarray([-0.25, 0.0, -1.75, 1.75], dtype=np.float32),
    )


def test_dataset_training_comparison_rejects_initial_parameter_mismatch(tmp_path: Path) -> None:
    import json
    from types import SimpleNamespace
    from fpgai.benchmark.training_compare import build_dataset_training_comparison

    def metric(*, cosine: float = 1.0, relative_l2: float = 0.0, mae: float = 0.0):
        return {
            "mae": mae, "max_abs": mae, "cosine": cosine,
            "l2": relative_l2, "relative_l2": relative_l2, "ref_norm": 1.0,
            "got_norm": 1.0, "max_ref_abs": 0.5, "max_got_abs": 0.5,
            "low_energy_ref": False, "low_energy_got": False,
            "cosine_reliable": True, "count": 4, "ref_count": 4, "got_count": 4,
        }

    results = {
        "grads": metric(),
        "weights_before": metric(cosine=0.8, relative_l2=0.2, mae=0.01),
        "weight_delta": metric(),
        "weights_after": metric(),
    }
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps(results), encoding="utf-8")
    summary_txt = tmp_path / "summary.txt"
    summary_txt.write_text("comparison", encoding="utf-8")

    payload = build_dataset_training_comparison(
        training_compare_result=SimpleNamespace(results_json=results_json, summary_txt=summary_txt),
        execution_payload={
            "sample_count_requested": 2,
            "sample_count_executed": 2,
            "optimizer_update_calls": 1,
        },
        reference_payload={"sample_count": 2, "optimizer_updates": 1},
    )

    assert payload["status"] == "failed_tolerance"
    assert payload["initial_weight_comparison"]["passed"] is False
    assert any(item["name"].startswith("initial_weights.") and not item["passed"] for item in payload["checks"])


def test_training_execution_schedule_separates_epochs_batches_and_record_visits() -> None:
    from fpgai.engine.training import resolve_training_execution_schedule

    schedule = resolve_training_execution_schedule(
        {
            "training": {
                "batch": {
                    "size": 2,
                    "epochs": 3,
                    "mode": "accumulated",
                    "shuffle": True,
                    "seed": 17,
                    "drop_last": False,
                }
            }
        },
        sample_count=5,
    )

    assert schedule.batch_size == 2
    assert schedule.epochs == 3
    assert schedule.batches_per_epoch == 3
    assert schedule.samples_per_epoch == 5
    assert schedule.total_optimizer_updates == 9
    assert schedule.total_forward_backward_calls == 15
    assert schedule.explicit_train_steps is None


def test_training_record_order_is_deterministic_and_epoch_specific() -> None:
    from fpgai.engine.training import training_record_order

    first = training_record_order(10, epoch_index=0, shuffle=True, seed=23)
    repeated = training_record_order(10, epoch_index=0, shuffle=True, seed=23)
    next_epoch = training_record_order(10, epoch_index=1, shuffle=True, seed=23)

    assert first == repeated
    assert sorted(first) == list(range(10))
    assert sorted(next_epoch) == list(range(10))
    assert first != next_epoch
    assert training_record_order(4, epoch_index=9, shuffle=False, seed=99) == [0, 1, 2, 3]


def test_training_testbench_emits_true_epoch_batch_schedule_and_checkpoints(tmp_path: Path) -> None:
    emit_tb_train_cpp(
        tmp_path,
        graph=_Graph(),
        top_name="deeplearn",
        in_words=4,
        out_words=10,
        weights_mode="embedded",
        weight_words=6,
        preload_weights=[0.0] * 6,
        training_cfg={
            "optimizer": {"type": "sgd", "learning_rate": 0.01},
            "loss": {"type": "cross_entropy"},
            "batch": {
                "size": 2,
                "epochs": 3,
                "mode": "accumulated",
                "shuffle": True,
                "seed": 7,
            },
        },
        raw_cfg={
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.01},
                "loss": {"type": "cross_entropy"},
                "batch": {
                    "size": 2,
                    "epochs": 3,
                    "mode": "accumulated",
                    "shuffle": True,
                    "seed": 7,
                },
            }
        },
        dataset_sample_count=5,
    )

    text = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert "const int batches_per_epoch" in text
    assert "configured_epochs = 3" in text
    assert "configured_epochs * batches_per_epoch" in text
    assert "make_epoch_order(" in text
    assert "std::fill(previous_accumulator.begin(), previous_accumulator.end(), 0.0f);" in text
    assert "training_epoch_curve.csv" in text
    assert "training_batch_curve.csv" in text
    assert "epoch_%04d_weights.bin" in text
    assert 'optimizer_location=hls_top_accumulated_optimizer\\n");' in text
    assert 'dataset_loss,accuracy,checkpoint\\n";' in text
    assert 'gradient_max_abs\\n";' in text
    assert r'\"epochs_completed\"' in text
    assert r'\"dataset_records_consumed\"' in text


def test_dataset_training_reference_runs_multiple_epochs_and_batches(tmp_path: Path) -> None:
    import csv
    import json
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference

    graph = Graph("tiny_multiepoch_dataset_training")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 2))
    graph.add_tensor("dense_out", (1, 2))
    graph.add_tensor("output", (1, 2))
    graph.constants["W"] = np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    graph.constants["B"] = np.zeros((2,), dtype=np.float32)
    graph.add_op(
        "Dense",
        ["input", "W", "B"],
        ["dense_out"],
        name="dense",
        attrs={"in_features": 2, "out_features": 2},
    )
    graph.add_op("Softmax", ["dense_out"], ["output"], name="softmax")

    result = run_training_dataset_reference(
        graph=graph,
        raw_cfg={
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.1},
                "loss": {"type": "cross_entropy"},
                "batch": {
                    "size": 1,
                    "epochs": 2,
                    "mode": "accumulated",
                    "shuffle": True,
                    "seed": 5,
                },
            }
        },
        out_dir=tmp_path,
        inputs=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        targets=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert payload["reference_scope"] == "deterministic_multi_epoch_accumulated_training"
    assert payload["optimizer_updates"] == 4
    assert payload["epochs_completed"] == 2
    assert payload["records_consumed"] == 4
    assert payload["execution_schedule"]["batches_per_epoch"] == 2
    assert payload["hardware_domain_reference"]["optimizer_updates"] == 4
    assert payload["hardware_domain_reference"]["epochs_completed"] == 2
    assert payload["hardware_domain_reference"]["records_consumed"] == 4

    with Path(payload["training_epoch_curve_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [0, 1, 2]
    assert [int(row["optimizer_updates"]) for row in rows] == [0, 2, 4]


def test_training_execution_report_publishes_curves_and_checkpoint_inventory(tmp_path: Path) -> None:
    import json

    hls_dir = tmp_path / "hls"
    artifact_dir = hls_dir / "project" / "sol1" / "csim" / "build"
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (artifact_dir / "training_multistep_summary.json").write_text(
        json.dumps({
            "dataset_input_records": 5,
            "dataset_target_records": 5,
            "dataset_records_consumed": 15,
            "optimizer_update_calls": 9,
            "epochs_completed": 3,
        }),
        encoding="utf-8",
    )
    (artifact_dir / "training_epoch_curve.csv").write_text(
        "epoch,optimizer_updates,records_consumed,dataset_loss,checkpoint\n",
        encoding="utf-8",
    )
    (artifact_dir / "training_batch_curve.csv").write_text(
        "epoch,batch,optimizer_update,actual_batch_size,records_consumed,gradient_l2_norm,gradient_max_abs\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "epoch_0001_weights.bin").write_bytes(b"1234")
    (checkpoint_dir / "epoch_0002_weights.bin").write_bytes(b"5678")
    (checkpoint_dir / "epoch_0003_weights.bin").write_bytes(b"abcd")

    compiler = Compiler.__new__(Compiler)
    report = compiler._emit_training_dataset_execution_report(
        out_dir=tmp_path / "out",
        hls_dir=hls_dir,
        training_dataset_artifacts={"status": "available", "sample_count": 5},
    )

    assert report is not None
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["dataset_sample_count"] == 5
    assert payload["unique_dataset_records"] == 5
    assert payload["record_visits_executed"] == 15
    assert payload["unique_records_executed"] == 5
    assert payload["forward_backward_calls"] == 0
    assert payload["optimizer_updates"] == 9
    assert payload["batches_completed"] == 9
    assert payload["checkpoint_count"] == 3
    assert len(payload["checkpoint_files"]) == 3
    assert Path(payload["training_epoch_curve_csv"]).exists()
    assert Path(payload["training_batch_curve_csv"]).exists()


def test_training_dataset_model_contract_rejects_input_word_mismatch(tmp_path: Path) -> None:
    import json
    from fpgai.ir.graph import Graph
    from fpgai.validation.dataset import emit_dataset_artifacts, emit_dataset_model_contract

    graph = Graph("shape_mismatch")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 8))
    graph.add_tensor("output", (1, 10))

    dataset_path = tmp_path / "mnist_like.npz"
    np.savez(
        dataset_path,
        inputs=np.ones((3, 784), dtype=np.float32),
        labels=np.asarray([0, 1, 2], dtype=np.int64),
    )
    artifacts = emit_dataset_artifacts(
        tmp_path / "build",
        raw_config={
            "validation": {
                "dataset": {
                    "source": "npz",
                    "path": str(dataset_path),
                    "inputs_key": "inputs",
                    "labels_key": "labels",
                }
            }
        },
    )
    contract = emit_dataset_model_contract(
        tmp_path / "build",
        graph=graph,
        dataset_artifacts=artifacts,
        require_supervision=True,
    )

    assert contract["status"] == "incompatible"
    assert contract["model"]["input_words"] == 8
    assert contract["dataset"]["input_words_per_sample"] == 784
    assert "784 words" in contract["reason"]
    assert "8 words" in contract["reason"]
    saved = json.loads(Path(contract["json_path"]).read_text(encoding="utf-8"))
    assert saved["status"] == "incompatible"


def test_training_dataset_model_contract_accepts_reshape_compatible_mnist_records(tmp_path: Path) -> None:
    from fpgai.ir.graph import Graph
    from fpgai.validation.dataset import emit_dataset_artifacts, emit_dataset_model_contract

    graph = Graph("mnist_compatible")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 1, 28, 28))
    graph.add_tensor("output", (1, 10))

    dataset_path = tmp_path / "balanced.npz"
    inputs = np.zeros((10, 784), dtype=np.float32)
    for index in range(10):
        inputs[index, index] = float(index + 1) / 10.0
    np.savez(
        dataset_path,
        inputs=inputs,
        labels=np.arange(10, dtype=np.int64),
    )
    artifacts = emit_dataset_artifacts(
        tmp_path / "build",
        raw_config={
            "validation": {
                "dataset": {
                    "source": "npz",
                    "path": str(dataset_path),
                }
            }
        },
    )
    contract = emit_dataset_model_contract(
        tmp_path / "build",
        graph=graph,
        dataset_artifacts=artifacts,
        require_supervision=True,
    )

    assert contract["status"] == "compatible"
    assert contract["compatibility_basis"] == "flattened_scalar_word_count"
    assert contract["dataset"]["labels"]["class_count"] == 10
    assert contract["claim_scope"] == "small_sample_learning_smoke_only"


def test_training_dataset_model_contract_marks_degenerate_dataset_as_mechanism_only(tmp_path: Path) -> None:
    from fpgai.ir.graph import Graph
    from fpgai.validation.dataset import emit_dataset_artifacts, emit_dataset_model_contract

    graph = Graph("degenerate")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 784))
    graph.add_tensor("output", (1, 10))

    dataset_path = tmp_path / "zero_single_class.npz"
    np.savez(
        dataset_path,
        inputs=np.zeros((10, 784), dtype=np.float32),
        labels=np.zeros((10,), dtype=np.int64),
    )
    artifacts = emit_dataset_artifacts(
        tmp_path / "build",
        raw_config={"validation": {"dataset": {"source": "npz", "path": str(dataset_path)}}},
    )
    contract = emit_dataset_model_contract(
        tmp_path / "build",
        graph=graph,
        dataset_artifacts=artifacts,
        require_supervision=True,
    )

    assert contract["status"] == "compatible"
    assert contract["claim_scope"] == "mechanism_smoke_only"
    assert "dataset inputs are all zero" in contract["warnings"]
    assert "classification labels contain only one class" in contract["warnings"]


def test_equal_update_budget_schedule_separates_updates_from_record_exposure() -> None:
    from fpgai.engine.training import resolve_training_execution_schedule

    base = {
        "training": {
            "batch": {
                "epochs": 3,
                "mode": "accumulated",
                "shuffle": True,
                "seed": 42,
                "drop_last": False,
                "max_updates": 6,
            }
        }
    }
    b2 = __import__("copy").deepcopy(base)
    b2["training"]["batch"]["size"] = 2
    b5 = __import__("copy").deepcopy(base)
    b5["training"]["batch"]["size"] = 5

    schedule_b2 = resolve_training_execution_schedule(b2, sample_count=10)
    schedule_b5 = resolve_training_execution_schedule(b5, sample_count=10)

    assert schedule_b2.total_optimizer_updates == 6
    assert schedule_b5.total_optimizer_updates == 6
    assert schedule_b2.total_forward_backward_calls == 12
    assert schedule_b5.total_forward_backward_calls == 30

def test_training_validation_split_contract_accepts_disjoint_ranges(tmp_path: Path) -> None:
    from fpgai.validation.dataset import (
        emit_dataset_artifacts,
        emit_training_validation_dataset_artifacts,
        emit_training_validation_split_contract,
    )

    dataset_path = tmp_path / "records.npz"
    inputs = np.arange(20 * 8, dtype=np.float32).reshape(20, 8)
    labels = np.arange(20, dtype=np.int64) % 2
    np.savez(dataset_path, inputs=inputs, labels=labels)
    raw = {
        "validation": {
            "dataset": {
                "source": "npz",
                "path": str(dataset_path),
                "sample_selection": {"mode": "first", "offset": 0, "count": 10},
            },
            "training_validation": {
                "dataset": {
                    "source": "npz",
                    "path": str(dataset_path),
                    "sample_selection": {"mode": "first", "offset": 10, "count": 5},
                }
            },
        }
    }
    train = emit_dataset_artifacts(tmp_path / "build", raw_config=raw)
    held_out = emit_training_validation_dataset_artifacts(tmp_path / "build", raw_config=raw)
    contract = emit_training_validation_split_contract(
        tmp_path / "build", training_artifacts=train, validation_artifacts=held_out
    )

    assert contract["status"] == "compatible"
    assert contract["same_source"] is True
    assert contract["overlap_count"] == 0
    assert contract["training"]["sample_count"] == 10
    assert contract["validation"]["sample_count"] == 5
    assert contract["claim_scope"] == "held_out_validation_mechanism"
    assert Path(contract["json_path"]).exists()
    assert Path(contract["markdown_path"]).exists()


def test_training_validation_split_contract_rejects_overlap(tmp_path: Path) -> None:
    from fpgai.validation.dataset import (
        emit_dataset_artifacts,
        emit_training_validation_dataset_artifacts,
        emit_training_validation_split_contract,
    )

    dataset_path = tmp_path / "records.npz"
    np.savez(
        dataset_path,
        inputs=np.ones((20, 8), dtype=np.float32),
        labels=np.arange(20, dtype=np.int64) % 2,
    )
    raw = {
        "validation": {
            "dataset": {
                "source": "npz",
                "path": str(dataset_path),
                "sample_selection": {"mode": "first", "offset": 0, "count": 10},
            },
            "training_validation": {
                "dataset": {
                    "source": "npz",
                    "path": str(dataset_path),
                    "sample_selection": {"mode": "first", "offset": 8, "count": 5},
                }
            },
        }
    }
    train = emit_dataset_artifacts(tmp_path / "build", raw_config=raw)
    held_out = emit_training_validation_dataset_artifacts(tmp_path / "build", raw_config=raw)
    contract = emit_training_validation_split_contract(
        tmp_path / "build", training_artifacts=train, validation_artifacts=held_out
    )

    assert contract["status"] == "incompatible"
    assert contract["overlap_count"] == 2
    assert contract["overlap_indices"] == [8, 9]
    assert contract["statistical_generalization_claim"] is False


def test_training_validation_execution_report_publishes_hls_artifacts(tmp_path: Path) -> None:
    import json

    hls_dir = tmp_path / "hls"
    artifact_dir = hls_dir / "project" / "sol1" / "csim" / "build"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "held_out_validation_summary.json").write_text(
        json.dumps({
            "sample_count": 3,
            "before": {"average_loss": 1.2, "accuracy": 0.0, "correct_count": 0},
            "after": {"average_loss": 0.9, "accuracy": 0.333333, "correct_count": 1},
            "claim_scope": "held_out_validation_mechanism_demonstrated",
        }),
        encoding="utf-8",
    )
    (artifact_dir / "held_out_curve.csv").write_text(
        "phase,sample_count,correct_count,average_loss,accuracy,checkpoint\n",
        encoding="utf-8",
    )
    (artifact_dir / "held_out_predictions_before.csv").write_text(
        "sample,target,prediction,loss\n",
        encoding="utf-8",
    )
    (artifact_dir / "held_out_predictions_after.csv").write_text(
        "sample,target,prediction,loss\n",
        encoding="utf-8",
    )

    compiler = Compiler.__new__(Compiler)
    report = compiler._emit_training_validation_execution_report(
        out_dir=tmp_path / "out",
        hls_dir=hls_dir,
        held_out_dataset_artifacts={"status": "available", "sample_count": 3},
    )

    assert report is not None
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["dataset_sample_count"] == 3
    assert abs(payload["loss_change"] + 0.3) < 1e-9
    assert abs(payload["accuracy_change"] - 0.333333) < 1e-9
    assert payload["statistical_generalization_claim"] is False
    assert Path(payload["curve_csv"]).exists()
    assert Path(payload["predictions_before_csv"]).exists()
    assert Path(payload["predictions_after_csv"]).exists()


def test_dataset_training_reference_persists_momentum_state_across_updates(tmp_path: Path) -> None:
    import json
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference

    graph = Graph("tiny_momentum_dataset_training")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 2))
    graph.add_tensor("dense_out", (1, 2))
    graph.add_tensor("output", (1, 2))
    graph.constants["W"] = np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    graph.constants["B"] = np.zeros((2,), dtype=np.float32)
    graph.add_op(
        "Dense",
        ["input", "W", "B"],
        ["dense_out"],
        name="dense",
        attrs={"in_features": 2, "out_features": 2},
    )
    graph.add_op("Softmax", ["dense_out"], ["output"], name="softmax")

    result = run_training_dataset_reference(
        graph=graph,
        raw_cfg={
            "training": {
                "optimizer": {"type": "momentum", "learning_rate": 0.1, "momentum": 0.75},
                "loss": {"type": "cross_entropy"},
                "execution": {
                    "epochs": 2,
                    "batch_size": 1,
                    "batch_mode": "accumulated",
                    "shuffle": False,
                },
            }
        },
        out_dir=tmp_path,
        inputs=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        targets=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert payload["optimizer_type"] == "momentum"
    assert payload["optimizer_updates"] == 4
    assert payload["optimizer_state_words"] == 6
    assert result.optimizer_state_before_flat_path is not None
    assert result.optimizer_state_after_flat_path is not None
    before = np.fromfile(result.optimizer_state_before_flat_path, dtype=np.float32)
    after = np.fromfile(result.optimizer_state_after_flat_path, dtype=np.float32)
    assert np.array_equal(before, np.zeros_like(before))
    assert np.linalg.norm(after) > 0.0
    assert payload["hardware_domain_reference"]["optimizer_type"] == "momentum"
    hardware_after = np.fromfile(
        payload["hardware_domain_reference"]["optimizer_state_after_ref_bin"],
        dtype=np.float32,
    )
    assert hardware_after.size == after.size
    assert np.linalg.norm(hardware_after) > 0.0


def test_dataset_training_reference_rejects_adam_until_stateful_multi_epoch_support(tmp_path: Path) -> None:
    from fpgai.ir.graph import Graph
    from fpgai.benchmark.training_dataset_reference import run_training_dataset_reference

    graph = Graph("tiny_adam_dataset_training")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 1))
    graph.add_tensor("output", (1, 1))
    graph.constants["W"] = np.asarray([[1.0]], dtype=np.float32)
    graph.constants["B"] = np.zeros((1,), dtype=np.float32)
    graph.add_op("Dense", ["input", "W", "B"], ["output"], name="dense", attrs={"in_features": 1, "out_features": 1})

    import pytest
    with pytest.raises(ValueError, match="supports SGD and Momentum"):
        run_training_dataset_reference(
            graph=graph,
            raw_cfg={
                "training": {
                    "optimizer": {"type": "adam", "learning_rate": 0.01},
                    "loss": {"type": "mse"},
                    "execution": {"epochs": 1, "batch_size": 1, "batch_mode": "accumulated"},
                }
            },
            out_dir=tmp_path,
            inputs=np.asarray([[1.0]], dtype=np.float32),
            targets=np.asarray([[0.0]], dtype=np.float32),
        )
