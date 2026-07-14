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
    hw_path.write_text(json.dumps({"grads": metric(), "weight_delta": metric(), "weights_after": metric()}), encoding="utf-8")
    float_path = tmp_path / "float.json"
    bad = metric(); bad["cosine"] = 0.2; bad["relative_l2"] = 2.0
    float_path.write_text(json.dumps({"grads": bad, "weight_delta": bad, "weights_after": metric()}), encoding="utf-8")
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
