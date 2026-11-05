"""Build a create-only six-graph HW-RL source-only scientific example cohort.

Two structurally distinct graphs per train/validation/test split: one
convolutional and one MatMul/dense. All weights and input samples are
deterministic synthetic fixtures; models are not trained. This only prepares
a research scenario *catalog*; native HLS preflight/qualification is NOT
implied, and no Vitis tools are invoked.

Run from repository root:
  python -m examples.make_hw_rl_scenario_demo \
    build/rl_hw09_scenario_inputs_01 --seed 111
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.make_mlp_onnx import save_model as save_mlp
from examples.make_neural_network_onnx import save_model as save_cnn

MODELS = (
    ("train", "train_cnn_pool", "cnn_pool_dense"),
    ("train", "train_dense_mlp", "dense_mlp"),
    ("validation", "validation_cnn_strided", "cnn_strided_dense"),
    ("validation", "validation_dense_toy", "dense_toy"),
    ("test", "test_cnn_two_conv", "cnn_two_conv_softmax"),
    ("test", "test_dense_chain", "dense_chain"),
)
CAPS = (53200, 106400, 220, 280, 0)


def _dense_model(network):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if network == "dense_toy":
        w1 = numpy_helper.from_array(
            (np.arange(12, dtype=np.float32).reshape(4, 3) - 5.) * .01,
            name="w1",
        )
        nodes = [
            helper.make_node("MatMul", ["x", "w1"], ["hidden"]),
            helper.make_node("Relu", ["hidden"], ["out"]),
        ]
        inputs = 4
        outputs = 3
        weights = [w1]
    elif network == "dense_chain":
        w1 = numpy_helper.from_array(
            (np.arange(20, dtype=np.float32).reshape(5, 4) - 8.) * .01,
            name="w1",
        )
        w2 = numpy_helper.from_array(
            (np.arange(8, dtype=np.float32).reshape(4, 2) - 3.) * .0125,
            name="w2",
        )
        nodes = [
            helper.make_node("MatMul", ["x", "w1"], ["hidden"]),
            helper.make_node("Relu", ["hidden"], ["activation"]),
            helper.make_node("MatMul", ["activation", "w2"], ["out"]),
        ]
        inputs = 5
        outputs = 2
        weights = [w1, w2]
    else:
        raise ValueError("Unknown deterministic dense model")
    graph = helper.make_graph(
        nodes, network,
        [helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, [1, inputs],
        )],
        [helper.make_tensor_value_info(
            "out", TensorProto.FLOAT, [1, outputs],
        )],
        initializer=weights,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
        producer_name="fpgai-hw-rl-synthetic-source-only",
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def make_demo_catalog(root: str | Path, *, seed=111,
                      part="xc7z020clg400-1", clock_ns=14.0,
                      resource_caps=CAPS, max_absolute_error=.1,
                      max_parallel_units=8) -> Path:
    """Create original samples/models under a fresh isolated directory."""
    from rl_environment.experiments.hardware_campaign import (
        TASK_SCHEMA, read_tasks,
    )
    from rl_environment.experiments.hardware_suite import (
        plan_scenario_suite,
    )

    if type(seed) is not int or not 0 <= seed <= 2**31:
        raise ValueError("Bounded explicit sample seed required")
    if (type(clock_ns) not in (int, float) or
            clock_ns != round(float(clock_ns), 2) or
            not 2. <= clock_ns <= 32.):
        raise ValueError("Canonical native target period in [2,32] ns required")
    if (len(resource_caps) != 5 or
            any(type(x) is not int or x < 0 for x in resource_caps)):
        raise ValueError("Five integer resource caps required")
    if type(max_parallel_units) is not int or not 1 <= max_parallel_units <= 128:
        raise ValueError("Valid max parallel unit bound required")
    if (type(max_absolute_error) not in (int, float)
            or not np.isfinite(max_absolute_error)
            or max_absolute_error <= 0):
        raise ValueError("Positive original numerical error bound required")

    dest = Path(root).expanduser()
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Example catalog is create-only; refusing overwrite")
    dest.mkdir(parents=True, exist_ok=False)
    models = dest / "models"
    samples = dest / "samples"
    models.mkdir(exist_ok=False)
    samples.mkdir(exist_ok=False)
    tasks = []
    for index, (split, name, network) in enumerate(MODELS):
        model = models / f"{name}.onnx"
        sample = samples / f"{name}.npy"
        if network in ("cnn_pool_dense", "cnn_strided_dense", "cnn_two_conv_softmax"):
            save_cnn(network, model)
        elif network == "dense_mlp":
            save_mlp(model)
        else:
            with model.open("xb") as f:
                f.write(_dense_model(network).SerializeToString())
        import onnx

        proto = onnx.load(str(model), load_external_data=False)
        info = proto.graph.input[0].type.tensor_type
        shape = tuple(int(x.dim_value) for x in info.shape.dim)
        if not shape or any(dim < 1 for dim in shape):
            raise ValueError("Example model lacks static runtime input shape")
        data = np.random.default_rng(seed + index).uniform(
            -.25, .25, size=(2, *shape),
        ).astype(np.float32)
        with sample.open("xb") as f:
            np.save(f, data, allow_pickle=False)
        tasks.append({
            "id": name, "split": split,
            "onnx": str(model.relative_to(dest)),
            "samples": str(sample.relative_to(dest)),
            "part": part, "resource_caps": list(resource_caps),
            "max_absolute_error": float(max_absolute_error),
            "initial_clock_ns": float(clock_ns),
            "max_parallel_units": max_parallel_units,
        })
    path = dest / "tasks.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump({"schema": TASK_SCHEMA, "tasks": tasks},
                  f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    verified = read_tasks(path)
    suite = plan_scenario_suite(verified, holdout="structure", seed=seed)
    assert suite["task_count"] == len(MODELS)
    assert suite["source_only"] and suite["native_vitis_calls"] == 0
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", help="New directory, never overwritten")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--part", default="xc7z020clg400-1")
    parser.add_argument("--clock-ns", type=float, default=14.)
    parser.add_argument("--resource-caps", type=int, nargs=5, default=CAPS)
    parser.add_argument("--max-absolute-error", type=float, default=.1)
    parser.add_argument("--max-parallel-units", type=int, default=8)
    args = parser.parse_args(argv)
    path = make_demo_catalog(
        args.output_root, seed=args.seed, part=args.part,
        clock_ns=args.clock_ns, resource_caps=tuple(args.resource_caps),
        max_absolute_error=args.max_absolute_error,
        max_parallel_units=args.max_parallel_units,
    )
    print(json.dumps({
        "status": "SOURCE_ONLY_EXAMPLE_TASK_CATALOG",
        "manifest": str(path), "tasks": len(MODELS),
        "split_count": 3, "operator_families": 2,
        "original_vitis_calls": 0, "native_qualified": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
