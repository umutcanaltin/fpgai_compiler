"""Source-only HW-RL-09 scenario manifest and structural holdout tests.

These tests do not execute vendor tools, produce rewards or make HLS claims.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from examples.make_mlp_onnx import save_model as save_mlp
from examples.make_neural_network_onnx import save_model as save_cnn
from rl_environment.experiments.hardware_campaign import TASK_SCHEMA, read_tasks
from rl_environment.experiments.hardware_suite import (
    SUITE_SCHEMA, plan_scenario_suite, write_scenario_suite, main,
)

CAPS = [53200, 106400, 220, 280, 0]
PART = "xc7z020clg400-1"


def _catalog(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    specs = [
        ("train", "cnn_pool_dense", "train_cnn"),
        ("validation", "cnn_two_conv_softmax", "validation_cnn"),
        ("test", "dense_mlp", "test_dense"),
    ]
    tasks = []
    for split, network, ident in specs:
        model = model_root / f"{ident}.onnx"
        if network == "dense_mlp":
            save_mlp(model)
        else:
            save_cnn(network, model)
        array = model_root / f"{ident}.npy"
        np.save(array, np.zeros((2, 1, 6, 6), np.float32), allow_pickle=False)
        tasks.append({
            "id": ident, "split": split,
            "onnx": str(model.relative_to(tmp_path)),
            "samples": str(array.relative_to(tmp_path)),
            "part": PART,
            "resource_caps": CAPS,
            "max_absolute_error": .1,
            "initial_clock_ns": 14.0,
        })
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"schema": TASK_SCHEMA, "tasks": tasks}))
    return manifest


def test_source_suite_marks_missing_regimes_explicit_and_keeps_full_splits(tmp_path):
    source = _catalog(tmp_path)
    tasks = read_tasks(source)
    plan = plan_scenario_suite(tasks)
    assert plan["schema"] == SUITE_SCHEMA
    assert plan["source_only"] is True
    assert plan["native_vitis_calls"] == 0
    assert len(plan["scenarios"]) == 36
    assert plan["task_count"] == 3
    assert len(plan["all_task_definitions"]) == 3
    by_name = {x["id"]: x for x in plan["scenarios"]}
    for split in ("train", "validation", "test"):
        assert by_name[f"{split}_same_fixed"]["status"] == "SOURCE_PLAN_AVAILABLE"
        assert by_name[f"{split}_same_search"]["status"] == "SOURCE_PLAN_AVAILABLE"
        assert by_name[f"{split}_family_fixed"]["status"] == "UNAVAILABLE_SOURCE_COHORT"
        assert by_name[f"{split}_cross_family_search"]["status"] == "UNAVAILABLE_SOURCE_COHORT"

    suite_root = tmp_path / "output_suite"
    assert write_scenario_suite(suite_root, plan) == suite_root
    assert (suite_root / "suite.json").is_file()
    for split in ("train", "validation", "test"):
        saved = read_tasks(suite_root / f"tasks_{split}.json")
        assert len(saved) == 1
        assert saved[0].split == split
        assert (suite_root / "scenarios" / f"{split}_same_fixed.json").is_file()
        assert read_tasks(
            suite_root / "scenarios" / f"{split}_same_fixed.json"
        )[0].onnx_sha256 == saved[0].onnx_sha256
    assert list(suite_root.rglob("case_*")) == []
    with pytest.raises(FileExistsError, match="overwrite"):
        write_scenario_suite(suite_root, plan)


def test_suite_cli_preview_is_read_only_and_explicit_output_is_create_only(
        tmp_path, capsys):
    source = _catalog(tmp_path)
    assert main(["--tasks", str(source), "--holdout", "structure"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source_only"] is True
    assert result["native_vitis_calls"] == 0
    assert result["source_manifest_sha256"]
    assert not (tmp_path / "runs").exists()
    assert main([
        "--tasks", str(source), "--output-root", str(tmp_path / "runs")
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["native_vitis_calls"] == 0
    assert output["available_scenarios"] > 0
    assert output["unavailable_scenarios"] > 0
    assert main([
        "--tasks", str(source), "--output-root", str(tmp_path / "runs")
    ]) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_structural_holdout_refuses_byte_distinct_but_same_graph(tmp_path):
    source = _catalog(tmp_path)
    manifest = json.loads(source.read_text())
    a = tmp_path / manifest["tasks"][0]["onnx"]
    test_path = tmp_path / manifest["tasks"][2]["onnx"]
    model = onnx.load(str(a))
    model.producer_name = "different-metadata-not-a-different-network"
    onnx.save(model, str(test_path))
    tasks = read_tasks(source)
    assert tasks[0].onnx_sha256 != tasks[2].onnx_sha256
    with pytest.raises(ValueError, match="graph structure leaks"):
        plan_scenario_suite(tasks, holdout="structure")
    weak = plan_scenario_suite(tasks, holdout="bytes")
    assert weak["holdout"] == "bytes"
    assert weak["source_only"] is True


def test_invalid_reward_contract_rejects_before_any_suite_allocation(tmp_path):
    tasks = read_tasks(_catalog(tmp_path))
    with pytest.raises(ValueError, match="Unknown original-HLS"):
        plan_scenario_suite(tasks, reward_mode="fake_proxy")
    with pytest.raises(ValueError, match="positive and finite"):
        plan_scenario_suite(tasks, reward_reference_ns=float("nan"))
    assert not (tmp_path / "output_suite").exists()
