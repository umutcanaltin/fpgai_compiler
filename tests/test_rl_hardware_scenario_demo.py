"""CPU-only six-example source cohort reproducibility tests, no HLS."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from examples.make_hw_rl_scenario_demo import make_demo_catalog
from rl_environment.experiments.hardware_campaign import read_tasks
from rl_environment.experiments.hardware_suite import plan_scenario_suite


def test_hw09_demo_source_models_split_and_samples_are_reproducible(tmp_path):
    first = make_demo_catalog(tmp_path / "first", seed=111)
    second = make_demo_catalog(tmp_path / "second", seed=111)
    a, b = read_tasks(first), read_tasks(second)
    assert len(a) == len(b) == 6
    assert {t.split for t in a} == {"train", "validation", "test"}
    assert [(t.onnx_sha256, t.samples_sha256) for t in a] == [
        (t.onnx_sha256, t.samples_sha256) for t in b
    ]
    for t in a:
        samples = np.load(t.samples, allow_pickle=False)
        assert samples.dtype == np.float32
        assert samples.shape[0] == 2
        assert np.isfinite(samples).all()
    suite = plan_scenario_suite(a, holdout="structure", seed=111)
    assert suite["task_count"] == 6
    assert suite["native_vitis_calls"] == 0
    named = {case["id"]: case for case in suite["scenarios"]}
    for split in ("train", "validation", "test"):
        assert named[f"{split}_cross_family_fixed"]["status"] == "SOURCE_PLAN_AVAILABLE"
        assert named[f"{split}_cross_family_search"]["status"] == "SOURCE_PLAN_AVAILABLE"
        assert named[f"{split}_family_fixed"]["status"] == "UNAVAILABLE_SOURCE_COHORT"


def test_hw09_demo_refuses_to_overwrite_models_or_reuse_output_root(tmp_path):
    root = tmp_path / "create_only"
    make_demo_catalog(root)
    with pytest.raises(FileExistsError, match="overwrite"):
        make_demo_catalog(root)
    assert len(list((root / "models").glob("*.onnx"))) == 6
