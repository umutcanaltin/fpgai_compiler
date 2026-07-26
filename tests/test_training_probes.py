import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fpgai.validation.training_probes import compare_training_probes, normalize_probe_config
from fpgai.validation.execution_trace import build_training_execution_trace


def test_probe_config_is_explicit_and_disabled_by_default():
    assert normalize_probe_config({}) == {"enabled": False, "selectors": [], "stages": []}
    cfg = {"validation":{"numeric":{"probes":{"enabled":True,"selectors":[{"operator":"dense0","parameter":"weight","tensor_index":[125,783]}],"stages":["parameter_gradient_accumulated"]}}}}
    got = normalize_probe_config(cfg)
    assert got["enabled"] is True
    assert got["selectors"][0]["tensor_index"] == [125, 783]


def test_probe_comparator_finds_first_comparable_divergence(tmp_path: Path):
    base = {"artifact_kind":"fpgai_training_probe_capture","schema_version":1}
    ref = {**base,"producer":"python_reference","entries":[
        {"stage":"forward_input","operator":"dense0","parameter":"dense0.weight","tensor_index":[1,2],"status":"captured","value":2.0},
        {"stage":"parameter_gradient_accumulated","operator":"dense0","parameter":"dense0.weight","tensor_index":[1,2],"status":"captured","value":6.0},
    ]}
    got = {**base,"producer":"hls_csim","entries":[
        {"stage":"forward_input","operator":"dense0","parameter":"dense0.weight","tensor_index":[1,2],"status":"unavailable","value":None},
        {"stage":"parameter_gradient_accumulated","operator":"dense0","parameter":"dense0.weight","tensor_index":[1,2],"status":"captured","value":7.0},
    ]}
    rp=tmp_path/'r.json'; gp=tmp_path/'g.json'; op=tmp_path/'o.json'
    rp.write_text(json.dumps(ref)); gp.write_text(json.dumps(got))
    compare_training_probes(rp,gp,op)
    payload=json.loads(op.read_text())
    assert payload["status"] == "diverged"
    assert payload["first_divergence"]["stage"] == "parameter_gradient_accumulated"


def test_execution_trace_prefers_lower_level_probe_divergence():
    numeric={"status":"failed","comparisons":{"parameter_gradients":{"status":"compared","passed":False,"max_abs_error":1.0}}}
    probes={"first_divergence":{"stage":"parameter_gradient_term","operator":"dense0","tensor_index":[125,783]}}
    trace=build_training_execution_trace(numeric, probes)
    assert trace["first_divergence"]["stage"] == "parameter_gradient_term"
    assert trace["first_divergence"]["source"] == "intermediate_probe"


def test_hls_probe_materializer_reads_direct_csim_values(tmp_path: Path):
    from fpgai.validation.training_probes import materialize_hls_observable_probes

    build = tmp_path / "hls" / "solution1" / "csim" / "build"
    build.mkdir(parents=True)
    np.asarray([1.0, 2.0, 3.0, 4.0, 0.5, 0.25, -0.01, 0.2, 0.19, 1.0, 1.0, 1.0], dtype=np.float32).tofile(
        build / "training_probe_values.bin"
    )
    output = tmp_path / "hls_training_probes.json"
    materialize_hls_observable_probes(
        hls_artifact_dir=tmp_path / "hls",
        probe_config={
            "enabled": True,
            "selectors": [{"operator": "dense0", "parameter": "weight", "tensor_index": [1, 1]}],
            "stages": list((
                "forward_input", "backward_output_gradient", "parameter_gradient_term",
                "parameter_gradient_accumulated", "optimizer_m", "optimizer_v",
                "optimizer_delta", "parameter_before", "parameter_after",
            )),
        },
        parameter_layout=[{"name": "dense0.weight", "shape": [2, 2], "offset": 0}],
        output_path=output,
    )
    payload = json.loads(output.read_text())
    assert payload["source_artifact"].endswith("training_probe_values.bin")
    assert payload["capture_valid"] is True
    assert payload["execution"] == {"loop_entered": True, "selected_index_hit": True, "capture_completed": True}
    assert [entry["status"] for entry in payload["entries"]] == ["captured"] * 9
    assert payload["entries"][2]["value"] == 3.0
    assert payload["entries"][8]["value"] == pytest.approx(0.19)


def test_hls_probe_materializer_rejects_unexecuted_zero_capture(tmp_path: Path):
    from fpgai.validation.training_probes import materialize_hls_observable_probes
    build = tmp_path / "hls" / "solution1" / "csim" / "build"
    build.mkdir(parents=True)
    np.zeros(12, dtype=np.float32).tofile(build / "training_probe_values.bin")
    output = tmp_path / "hls_training_probes.json"
    materialize_hls_observable_probes(
        hls_artifact_dir=tmp_path / "hls",
        probe_config={"enabled": True, "selectors": [{"operator": "dense0", "parameter": "weight", "tensor_index": [1, 1]}], "stages": list((
            "forward_input", "backward_output_gradient", "parameter_gradient_term",
            "parameter_gradient_accumulated", "optimizer_m", "optimizer_v",
            "optimizer_delta", "parameter_before", "parameter_after",
        ))},
        parameter_layout=[{"name": "dense0.weight", "shape": [2, 2], "offset": 0}],
        output_path=output,
    )
    payload = json.loads(output.read_text())
    assert payload["capture_valid"] is False
    assert payload["execution"] == {"loop_entered": False, "selected_index_hit": False, "capture_completed": False}
    assert all(entry["status"] == "unavailable" for entry in payload["entries"])


def test_hls_probe_materializer_reads_activation_boundary_values(tmp_path: Path):
    from fpgai.validation.training_probes import materialize_hls_observable_probes
    build = tmp_path / "hls" / "solution1" / "csim" / "build"
    build.mkdir(parents=True)
    values = np.asarray([
        1.0, 2.0, 3.0, 4.0, 0.5, 0.25, -0.01, 0.2, 0.19,
        1.0, 1.0, 1.0, -0.02, 0.0, 0.75, 0.0,
    ], dtype=np.float32)
    values.tofile(build / "training_probe_values.bin")
    output = tmp_path / "hls_training_probes.json"
    stages = [
        "dense_forward_output", "activation_forward_output",
        "activation_upstream_gradient", "activation_backward_output",
    ]
    materialize_hls_observable_probes(
        hls_artifact_dir=tmp_path / "hls",
        probe_config={"enabled": True, "selectors": [{"operator": "dense0", "parameter": "weight", "tensor_index": [1, 1]}], "stages": stages},
        parameter_layout=[{"name": "dense0.weight", "shape": [2, 2], "offset": 0}],
        output_path=output,
    )
    payload = json.loads(output.read_text())
    assert [entry["value"] for entry in payload["entries"]] == pytest.approx([-0.02, 0.0, 0.75, 0.0])
