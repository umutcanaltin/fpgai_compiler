import json
from pathlib import Path

from fpgai.experiments.result_store import ResultStore


def test_result_store_writes_json_csv_and_resume(tmp_path):
    store = ResultStore(tmp_path / "exp")
    store.append_record({
        "design_index": 0,
        "design_name": "p0",
        "status": "passed",
        "returncode": 0,
        "duration_sec": 0.1,
        "commit_hash": "abc",
        "config_path": "fpgai.yml",
        "model_path": "m.onnx",
        "tool_version": "test",
        "board": "kv260",
        "parameters": {"policy": "balanced"},
        "metrics": {"latency_ms": 1.2},
    })
    assert store.results_path.exists()
    assert store.csv_path.exists()
    payload = json.loads(store.results_path.read_text())
    assert payload["result_count"] == 1
    assert store.completed_design_names() == {"p0"}
    assert "param.policy" in store.csv_path.read_text()
