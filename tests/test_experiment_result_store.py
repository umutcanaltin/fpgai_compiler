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
        "config_path": "configs/examples/default_compile.yml",
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


def test_failed_design_is_retryable_and_latest_attempt_is_canonical(tmp_path):
    store = ResultStore(tmp_path / "exp_retry")
    base = {
        "design_index": 0,
        "design_name": "retry_me",
        "duration_sec": 0.1,
        "commit_hash": "abc",
        "config_path": "configs/retry.yml",
        "model_path": "m.onnx",
        "tool_version": "test",
        "board": "kv260",
        "parameters": {},
        "metrics": {},
    }
    store.append_record({**base, "status": "failed", "returncode": 2, "error": "first attempt"})
    assert store.completed_design_names() == set()

    store.append_record({**base, "status": "passed", "returncode": 0})
    payload = json.loads(store.results_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["attempt_count"] == 2
    assert payload["result_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["passed_count"] == 1
    assert payload["results"][0]["status"] == "passed"
    assert store.completed_design_names() == {"retry_me"}
