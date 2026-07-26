import json
from pathlib import Path

from fpgai.validation.execution_trace import (
    build_training_execution_trace,
    write_training_execution_trace,
)


def _comparison(passed: bool, index: int = 0):
    return {
        "status": "compared",
        "passed": passed,
        "max_abs_error": 0.0 if passed else 7.0,
        "worst_mismatch": {
            "flat_index": index,
            "parameter": "dense0.weight",
            "tensor_index": [125, 783],
        },
    }


def test_execution_trace_identifies_parameter_gradient_as_first_divergence():
    report = {
        "status": "failed",
        "comparisons": {
            "pre_update_loss": _comparison(True),
            "parameter_gradients": _comparison(False, 98783),
            "optimizer_m_after": _comparison(False, 98783),
            "optimizer_v_after": _comparison(False, 100604),
            "optimizer_step_after": _comparison(True),
            "weights_after": _comparison(False, 14956),
            "biases_after": _comparison(False, 78),
            "post_update_loss": _comparison(False),
        },
    }
    trace = build_training_execution_trace(report)
    assert trace["status"] == "diverged"
    assert trace["first_divergence"]["stage"] == "parameter_gradient"
    assert trace["first_divergence"]["role"] == "parameter_gradients"
    assert trace["first_divergence"]["worst_mismatch"]["tensor_index"] == [125, 783]


def test_execution_trace_writer(tmp_path: Path):
    source = tmp_path / "numeric.json"
    target = tmp_path / "trace.json"
    source.write_text(json.dumps({"status": "ready", "comparisons": {"pre_update_loss": _comparison(True)}}))
    written = write_training_execution_trace(source, target)
    assert written == target
    payload = json.loads(target.read_text())
    assert payload["artifact_kind"] == "fpgai_training_execution_trace"
    assert payload["source_numeric_report"] == str(source)
