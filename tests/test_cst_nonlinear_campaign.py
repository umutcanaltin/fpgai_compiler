"""Controlled nonlinear E1 study must preserve matched graph/source/host evidence."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_nonlinear_campaign import run


def test_nonlinear_chain_design_space_retains_nonlinearity_and_source_pairing(tmp_path):
    target=tmp_path/"nonlinear"
    report=run(output=target,host_cases=2,trace_cases=1)
    assert report["generated_cases"] == 16
    assert report["fixed_functional_graphs"] == 6
    assert report["selected_host_checks"] == 2
    assert report["host_numerical_passes"] == 2
    assert report["executed_row_event_passes"] == 1
    assert report["trained_weights_used"] is False
    assert report["original_vitis_calls"] == 0
    with (target/"cases.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==16
    assert all(row["actual_relu_lowered"]=="True" for row in rows)
    assert all(row["constant_bias_preserved"]=="True" for row in rows)
    assert json.loads((target/"summary.json").read_text())==report
    with pytest.raises(FileExistsError,match="Create-only"):
        run(output=target,host_cases=0,trace_cases=0)


@pytest.mark.parametrize("host,trace", [(-1,0),(17,0),(2,3),(0,1)])
def test_nonlinear_experiment_rejects_invalid_host_budgets(tmp_path,host,trace):
    with pytest.raises(ValueError,match="cases"):
        run(output=tmp_path/"bad",host_cases=host,trace_cases=trace)
