"""CPU-only create-only chain experiment qualification and integrity tests."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_chain_campaign import run


def test_three_and_four_op_campaign_preserves_source_and_rows(tmp_path):
    destination = tmp_path / "chain"
    result = run(output=destination, max_cases=6, host_cases=0, full_trace_cases=0)
    assert result["cases"] == 6
    assert result["chains_with_three_operators"] > 0
    assert result["chains_with_four_operators"] > 0
    assert result["host_checked"] == 0
    assert result["vitis_calls"] == 0
    with (destination/"cases.csv").open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == 6
    assert all(r["whole_tensor_cpp_sha256"] != r["row_chain_cpp_sha256"]
               for r in records)
    assert all(int(r["original_vitis_calls"]) == 0 for r in records)
    assert json.loads((destination/"summary.json").read_text()) == result
    with pytest.raises(FileExistsError, match="Create-only"):
        run(output=destination, max_cases=6, host_cases=0, full_trace_cases=0)


@pytest.mark.parametrize("n,host,trace", [(0,0,0), (50,0,0), (4,5,0), (4,0,5),
                                          (2,-1,0), (2,0,-1)])
def test_chain_campaign_rejects_bad_budgets(tmp_path, n, host, trace):
    with pytest.raises(ValueError, match="max_cases|budget"):
        run(output=tmp_path/"bad", max_cases=n,
            host_cases=host, full_trace_cases=trace)
