"""C2 fair source cost tests. Synthetic timing never treated as FPGA evidence."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json

import pytest

from fpgai_research.experiments import cst_paper_c2_fair_source_cost as sut


@dataclass
class FakeDecision:
    status: str
    original_state: object
    parent_state_sha256: str
    action: int
    cpp_sha256: str
    state_sha256: str
    certificate_sha256: str
    e0_certified: bool


class FakeCatalog:
    def __init__(self, decisions):
        self.decisions = decisions

    def mask(self):
        return tuple(d.e0_certified for d in self.decisions)


@pytest.fixture
def fake_source(monkeypatch):
    calls = []
    def hashed(state):
        return state["hash"]

    def compiler(state, action):
        calls.append(action)
        allowed = action in (0, 2, 5)
        return FakeDecision(
            sut.CERTIFIED if allowed else "REJECTED",
            state, state["hash"], action, "a" * 64, "b" * 64,
            "c" * 64, allowed)

    def catalog(state, actions):
        return FakeCatalog(tuple(compiler(state, a) for a in actions))

    def selected(catalog, i, *, current_state):
        value = catalog.decisions[i]
        if not value.e0_certified or value.original_state is not current_state:
            raise ValueError("Not certified for parent")
        return value

    monkeypatch.setattr(sut, "unified_transition", compiler)
    monkeypatch.setattr(sut, "unified_state_hash", hashed)
    monkeypatch.setattr(sut, "build_catalog", catalog)
    monkeypatch.setattr(sut, "select_certified", selected)
    monkeypatch.setattr(
        sut, "_case", lambda n, w: ({"hash": "x"*64}, tuple(range(8))))
    return calls


def test_direct_lazy_eager_equal_results_distinct_real_calls(fake_source):
    state = {"hash": "x"*64}
    actions = tuple(range(8))
    queries = (0, 2, 0, 2, 0)
    outcomes = {
        method: sut._method(state, actions, queries, method)
        for method in sut.METHODS
    }
    assert [outcomes[m]["source_compiler_calls"] for m in sut.METHODS] == [
        5, 2, 8]
    assert len(fake_source) == 15
    assert all(outcomes[m]["decision_identities"] ==
               outcomes["direct"]["decision_identities"]
               for m in sut.METHODS)
    assert all(outcomes[m]["wall_ns"] >= 0 for m in sut.METHODS)


def test_preregistered_query_sequences_identical_for_all_methods(fake_source):
    r = sut.evaluate("chain_d2", 2, seeds=(111, 222),
                     request_counts=(1, 4), streams=sut.STREAMS)
    assert r["source_preflight_compiler_calls_excluded_from_all_methods"] == 8
    assert r["preflight_known_certified_action_indices"] == [0, 2, 5]
    rows = r["records"]
    assert len(rows) == 2 * 2 * 2 * 3
    per_trial = {}
    for x in rows:
        key = (x["stream"], x["seed"], x["queries"])
        per_trial.setdefault(key, []).append(x)
        assert x["matched_compiled_results"] is True
    for block in per_trial.values():
        assert {tuple(x["request_indices"]) for x in block}.__len__() == 1
        assert {x["result_sha256"] for x in block}.__len__() == 1
        assert {x["method"] for x in block} == set(sut.METHODS)
        assert sorted(x["execution_order"] for x in block) == [0,1,2]
        q = block[0]["queries"]
        for x in block:
            expected = q if x["method"] == "direct" else (
                len(set(x["request_indices"])) if x["method"] == "lazy"
                else 8)
            assert x["source_compiler_calls"] == expected


def test_single_query_eager_has_more_compile_calls_than_lazy(fake_source):
    r = sut.run(cohorts=(sut.COHORTS[0],), seeds=(111,),
                request_counts=(1,), streams=("repeat_certified",))
    block = {x["method"]: x for x in r["records"]}
    assert block["eager"]["source_compiler_calls"] == 8
    assert block["lazy"]["source_compiler_calls"] == 1
    assert block["direct"]["source_compiler_calls"] == 1
    assert r["hardware_latency"] is None
    assert r["native_vivado_calls"] == 0
    assert r["trained_rl_agents"] == 0
    assert r["original_source_preflight_compiler_calls_excluded"] == 8
    assert r["all_compiled_source_results_matched"] is True
    assert len(r["by_condition"]) == 3


def test_randomized_order_does_not_change_requested_positives(fake_source):
    first = sut.evaluate("chain_d2", 2, request_counts=(8,),
                         seeds=(111,), streams=("cycle_certified",))
    second = sut.evaluate("chain_d2", 2, request_counts=(8,),
                          seeds=(111,), streams=("cycle_certified",))
    lhs = [(r["method"], r["request_indices"],
            r["source_compiler_calls"], r["execution_order"])
           for r in first["records"]]
    rhs = [(r["method"], r["request_indices"],
            r["source_compiler_calls"], r["execution_order"])
           for r in second["records"]]
    assert lhs == rhs


@pytest.mark.parametrize("case", [
    {"request_counts": (0,)},
    {"seeds": (111,111)},
    {"streams": ("unsupported",)},
    {"request_counts": (1,1)},
])
def test_unpreregistered_benchmarks_fail_closed(fake_source, case):
    with pytest.raises(ValueError):
        sut.evaluate("chain_d2",2,**case)


def test_unmatched_compiler_source_receipt_rejected(fake_source, monkeypatch):
    real = sut.unified_transition
    flip = {"n":0}
    def unstable(state,action):
        d = real(state,action)
        if d.e0_certified:
            flip["n"] += 1
            if flip["n"] % 2:
                d.cpp_sha256 = "f" * 64
        return d
    monkeypatch.setattr(sut,"unified_transition",unstable)
    with pytest.raises(AssertionError,match="compiler source outputs diverged"):
        sut.evaluate("chain_d2",2,seeds=(111,),
                     request_counts=(1,),streams=("repeat_certified",))


def test_create_only_sha_audit_and_original_results_immutable(fake_source,tmp_path):
    report = sut.run(cohorts=(sut.COHORTS[0],),seeds=(111,),
                     request_counts=(1,),streams=("repeat_certified",))
    original = json.dumps(report,sort_keys=True)
    out = tmp_path/"new_c2_cost"
    sut.create_only(report,output=out,source_revision="a"*40)
    audit = sut.audit(out)
    assert audit["measured_cases"] == 3
    assert audit["new_vendor_calls"] == 0
    assert len(audit["report_sha256"]) == 64
    assert json.dumps(report,sort_keys=True) == original
    with pytest.raises(FileExistsError):
        sut.create_only(report,output=out,source_revision="a"*40)
    path = out/"report.json"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="checksum drift"):
        sut.audit(out)


def test_existing_actual_compiler_runs_small_cpu_smoke():
    # Genuine compiler source paths, not mock-only correctness.
    r = sut.run(cohorts=(sut.COHORTS[0],), seeds=(111,),
                request_counts=(1,), streams=("repeat_certified",))
    assert r["schema"] == sut.SCHEMA
    assert r["cohorts"] == 1
    assert r["measured_cases"] == 3
    assert r["all_compiled_source_results_matched"] is True
    assert r["original_source_preflight_compiler_calls_excluded"] == 8
    calls = {x["method"]: x["source_compiler_calls"] for x in r["records"]}
    assert calls == {"direct": 1, "lazy": 1, "eager": 8}
