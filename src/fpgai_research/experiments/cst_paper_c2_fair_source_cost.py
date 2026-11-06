"""C2 paper experiment: fair source-compiler cost of eager vs lazy vs direct.

Three ACTUAL methods on precisely matched source states/requests:
  direct: recompile the requested action for every query;
  lazy: compile on FIRST use, validate/cache by exact state/source identity;
  eager: precompile all eight typed actions, choose from frozen catalog.

This is a **candidate-consumption cost** benchmark, NOT training, search
quality, native FPGA timing or demonstrated RL benefit. A catalog
preflight identifies certified actions and is explicitly accounted for
as EXPERIMENTAL SETUP (excluded from each timed algorithm). The
benchmark therefore cannot estimate blind policy acceptance costs.

Run source-only, produce create-only SHA-attested receipts, preserve
negative results and all data. No GCC/Vitis/Vivado or GitHub Actions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from statistics import median
from time import perf_counter_ns

from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import (
    _git_revision,
)
from fpgai_research.experiments.cst_r3f_c2_variable_graph_coverage import (
    COHORTS, _case,
)
from fpgai_research.model_ir.compiled_action_catalog import (
    build_catalog, select_certified,
)
from fpgai_research.model_ir.unified_decision_interface import (
    unified_state_hash, unified_transition,
)

SCHEMA = "fpgai-research/C2-direct-lazy-eager-source-compiler-cost/v1"
METHODS = ("direct", "lazy", "eager")
STREAMS = ("repeat_certified", "cycle_certified")
DEFAULT_SEEDS = (111, 222, 333)
DEFAULT_REQUESTS = (1, 8, 32)
CERTIFIED = "BOUNDED_CERTIFIED_E0"


def _receipt(decision) -> tuple:
    if decision.status != CERTIFIED or decision.e0_certified is not True:
        raise ValueError("C2 cost trial requested noncertified source action")
    if not (isinstance(decision.cpp_sha256, str)
            and isinstance(decision.state_sha256, str)
            and isinstance(decision.certificate_sha256, str)):
        raise ValueError("C2 cost trial lacks compiled source identity")
    return (decision.status, decision.cpp_sha256,
            decision.state_sha256, decision.certificate_sha256)


def _method(state, actions, indices: tuple[int, ...], method: str):
    """Timed workload includes actual source calls, cache checks and selection."""
    if method not in METHODS:
        raise ValueError("Unknown source compiler cost method")
    original_sha = unified_state_hash(state)
    calls = 0
    seen = {}
    decisions = []
    started = perf_counter_ns()
    if method == "eager":
        catalog = build_catalog(state, actions)
        # Use the compiled catalog's own recorded transition count.
        calls = catalog.compiler_transitions
        if calls != len(actions):
            raise ValueError("Eager source catalog call-count receipt incorrect")
    for index in indices:
        if method == "direct":
            decision = unified_transition(state, actions[index])
            calls += 1
        elif method == "lazy":
            if index not in seen:
                seen[index] = unified_transition(state, actions[index])
                calls += 1
            decision = seen[index]
            # A lazy cache must never reuse a certificate on a different
            # parent source identity or after source mutation.
            if (decision.original_state is not state
                    or decision.parent_state_sha256 != unified_state_hash(state)
                    or decision.action != actions[index]):
                raise ValueError("C2 lazy source cache parent/action changed")
        else:
            decision = select_certified(catalog, index, current_state=state)
        decisions.append(decision)
    elapsed = perf_counter_ns() - started
    if unified_state_hash(state) != original_sha:
        raise ValueError("Source compiler cost benchmark mutated its parent")
    return {
        "method": method,
        "wall_ns": elapsed,
        "source_compiler_calls": calls,
        "distinct_requested_actions": len(set(indices)),
        "query_count": len(indices),
        "decision_identities": [_receipt(dec) for dec in decisions],
    }


def _stream(certified: tuple[int, ...], *, kind: str, count: int, seed: int):
    if not certified or count < 1 or kind not in STREAMS:
        raise ValueError("Invalid C2 source cost request stream")
    if kind == "repeat_certified":
        return (certified[0],) * count
    # Stable seed chooses the cycle offset; every method gets the same requests.
    shift = random.Random(seed).randrange(len(certified))
    return tuple(certified[(shift + i) % len(certified)] for i in range(count))


def evaluate(name: str, width: int, *, request_counts=DEFAULT_REQUESTS,
             seeds=DEFAULT_SEEDS, streams=STREAMS) -> dict:
    if (name, width) not in COHORTS:
        raise ValueError("C2 cost cohort must be in frozen R3f population")
    if (not seeds or not request_counts
            or any(type(x) is not int or x < 1 for x in request_counts)
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or len(set(request_counts)) != len(request_counts)
            or not streams or any(x not in STREAMS for x in streams)):
        raise ValueError("C2 invalid preregistered CPU benchmark protocol")
    state, actions = _case(name, width)
    if len(actions) != 8:
        raise ValueError("R3f source candidate count changed")
    parent = unified_state_hash(state)
    # Preflight is explicit selection of known-certified actions, NOT free
    # exploration or an unbiased query stream. It is excluded from ALL three
    # measured methods and counted separately in every report.
    pilot = build_catalog(state, actions)
    certified = tuple(i for i, value in enumerate(pilot.mask()) if value)
    if not certified:
        raise ValueError("C2 benchmark has no source-certified candidate")
    expected = {i: _receipt(pilot.decisions[i]) for i in certified}
    rows = []
    for count in request_counts:
        for stream_name in streams:
            for seed in seeds:
                indices = _stream(certified, kind=stream_name,
                                  count=count, seed=seed)
                # A deterministic balanced Latin rotation: for full
                # (111,222,333) each method appears once in each timing
                # position, preventing a systematic eager-first bias.
                order = list(METHODS)
                random.Random(count * 1009 + width * 251 +
                              len(name) * 29 +
                              len(stream_name) * 41).shuffle(order)
                shift = list(seeds).index(seed) % len(METHODS)
                order = order[shift:] + order[:shift]
                measured = {}
                for method in order:
                    measured[method] = _method(state, actions, indices, method)
                identical = (all(
                    tuple(x["decision_identities"]) ==
                    tuple(measured["direct"]["decision_identities"])
                    for x in measured.values())
                    and all(
                        tuple(x["decision_identities"]) ==
                        tuple(expected[i] for i in indices)
                        for x in measured.values()))
                if not identical:
                    raise AssertionError(
                        "Direct, lazy and eager compiler source outputs diverged")
                for method in METHODS:
                    row = measured[method]
                    expected_calls = (
                        count if method == "direct"
                        else len(set(indices)) if method == "lazy"
                        else len(actions))
                    if row["source_compiler_calls"] != expected_calls:
                        raise AssertionError("C2 direct/lazy/eager call audit mismatch")
                    rows.append({
                        "cohort": name, "width": width,
                        "source_parent_sha256": parent,
                        "stream": stream_name, "seed": seed,
                        "queries": count, "distinct_requested": len(set(indices)),
                        "method": method, "execution_order": order.index(method),
                        "wall_ns": row["wall_ns"],
                        "source_compiler_calls": row["source_compiler_calls"],
                        "result_sha256": sha256(json.dumps(
                            row["decision_identities"],
                            separators=(",", ":")).encode()).hexdigest(),
                        "request_indices": list(indices),
                        "matched_compiled_results": identical,
                    })
    return {
        "cohort": name,
        "width": width,
        "parent_source_sha256": parent,
        "source_preflight_compiler_calls_excluded_from_all_methods":
            len(actions),
        "preflight_known_certified_action_indices": list(certified),
        "preflight_known_certified_action_count": len(certified),
        "records": rows,
    }


def summarize(studies: list[dict], *, seeds, requests, streams) -> dict:
    rows = [row for study in studies for row in study["records"]]
    expected = len(studies) * len(seeds) * len(requests) * len(streams) * len(METHODS)
    if len(rows) != expected:
        raise ValueError("C2 cost benchmark missing a planned observation")
    results = []
    for stream in streams:
        for count in requests:
            for method in METHODS:
                sample = [r for r in rows if r["stream"] == stream
                          and r["queries"] == count and r["method"] == method]
                if not sample:
                    raise ValueError("C2 cost timing sample missing")
                timings = sorted(r["wall_ns"] for r in sample)
                results.append({
                    "stream": stream, "queries": count, "method": method,
                    "n": len(timings),
                    "median_wall_ns": int(median(timings)),
                    "min_wall_ns": timings[0],
                    "max_wall_ns": timings[-1],
                    "median_compiler_calls": median(
                        r["source_compiler_calls"] for r in sample),
                })
    return {
        "schema": SCHEMA,
        "status": "SOURCE_COMPILER_COST_OBSERVATIONS_NO_SEARCH_SUPERIORITY_CLAIM",
        "cohorts": len(studies),
        "measured_cases": len(rows),
        "seeds": list(seeds),
        "query_counts": list(requests),
        "streams": list(streams),
        "original_source_preflight_compiler_calls_excluded":
            sum(s["source_preflight_compiler_calls_excluded_from_all_methods"]
                for s in studies),
        "all_compiled_source_results_matched": all(
            r["matched_compiled_results"] for r in rows),
        "by_condition": results,
        "records": rows,
        "native_gcc_calls": 0, "native_vitis_calls": 0,
        "native_vivado_calls": 0, "trained_rl_agents": 0,
        "hardware_latency": None,
        "limitations": [
            "Same eight predeclared candidates as R3f; not unseen graph families",
            "Certified-only queries chosen by an EXCLUDED full-catalog preflight",
            "Eager precompilation is compared with an equally valid lazy cache",
            "CPU wall measurements are platform/order sensitive, not timing proof",
            "No search quality, multi-state search, learning or FPGA advantage",
            "Full default three-seed runs use balanced method timing order; quick mode does not",
            "Repeated requests are counterbalanced trials, not independent models",
        ],
    }


def run(*, cohorts=COHORTS, seeds=DEFAULT_SEEDS,
        request_counts=DEFAULT_REQUESTS, streams=STREAMS):
    if not cohorts or any(tuple(c) not in COHORTS for c in cohorts):
        raise ValueError("C2 cost experiment requires original prespecified cohorts")
    if len(set(tuple(c) for c in cohorts)) != len(cohorts):
        raise ValueError("C2 cost experiment has duplicate cohort")
    studies = [evaluate(n, w, request_counts=request_counts,
                        seeds=seeds, streams=streams) for n, w in cohorts]
    return summarize(studies, seeds=seeds, requests=request_counts,
                     streams=streams)


def create_only(report: dict, *, output: str | Path,
                source_revision: str) -> None:
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("C2 source cost output is create-only")
    if (report.get("schema") != SCHEMA
            or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
            or not report.get("all_compiled_source_results_matched")
            or report.get("native_vivado_calls") != 0
            or report.get("native_vitis_calls") != 0):
        raise ValueError("C2 cost run not source-qualified for export")
    dest.mkdir(parents=True, exist_ok=False)
    src = {**report, "experiment_source_revision": source_revision}
    text = json.dumps(src, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (dest / "report.json").write_text(text)
    manifest = {"report.json": sha256(text.encode()).hexdigest()}
    (dest / "artifact_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def audit(study: str | Path):
    root = Path(study)
    receipt = root / "report.json"
    manifest = root / "artifact_sha256.json"
    if (root.is_symlink() or not root.is_dir()
            or receipt.is_symlink() or manifest.is_symlink()
            or not receipt.is_file() or not manifest.is_file()):
        raise ValueError("C2 cost original output missing or unsafe")
    expected = json.loads(manifest.read_text())
    if (expected != {"report.json": sha256(receipt.read_bytes()).hexdigest()}):
        raise ValueError("C2 cost report checksum drift")
    j = json.loads(receipt.read_text())
    if (j.get("schema") != SCHEMA
            or j.get("all_compiled_source_results_matched") is not True
            or len(j.get("records", [])) != j.get("measured_cases")
            or j.get("native_vivado_calls") != 0
            or j.get("native_vitis_calls") != 0
            or re.fullmatch(r"[0-9a-f]{40}",
                            j.get("experiment_source_revision") or "") is None):
        raise ValueError("C2 cost source report schema or source identity invalid")
    return {
        "schema": SCHEMA,
        "source_revision": j["experiment_source_revision"],
        "measured_cases": j["measured_cases"],
        "source_preflight_calls_excluded":
            j["original_source_preflight_compiler_calls_excluded"],
        "all_compiled_source_results_matched": True,
        "report_sha256": expected["report.json"],
        "new_vendor_calls": 0,
        "hardware_latency": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--study")
    parser.add_argument("--quick", action="store_true",
                        help="Small smoke cohort; NOT a publication benchmark")
    args = parser.parse_args(argv)
    if args.study:
        result = audit(args.study)
    else:
        destination = Path(args.output)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("C2 cost output exists: original study immutable")
        revision = _git_revision()
        opts = ({"cohorts": COHORTS[:1], "seeds": (111,),
                 "request_counts": (1, 4), "streams": STREAMS}
                if args.quick else {})
        try:
            result = run(**opts)
            create_only(result, output=destination, source_revision=revision)
        except Exception as exc:
            # A negative or incomplete original CPU experiment is evidence.
            # Don't rerun with changed queries to fabricate a positive result.
            if not destination.exists() and not destination.is_symlink():
                destination.mkdir(parents=True, exist_ok=False)
            if not (destination / "report.json").exists():
                (destination / "FAILED.json").write_text(json.dumps({
                    "schema": SCHEMA + "/failed-source-attempt",
                    "experiment_source_revision": revision,
                    "quick_smoke_only": args.quick,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "native_vitis_calls": 0, "native_vivado_calls": 0,
                    "scientific_status": "FAILED_PRESERVE_COUNTEREXAMPLE",
                }, indent=2, sort_keys=True) + "\n")
            raise
        result = {
            "schema": SCHEMA, "source_revision": revision,
            "measured_cases": result["measured_cases"],
            "all_compiled_source_results_matched":
                result["all_compiled_source_results_matched"],
            "preflight_calls_excluded":
                result["original_source_preflight_compiler_calls_excluded"],
            "native_vendor_calls": 0,
            "hardware_latency": None,
            "output": args.output,
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
