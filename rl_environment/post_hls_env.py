"""One online HLS environment step == one *original* source-attested Vitis run.

No surrogate, no Vivado, no cached fake metrics. reset and plan are read-only
with respect to licensed tools; only explicit execute_native=True may invoke
Vitis. An attempted case is immutable even on tool failure or interruption.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import numpy as np

from fpgai_research.experiments.architecture_knobs_v2 import (
    build_catalog, compile_candidate, legal_candidate,
)
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package, make_testbench
from fpgai_research.toolchains.vitis_hls.runner import run_vitis
from fpgai_research.experiments.results import summarize_case

from .tasks import Task, RESOURCES
from .clock_actions import validate_clock_action
from .timing import audited_hls_timing
from .reward import (qualified_latency_ns, latency_only_reward,
                     beginner_real_hls_latency_reward,
                     curriculum_real_hls_cycle_reward)

SCHEMA="fpgai-research/online-post-hls-env/v1"


def _new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _receipt(case, task):
    """Independently re-audit the original native source, numerical and E3 case."""
    summary=summarize_case(str(case))
    prepared=json.loads((case/"manifest.json").read_text())
    action=json.loads((case/"action.json").read_text())
    # Independently reconstitute the CSim cohort from the immutable task.
    # A manifest hash and tool log alone cannot establish that the staged
    # testbench actually used this task's numeric inputs/precision/tolerance.
    expected_testbench=make_testbench(
        task.model,task.samples,
        int(prepared["precision_bits"]),int(prepared["integer_bits"]),
        tolerance=task.max_absolute_error)
    expected_testbench_sha=sha256(expected_testbench.encode()).hexdigest()
    if (action.get("task_sha")!=task.task_sha
            or action.get("model_sha")!=task.model_sha
            or action.get("sample_sha")!=task.sample_sha
            or action.get("cpp_sha256")!=prepared.get("cpp_sha256")
            or prepared.get("functional_sha256")!=task.model_sha
            or prepared.get("testbench_sha256")!=expected_testbench_sha
            or sha256((case/"tb.cpp").read_bytes()).hexdigest()!=expected_testbench_sha):
        raise ValueError("Native numerical cohort or source identity differs from task")
    vendor=case/"vitis_invocation.json"
    # Results must include an actual fresh, authenticated ORIGINAL Vitis run.
    native=json.loads(vendor.read_text()) if vendor.is_file() else {}
    requested_clock = validate_clock_action(task, action["requested_clock_ns"])
    source_valid = (
        summary.get("source_sha256") ==
            json.loads((case/"manifest.json").read_text())["cpp_sha256"]
        and summary.get("evidence")=="E3-HLS-synthesis-report"
        and summary.get("csim_passed") is True
        and native.get("completed_successfully") is True
        and native.get("tool")=="vitis_hls"
        and summary.get("target_part")==task.part
        and float(summary.get("clock_ns"))==requested_clock
        and requested_clock == action["requested_clock_ns"]
        and float(summary.get("numerical_tolerance"))<=task.max_absolute_error
        and (task.curriculum_stage is None or
             prepared.get("explicit_clock_uncertainty_ns") ==
             round(requested_clock * .27, 2))
        and (task.curriculum_stage is not None or not task.beginner_pilot or
             prepared.get("explicit_clock_uncertainty_ns") == 8.64)
    )
    gate_failures = []
    if not source_valid:
        gate_failures.append("SOURCE_OR_NATIVE_EVIDENCE")
    approved = source_valid
    numeric=summary.get("csim_receipt") or {}
    error=numeric.get("max_absolute_error")
    if summary.get("csim_passed") is not True:
        gate_failures.append("CSIM_NOT_VERIFIED_OR_FAILED")
    if not (type(error) in (float,int) and np.isfinite(error)
            and error<=task.max_absolute_error):
        approved=False
        gate_failures.append("NUMERICAL_ERROR_MISSING_OR_EXCEEDS_TOLERANCE")
    metrics=summary.get("metrics") or {}
    cycles=metrics.get("latency_worst")
    if not (type(cycles) is int or
            isinstance(cycles,str) and cycles.isdecimal()):
        approved=False
        gate_failures.append("HLS_WORST_CYCLES_MISSING_OR_NONINTEGER")
        cycles=None
    else:
        cycles=int(cycles)
        if cycles<=0:
            approved=False
            gate_failures.append("HLS_WORST_CYCLES_NONPOSITIVE")
    measured={}
    for field,limit in task.budgets.items():
        amount=metrics.get(field)
        if not (type(amount) is int or
                isinstance(amount,str) and amount.isdecimal()):
            approved=False
            gate_failures.append("MISSING_RESOURCE_" + field.upper())
            measured[field]=None
        else:
            measured[field]=int(amount)
            if measured[field]>limit:
                approved=False
                gate_failures.append("RESOURCE_EXCEEDS_LIMIT_" + field.upper())
    timing = None
    timing_parse_issue = None
    try:
        if approved:
            timing = audited_hls_timing(case, native, summary, requested_clock)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        timing_parse_issue = type(exc).__name__ + ": " + str(exc)
        approved = False
        gate_failures.append("ORIGINAL_HLS_TIMING_REPORT_INVALID")
    if timing is None or not timing["hls_timing_feasible"]:
        approved = False
        if timing is not None:
            gate_failures.append("HLS_SCHEDULING_TIMING_INFEASIBLE")
    latency_ns = qualified_latency_ns(cycles, requested_clock, feasible=bool(approved))
    duration=native.get("duration_seconds")
    if type(duration) not in (float,int) or not np.isfinite(duration) or duration<0:
        duration=None
    return {
        "qualified":bool(approved),
        "reason":"NUMERIC_RESOURCE_HLS_TIMING_ACCEPTED" if approved
                 else "FAILED_OR_MISSING_CSIM_RESOURCE_HLS_TIMING_EVIDENCE",
        "qualification_gates_failed": gate_failures,
        "source_and_original_native_evidence_valid": source_valid,
        "csim_passed": summary.get("csim_passed"),
        "raw_hls_worst_case_cycles": cycles,
        "original_hls_timing_parse_issue": timing_parse_issue,
        "hls_summary_warnings": list(summary.get("warnings", [])),
        "hls_worst_case_cycles":cycles if approved else None,
        "hls_estimated_resources":measured,
        "csim_max_absolute_error":error,
        "hls_report_sha256":summary.get("hls_report_sha256"),
        "vendor_seconds":duration,
        "vitis_invocation_sha256":sha256(vendor.read_bytes()).hexdigest()
            if vendor.is_file() else None,
        "requested_clock_ns":requested_clock,
        "proposed_target_frequency_mhz":1000.0/requested_clock,
        "hls_qualified_requested_frequency_mhz":(
            1000.0/requested_clock if approved else None),
        "hls_estimated_latency_ns":latency_ns,
        "hls_estimated_clock_ns":timing["hls_estimated_clock_ns"] if timing else None,
        "hls_clock_uncertainty_ns":timing["hls_clock_uncertainty_ns"] if timing else None,
        "hls_effective_scheduling_budget_ns":(
            timing["hls_effective_scheduling_budget_ns"] if timing else None),
        "hls_scheduling_margin_ns":(
            timing["hls_scheduling_margin_ns"] if timing else None),
        "hls_timing_feasible":timing["hls_timing_feasible"] if timing else None,
        "achieved_fmax_mhz":None,
        "routed_latency_ns":None,
        "board_qualified":False,
        "evidence":summary.get("evidence"),
    }


class PostHLSEnv:
    def __init__(self,task:Task,root,*,seed=101):
        self.task=task
        self.root=Path(root)
        if self.root.exists() or self.root.is_symlink():
            raise FileExistsError("Native episode exists; never overwrite or re-run it")
        self.root.mkdir(parents=True)
        self.catalog=build_catalog(task.model)
        self.freeze_numeric_format = task.freeze_numeric_format
        self.beginner_pilot = task.beginner_pilot
        from .curriculum import get_stage
        self.stage = get_stage(task.curriculum_stage)
        self.beginner_fields = frozenset(("parallelism", "pes", "simd", "serialization"))
        self.fixed_numeric_indices = tuple(row[:2] for row in self.catalog.baseline)
        self.seed=seed
        self.tried=set()
        self.tried_configs=set()
        self.current=None
        self.calls=0
        self.best_cycles=None
        self.best_latency_ns=None
        self.max_observed_hls_qualified_requested_mhz=None
        self.best_candidate=None
        self.last_result=None
        self.history=[]
        self.done=False
        _new_json(self.root/"task.json",{
            "schema":SCHEMA,
            "task":task.public_descriptor(),
            "seed":seed,
            "source_only_on_initialization":True,
            "original_vendor_calls":0,
        })

    def candidates(self,max_candidates=12):
        """Compiler-legal mix of global proposals and local refinements.

        Do not force a full original HLS evaluation of the baseline first.
        Candidate generation itself is deterministic per episode state and
        invokes no tool. Every offered architecture is compiler-validated.
        """
        if self.done:
            return []
        base = self.catalog.baseline if self.current is None else self.current
        rng = Random(self.seed + 10007 * self.calls + len(self.tried) * 131)
        actions = [a for a in self.catalog.actions
                   if ((not self.freeze_numeric_format or a[1] not in
                        ("precision_bits", "integer_bits"))
                       and (not self.beginner_pilot or a[1] in self.beginner_fields)
                       and (self.stage is None or
                            a[1] in self.stage.knob_fields))]
        rng.shuffle(actions)
        options = []
        seen = set(self.tried)

        def add(candidate):
            if (candidate is not None and candidate not in seen
                    and (not self.freeze_numeric_format or
                         all(row[:2] == self.fixed_numeric_indices[i]
                             for i, row in enumerate(candidate)))
                    and (not self.beginner_pilot or
                         all(all(row[j] == self.catalog.baseline[i][j]
                                 for j in (0, 1, 6, 7))
                             for i, row in enumerate(candidate)))
                    and (self.stage is None or all(
                        all(row[j] == self.catalog.baseline[i][j]
                            for j, key in enumerate(
                                ("precision_bits", "integer_bits",
                                 "parallelism", "pes", "simd",
                                 "serialization", "tile", "pipeline_ii"))
                            if key not in self.stage.knob_fields)
                        for i, row in enumerate(candidate)))
                    and legal_candidate(self.task.model, self.catalog, candidate)):
                seen.add(candidate)
                options.append(candidate)

        # Baseline remains a selectable *control*, not a required initial call.
        add(self.catalog.baseline)
        # Global exploration from varied legal random walks on the baseline;
        # uses the true compiler legality oracle, never a proxy QoR estimate.
        attempts = 0
        # Must be prefix-stable between Gym max_candidates=12 and backend
        # authorization max_candidates=128; otherwise the actor can select
        # a legal *offered* architecture rejected by the backend.
        global_target = min(6, max_candidates)
        while len(options) < global_target and attempts < 960:
            attempts += 1
            candidate = self.catalog.baseline
            for _ in range(rng.randint(2, 14)):
                trial = self.catalog.mutate(candidate, rng.choice(actions))
                if trial is not None and legal_candidate(
                        self.task.model, self.catalog, trial):
                    candidate = trial
            add(candidate)
        # Local actions from current/best state are still provided after each
        # HLS observation, enabling exploitation with the same shared policy.
        local_anchors = [base]
        if self.best_candidate is not None:
            local_anchors.append(self.best_candidate[0])
        for anchor in local_anchors:
            for action in actions:
                add(self.catalog.mutate(anchor, action))
                if len(options) >= max_candidates:
                    return options[:max_candidates]
        # The first 12 candidates must remain identical independent of the
        # caller's larger authorization limit.
        while len(options) < max_candidates and attempts < 20480:
            attempts += 1
            candidate = rng.choice(local_anchors)
            for _ in range(rng.randint(1, 7)):
                trial = self.catalog.mutate(candidate, rng.choice(actions))
                if trial is not None and legal_candidate(
                        self.task.model, self.catalog, trial):
                    candidate = trial
            add(candidate)
        return options[:max_candidates]

    def observe(self):
        return {
            "schema":SCHEMA,"task":self.task.public_descriptor(),
            "best_qualified_hls_cycles":self.best_cycles,
            "best_qualified_hls_estimated_ns":self.best_latency_ns,
            "max_observed_hls_qualified_requested_mhz":
                self.max_observed_hls_qualified_requested_mhz,
            "last_original_hls_result":self.last_result,
            "calls_consumed":self.calls,
            "calls_remaining":self.task.max_vitis_calls-self.calls,
            "candidate_count":len(self.candidates()) if not self.done else 0,
            "done":self.done,"real_hls_results_only":True,
        }

    def step(self,candidate,*,clock_ns=None,execute_native=False,
             vitis_executable="vitis_hls",timeout_seconds=7200):
        if self.done:
            raise RuntimeError("Episode already complete")
        if clock_ns is None and self.task.clock_ns is None:
            raise ValueError("Policy must output the requested floating clock")
        period = validate_clock_action(
            self.task, self.task.clock_ns if clock_ns is None else clock_ns)
        key = (candidate, period)
        if key in self.tried_configs:
            raise ValueError("Repeated architecture-clock pair is not a fresh native case")
        offered = self.candidates(max_candidates=128)
        if candidate not in offered and candidate not in (
                self.current, self.catalog.baseline,
                self.best_candidate[0] if self.best_candidate else None):
            raise ValueError("Choose a compiler-legal offered configuration")
        if not legal_candidate(self.task.model, self.catalog, candidate):
            raise ValueError("Architecture is not compiler legal")
        if self.freeze_numeric_format and any(
                row[:2] != self.fixed_numeric_indices[i]
                for i, row in enumerate(candidate)):
            raise ValueError("Frozen numeric format architecture cannot vary precision/integer bits")
        if self.beginner_pilot and any(
                any(row[j] != self.catalog.baseline[i][j]
                    for j in (0, 1, 6, 7))
                for i, row in enumerate(candidate)):
            raise ValueError("Beginner architecture cannot vary precision, tile or pipeline II")
        if self.stage is not None:
            from fpgai_research.experiments.architecture_knobs_v2 import FIELDS
            if any(any(row[j] != self.catalog.baseline[i][j]
                       for j, key in enumerate(FIELDS)
                       if key not in self.stage.knob_fields)
                   for i, row in enumerate(candidate)):
                raise ValueError("Architecture mutated a curriculum-locked knob")
        if not execute_native:
            return {
                "status":"DRY_RUN_NO_VENDOR_EXECUTION",
                "candidate":[list(x) for x in candidate],
                "requested_clock_ns":period,
                "requires_explicit_native_opt_in":True,
                "native_vendor_calls_executed":0,
            }
        case=self.root/f"case_{self.calls:04d}"
        if case.exists() or case.is_symlink():
            raise FileExistsError("Immutable case already attempted")
        emitted,_,_=compile_candidate(self.task.model,self.catalog,candidate)
        write_vitis_package(
            str(case),self.task.model,emitted,self.task.samples,
            part=self.task.part,clock_ns=period,
            tolerance=self.task.max_absolute_error,
            clock_uncertainty_ns=(
                round(period * .27, 2) if self.stage is not None
                else (8.64 if self.beginner_pilot else None)))
        _new_json(case/"action.json",{
            "task_sha":self.task.task_sha,
            "model_sha":self.task.model_sha,
            "sample_sha":self.task.sample_sha,
            "choices":[list(row) for row in candidate],
            "requested_clock_ns":period,
            "cpp_sha256":emitted.manifest["cpp_sha256"],
        })
        self.calls+=1
        self.tried.add(candidate)
        self.tried_configs.add(key)
        try:
            run_vitis(case,executable=vitis_executable,
                      timeout_seconds=timeout_seconds)
        except BaseException:
            # Source, logs, action and native invocation (if present) MUST
            # remain intact. The same case is never rerun after a crash.
            self.done=True
            raise
        accepted=_receipt(case,self.task)
        self.last_result=accepted
        self.history.append((candidate, period, accepted))
        previous=self.best_latency_ns
        if accepted["qualified"]:
            qualified_mhz=accepted["hls_qualified_requested_frequency_mhz"]
            if (self.max_observed_hls_qualified_requested_mhz is None
                    or qualified_mhz > self.max_observed_hls_qualified_requested_mhz):
                self.max_observed_hls_qualified_requested_mhz=qualified_mhz
            latency=accepted["hls_estimated_latency_ns"]
            if self.best_latency_ns is None or latency<self.best_latency_ns:
                self.best_latency_ns=latency
                self.best_cycles=accepted["hls_worst_case_cycles"]
                self.best_candidate=(candidate,period)
        self.current=candidate
        self.done=self.calls>=self.task.max_vitis_calls
        # Do not fabricate missing latency/area in the reward. A rejection
        # receives feasibility/cost feedback, not invented HLS cycles.
        if self.stage is not None:
            reward = curriculum_real_hls_cycle_reward(
                qualified=accepted["qualified"],
                hls_cycles=accepted["hls_worst_case_cycles"],
                model=self.task.model,
                requested_clock_ns=clock_ns)
        elif self.beginner_pilot:
            reward = beginner_real_hls_latency_reward(
                qualified=accepted["qualified"],
                hls_cycles=accepted["hls_worst_case_cycles"],
                input_width=self.task.model.tensor(
                    self.task.model.inputs[0]).shape[-1],
                output_width=self.task.model.tensor(
                    self.task.model.outputs[0]).shape[-1])
        else:
            reward=latency_only_reward(
                previous, self.best_latency_ns, qualified=accepted["qualified"])
        outcome={
            "status":"AUDITED_POST_HLS_STEP",
            "task_sha":self.task.task_sha,
            "candidate":[list(x) for x in candidate],
            "result":accepted,
            "reward":reward,
            "reward_scope":(
                "CURRICULUM_REAL_HLS_NS_EFFICIENCY_PER_FEASIBLE_CANDIDATE"
                if self.stage is not None
                else ("BEGINNER_REAL_HLS_CYCLE_EFFICIENCY_PER_FEASIBLE_CANDIDATE"
                      if self.beginner_pilot
                      else "ONLINE_REAL_HLS_TIMING_QUALIFIED_LATENCY_NS")),
            "best_cycles":self.best_cycles,
            "best_hls_estimated_latency_ns":self.best_latency_ns,
            "max_observed_hls_qualified_requested_mhz":
                self.max_observed_hls_qualified_requested_mhz,
            "requested_clock_ns":period,
            "native_calls_used":self.calls,
            "done":self.done,
        }
        _new_json(case/"online_step.json",outcome)
        return outcome
