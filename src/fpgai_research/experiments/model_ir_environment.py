"""Interactive compiler-backed architecture-search environment for research RL.

reset()/step(action) interact with the ACTUAL FPGAI Functional -> Architectural
-> Lowered Model IR -> HLS source compiler. Synthesis is NEVER run implicitly.

Two strict reward modes:
* "proxy" = hand-designed analytic source-only pilot score; NOT FPGA QoR.
* "hls" = only numerically accepted, source-attested Vitis HLS evidence from
  separately executed native cases; an unmeasured action returns PENDING and
  reward=None, WITHOUT changing the state or pretending to measure anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from fpgai_research.experiments import model_ir_search as search
from fpgai_research.experiments.model_ir_drl import encode_state
from fpgai_research.model_ir.semantics import functional_fingerprint
from fpgai_research.experiments.results import summarize_case


SCHEMA = "fpgai-research/model-ir-interactive-compiler-environment/v1"
HLS_RESOURCE_KEYS = ("lut", "ff", "dsp", "bram_18k", "uram")
HLS_LATENCY_KEY = "latency_worst"


def _sha(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Missing, linked or oversized original source manifest")
    return sha256(p.read_bytes()).hexdigest()


def _read_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


class NativeHlsReceiptOracle:
    """Read-only accepted HLS E3 receipts bound to exact source and CSim.

    This class does not run Vitis, edit native outputs or manufacture estimates.
    An E3 report without source-matched, successful fixed-point CSim is excluded.
    """

    def __init__(self, stage_directory):
        stage = Path(stage_directory)
        if stage.is_symlink() or not stage.is_dir():
            raise ValueError("Missing or linked native HLS stage")
        doc = _read_json(stage / "stage.json")
        if doc.get("schema") != "fpgai-research/heldout-native-hls-stage/v1":
            raise ValueError("Unrecognized HLS study source provenance")
        cases = doc.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("Empty staged native HLS inventory")
        self.stage = stage
        self.target_part = doc["target_part"]
        self.clock_ns = doc["clock_ns"]
        self.index = {}
        self.pending = {}
        self.negative = {}
        for case in cases:
            name = case["case"]
            directory = stage / name
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("Unsafe native case directory")
            if _sha(directory / "model.cpp") != case["cpp_sha256"]:
                raise ValueError("Native HLS input source has changed")
            summary = summarize_case(str(directory))
            if summary["source_sha256"] != case["cpp_sha256"]:
                raise ValueError("Native evidence disagrees with generated source")
            key = (case["heldout_model_sha256"], case["cpp_sha256"])
            if key in self.index or key in self.pending or key in self.negative:
                raise ValueError("Duplicate HLS source and model identity")
            entry = {
                "case": name,
                "model_sha256": case["heldout_model_sha256"],
                "cpp_sha256": case["cpp_sha256"],
                "evidence": summary["evidence"],
                "csim_passed": summary["csim_passed"],
                "warnings": summary.get("warnings", []),
            }
            if (summary["evidence"] == "E3-HLS-synthesis-report"
                    and summary["csim_passed"] is True):
                metrics = summary["metrics"]
                values = {}
                for name in (HLS_LATENCY_KEY, *HLS_RESOURCE_KEYS):
                    raw = metrics.get(name)
                    if raw is not None:
                        if (type(raw) is int and raw >= 0):
                            values[name] = raw
                        elif isinstance(raw, str) and raw.isdecimal():
                            values[name] = int(raw)
                        else:
                            raise ValueError("Invalid native HLS metric: " + name)
                if (HLS_LATENCY_KEY not in values
                        or values[HLS_LATENCY_KEY] <= 0
                        or any(name not in values for name in HLS_RESOURCE_KEYS)):
                    self.negative[key] = {
                        **entry, "reason": "INCOMPLETE_NATIVE_OBJECTIVE_VECTOR"
                    }
                else:
                    entry.update(
                        metrics=values,
                        report_sha256=summary["hls_report_sha256"],
                        source_attested_numeric_qualified=True,
                    )
                    self.index[key] = entry
            elif summary["csim_passed"] is False:
                self.negative[key] = {**entry, "reason": "FAILED_NUMERICAL_CSIM"}
            else:
                self.pending[key] = {**entry, "reason": "NATIVE_E3_NOT_QUALIFIED"}

    def lookup(self, model_sha, cpp_sha):
        key = (model_sha, cpp_sha)
        return self.index.get(key), self.negative.get(key), self.pending.get(key)


@dataclass(frozen=True)
class RewardSettings:
    """Predeclared source/HLS objective normalization, never fitted on test."""
    latency_reference: float = 256.0
    area_reference: float = 512.0
    hls_latency_reference_cycles: float = 1000.0
    # Explicit unit-specific denominators; not calibrated to physical power.
    hls_resource_references: tuple[float, ...] = (1000.0, 5000.0, 100.0, 100.0, 100.0)
    invalid_penalty: float = -0.1
    repeated_penalty: float = -0.025

    def __post_init__(self):
        from math import isfinite
        metrics = (self.latency_reference, self.area_reference,
                   self.hls_latency_reference_cycles,
                   *self.hls_resource_references)
        if (len(self.hls_resource_references) != len(HLS_RESOURCE_KEYS)
                or any(not isinstance(x, (int, float))
                       or not isfinite(x) or x <= 0 for x in metrics)):
            raise ValueError(
                "Every unit-specific reference must be finite and positive")


class ModelIRCompilerEnv:
    """Gymnasium-shaped, zero-dependency RL API using the real compiler.

    Observation keys:
      node_features, adjacency, action_mask, candidate,
      functional_sha256, compiler_cpp_sha256, best_score,
      objective_evidence, evaluations, proposal_count.

    step returns (observation, reward, terminated, truncated, info).
    'reward=None' means NO measured objective exists, and the candidate
    transition is suspended; the agent must obtain native evidence first.
    """

    def __init__(self, model, *, feature_mode="model_ir", reward_mode="proxy",
                 native_oracle=None, max_parallel_units=8,
                 max_evaluations=30, max_proposals=300,
                 mask_illegal=True, mask_repeated=True,
                 settings=None):
        if feature_mode not in ("raw", "graph", "model_ir"):
            raise ValueError("Unsupported feature ablation")
        if reward_mode not in ("proxy", "hls"):
            raise ValueError("Unknown reward evidence level")
        if reward_mode == "hls" and native_oracle is None:
            raise ValueError("HLS reward requires a real verified native receipt oracle")
        if type(max_parallel_units) is not int or max_parallel_units < 1:
            raise ValueError("Invalid target structural constraint")
        if (type(max_evaluations) is not int or not 2 <= max_evaluations <= 500
                or type(max_proposals) is not int or max_proposals < 1):
            raise ValueError("Invalid finite episode budget")
        if not 1 <= len(model.operators) <= 4 or any(
                op.kind not in ("MatMul", "Gemm") for op in model.operators):
            raise ValueError("Research environment currently supports 1..4 MatMul/Gemm operators")
        self.model = model
        self.functional_sha256 = functional_fingerprint(model)
        self.feature_mode = feature_mode
        self.reward_mode = reward_mode
        self.native_oracle = native_oracle
        self.max_parallel_units = max_parallel_units
        self.max_evaluations = max_evaluations
        self.max_proposals = max_proposals
        self.mask_illegal = mask_illegal
        self.mask_repeated = mask_repeated
        self.settings = settings or RewardSettings()
        self.actions = search.actions(len(model.operators))
        self.initialized = False

    def _cost(self, result):
        if self.reward_mode == "proxy":
            return (result.latency_proxy / self.settings.latency_reference
                    + result.area_proxy / self.settings.area_reference), {
                        "evidence": "source-only-hand-designed-analytic-proxy",
                        "latency_proxy": result.latency_proxy,
                        "area_proxy": result.area_proxy,
                    }, None
        entry, negative, pending = self.native_oracle.lookup(
            result.functional_sha256, result.cpp_sha256)
        if entry is None:
            return None, None, negative or pending or {
                "reason": "NATIVE_MEASUREMENT_NOT_STAGED",
                "cpp_sha256": result.cpp_sha256,
            }
        values = entry["metrics"]
        resources = sum(
            values[name] / reference for name, reference in zip(
                HLS_RESOURCE_KEYS, self.settings.hls_resource_references)
        )
        return (values[HLS_LATENCY_KEY] / self.settings.hls_latency_reference_cycles
                + resources), {
                    "evidence": "numerically-accepted-source-attested-HLS-E3",
                    "estimated_latency_cycles": values[HLS_LATENCY_KEY],
                    "estimated_resources": {
                        name: values[name] for name in HLS_RESOURCE_KEYS
                    },
                    "report_sha256": entry["report_sha256"],
                    "case": entry["case"],
                }, None

    def action_mask(self):
        if not self.initialized:
            raise RuntimeError("Call reset() before consulting the environment")
        valid = []
        for action in self.actions:
            next_candidate = search.apply(self.current, action)
            if next_candidate is None:
                valid.append(False)
            elif self.mask_illegal and not search.legal(
                    self.model, next_candidate,
                    max_parallel_units=self.max_parallel_units):
                valid.append(False)
            elif self.mask_repeated and next_candidate in self.observed:
                valid.append(False)
            else:
                valid.append(True)
        return np.array(valid, dtype=np.bool_)

    def _observation(self):
        features, adjacency = encode_state(
            self.model, self.current, self.feature_mode,
            max_parallel_units=self.max_parallel_units)
        # The agent also needs episode memory and incumbent quality.
        # These four features are shared across all raw/graph/Model IR
        # ablations. They capture search history, not hidden HLS estimates.
        incumbent = (0.0 if self.best_score is None else
                     float(self.best_score) / (1.0 + abs(self.best_score)))
        history = np.array([
            self.evaluations / self.max_evaluations,
            self.proposals / self.max_proposals,
            len(self.observed) / self.max_evaluations,
            incumbent,
        ], dtype=np.float32)
        full_features = np.concatenate([
            features, np.broadcast_to(history, (len(features), len(history)))
        ], axis=1)
        return {
            "node_features": full_features.copy(),
            "adjacency": adjacency.copy(),
            "action_mask": self.action_mask(),
            "candidate": tuple(tuple(row) for row in self.current),
            "functional_sha256": self.functional_sha256,
            "compiler_cpp_sha256": self.observed[self.current]["source"].cpp_sha256,
            "best_score": self.best_score,
            "objective_evidence": (
                "E0/E1-analytic-proxy" if self.reward_mode == "proxy"
                else "E3-measured-Vitis-HLS-if-available"),
            "evaluations": self.evaluations,
            "proposal_count": self.proposals,
        }

    def reset(self, *, seed=None):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("Expected nonnegative 32-bit episode seed")
        self.current = search.baseline(len(self.model.operators))
        initial = search.evaluate_source(
            self.model, self.current,
            max_parallel_units=self.max_parallel_units)
        if initial.functional_sha256 != self.functional_sha256:
            raise AssertionError("Functional Model IR changed at reset")
        score, evidence, missing = self._cost(initial)
        self.observed = {
            self.current: {
                "source": initial, "score": score,
                "evidence": evidence,
            }
        }
        self.best_candidate = self.current
        self.best_score = score
        self.evaluations = 1  # baseline also genuinely emitted HLS source
        self.proposals = 0
        self.terminated = False
        self.initialized = True
        return self._observation(), {
            "schema": SCHEMA, "reset": True,
            "model_sha256": self.functional_sha256,
            "baseline_source_sha256": initial.cpp_sha256,
            "baseline_evidence": evidence,
            "baseline_missing_hls": missing,
        }

    def step(self, action_index):
        if not self.initialized:
            raise RuntimeError("Call reset() before a compiler transition")
        if self.terminated:
            raise RuntimeError("Episode ended; reset before another action")
        if type(action_index) is not int or not 0 <= action_index < len(self.actions):
            raise ValueError("Action must be an index in the compiler action space")
        self.proposals += 1
        action = self.actions[action_index]
        proposal = search.apply(self.current, action)
        info = {
            "schema": SCHEMA, "action": action, "proposal_index": self.proposals,
            "previous_candidate": self.current,
            "candidate": proposal,
            "compiler_stages": (
                "Functional Model IR", "Architectural Model IR",
                "Lowered Model IR", "HLS C++",
            ),
            "vendor_tool_executed": False,
            "reward_mode": self.reward_mode,
        }
        if proposal is None or not search.legal(
                self.model, proposal,
                max_parallel_units=self.max_parallel_units):
            reward = self.settings.invalid_penalty
            info.update(status="ILLEGAL_ARCHITECTURE", compiler_rejected=True)
        elif proposal in self.observed:
            reward = self.settings.repeated_penalty
            if self.mask_repeated:
                info.update(status="REPEATED_CANDIDATE", cache_hit=True)
            else:
                # A graph search agent MUST be able to traverse an already
                # evaluated state to reach new architecture neighbors.
                # Reuse the original compiler evidence, never charge a
                # second source compilation or fabricate a fresh score.
                self.current = proposal
                info.update(status="CACHED_ARCHITECTURE", cache_hit=True,
                            reusing_evidence=True,
                            cpp_sha256=self.observed[proposal]["source"].cpp_sha256)
        else:
            # THIS is the real interactive compiler call, not a fabricated
            # source estimate or a graph-only synthetic environment.
            compiled = search.evaluate_source(
                self.model, proposal, max_parallel_units=self.max_parallel_units)
            if compiled.functional_sha256 != self.functional_sha256:
                raise AssertionError("Compiler transformation changed the model")
            score, evidence, missing = self._cost(compiled)
            if missing is not None:
                # A genuinely missing HLS result may never be rewarded using
                # a proxy or guessed resource/latency number.
                info.update(
                    status="MEASUREMENT_REQUIRED",
                    pending_cpp_sha256=compiled.cpp_sha256,
                    pending_source_mapping_sha256=compiled.source_signature,
                    evidence_status=missing,
                )
                return self._observation(), None, False, True, info
            previous_best = self.best_score
            self.current = proposal
            self.observed[proposal] = {
                "source": compiled, "score": score, "evidence": evidence
            }
            self.evaluations += 1
            if self.best_score is None or score < self.best_score:
                self.best_score = score
                self.best_candidate = proposal
            # Dense, bounded improvement-only reward: does not confuse a
            # worse intermediate architecture with hardware achievement.
            reward = (max(0.0, previous_best - self.best_score)
                      if previous_best is not None else 0.0)
            info.update(
                status="COMPILER_ACCEPTED",
                cpp_sha256=compiled.cpp_sha256,
                mapping_sha256=compiled.source_signature,
                functional_sha256=compiled.functional_sha256,
                evidence=evidence,
                score=score,
                best_score=self.best_score,
                unique_costed_evaluation=self.evaluations,
            )
        self.terminated = bool(
            self.evaluations >= self.max_evaluations or
            self.proposals >= self.max_proposals or
            not self.action_mask().any()
        )
        return self._observation(), float(reward), self.terminated, False, info
