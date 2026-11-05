"""HW-RL-03: Gymnasium wrapper around *original, independently audited* Vitis.

A step is one immutable full-network hardware architecture plus a continuous
requested clock, one NEW original Vitis HLS invocation, and a separate read-
only source/CSim/CSynth audit. Legacy post_hls_env, Gym agents and paper
cases are not modified. Reset/observation never invokes licensed tools.

Crucially the source-only action mask does NOT promise numerical, timing or
physical feasibility. No trained agent or physical FPGA speedup is claimed.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from math import isfinite, log1p, prod
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym, spaces = None, None

from ..infrastructure import hardware_native
from ..clock_actions import CLOCK_MIN_NS, CLOCK_MAX_NS, normalize_clock_period_ns
from .hardware_space import (
    HardwareEdit, HardwareState, RESOURCE_NAMES, action_mask,
    candidate_actions, graph_observation, initial_state, transition,
)
from ..reward import latency_only_reward

MAX_NODES = 256
MAX_EDGES = 512
MAX_ACTIONS = 512
NODE_FEATURES = 40
ACTION_FEATURES = 24
GLOBAL_FEATURES = 20
MAX_HISTORY = 30
HISTORY_FEATURES = 18
KIND_ORDER = (
    "Conv", "MatMul", "Gemm", "Relu", "MaxPool", "AveragePool",
    "Flatten", "Softmax", "Add", "Sub", "Mul", "Sigmoid", "Other",
)
SCHEMA = "fpgai-research/rl-hardware-gym-native/v1"


def _ln(value, divisor=16.):
    return float(log1p(float(value))) / divisor


def _clock(value) -> float:
    """Policy may output a float; Vitis 2023.2 attests target at 0.01 ns."""
    period = normalize_clock_period_ns(value, decimals=2)
    if not isfinite(period):
        raise ValueError("Invalid learned requested clock")
    return period


def _features_for_action(action: HardwareEdit | None) -> np.ndarray:
    row = np.zeros(ACTION_FEATURES, np.float32)
    if action is None:
        row[0] = 1.  # clock-only option
        return row
    row[{"clock": 1, "numeric": 2, "operator": 3}[action.scope]] = 1.
    row[4] = 0 if action.operator_index is None else (action.operator_index + 1) / MAX_NODES
    fields = ("precision_bits", "integer_bits", "parallelism",
              "pes", "simd", "serialization", "tile", "pipeline_ii")
    row[5 + fields.index(action.field)] = 1.
    if action.value is None:
        row[13] = 1.
    elif isinstance(action.value, (tuple, list)):
        row[14] = 1.
        for i, element in enumerate(action.value[:8]):
            row[15+i] = _ln(element) if type(element) is int and element >= 0 else 0.
    elif type(action.value) is int:
        row[23] = _ln(action.value) if action.value >= 0 else 0.
    else:
        raise TypeError("Invalid typed RL hardware action")
    return row


def _validated_native_result(result: dict, *, case: Path, state: HardwareState,
                             plan: dict) -> None:
    """Do not pay an RL reward for an unaudited or inconsistent result."""
    if not isinstance(result, dict):
        raise RuntimeError("Native audit did not return a typed source receipt")
    checks = (
        result.get("schema") == hardware_native.AUDIT_SCHEMA,
        result.get("state_sha256") == state.state_sha256,
        result.get("functional_sha256") == state.functional_sha256,
        result.get("source_cpp_sha256") == plan["cpp_sha256"],
        result.get("sample_sha256") == plan["sample_sha256"],
        result.get("case") == str(case.absolute()),
        result.get("requested_clock_ns") == state.clock_ns,
        result.get("native_calls_by_audit") == 0,
        result.get("native_invocation_present") is True,
        result.get("reward") is None,
        type(result.get("qualified")) is bool,
        type(result.get("gate_failures")) is list,
        result.get("board_qualified") is False,
    )
    if not all(checks):
        raise RuntimeError("Original Vitis HLS report lacks independent source/task attestation")
    # A failed/partial vendor invocation is an infrastructure error,
    # NOT a verified infeasible hardware design. All trainable negative
    # rewards must also be backed by full original E3 + passing CSim,
    # numeric metrics, real whole-top cycles/resources and parsed timing.
    measured = result.get("original_hls_resource_consumed")
    timing = result.get("hls_timing") or {}
    raw_cycles = result.get("raw_hls_worst_case_cycles")
    error = result.get("csim_max_absolute_error")
    if (
        result.get("original_vitis_completed_successfully") is not True
        or result.get("csim_passed") is not True
        or result.get("evidence") != "E3-HLS-synthesis-report"
        or type(raw_cycles) is not int or raw_cycles <= 0
        or not isinstance(measured, dict)
        or any(type(measured.get(name)) is not int or measured[name] < 0
               for name in RESOURCE_NAMES)
        or type(error) not in (float, int) or not isfinite(error) or error < 0
        or not isinstance(timing, dict)
        or type(timing.get("hls_timing_feasible")) is not bool
    ):
        raise RuntimeError("Incomplete original E3/CSim/native evidence cannot earn a learning reward")
    if result["qualified"]:
        cycles = result.get("hls_worst_case_cycles")
        latency = result.get("hls_estimated_latency_ns")
        timing = result.get("hls_timing") or {}
        if (
            result["gate_failures"]
            or result.get("original_vitis_completed_successfully") is not True
            or result.get("csim_passed") is not True
            or result.get("evidence") != "E3-HLS-synthesis-report"
            or type(cycles) is not int or cycles <= 0
            or type(latency) not in (float, int)
            or not isfinite(latency) or latency <= 0
            or abs(latency - cycles * state.clock_ns) > 1e-8
            or timing.get("hls_timing_feasible") is not True
        ):
            raise RuntimeError("Unqualified or contradictory original HLS timing cannot earn reward")
        if any(measured[name] > cap
               for name, cap in zip(RESOURCE_NAMES, state.resource_caps)):
            raise RuntimeError("Qualified HLS resource usage over budget")
        error = result.get("csim_max_absolute_error")
        if type(error) not in (int, float) or not isfinite(error) or error > state.max_absolute_error:
            raise RuntimeError("Qualified HLS numerical error not independently bounded")
    else:
        if (result.get("hls_estimated_latency_ns") is not None
                or result.get("hls_worst_case_cycles") is not None
                or not result["gate_failures"]):
            raise RuntimeError("Infeasible HLS action cannot have synthetic latency or empty failure list")
        if error <= state.max_absolute_error and all(
            measured[name] <= cap for name, cap in zip(RESOURCE_NAMES, state.resource_caps)
        ) and timing.get("hls_timing_feasible") is True:
            raise RuntimeError("All observed gates pass; cannot label this architecture infeasible")


class GymHardwareNativeEnv(gym.Env if gym is not None else object):
    """Policy-facing Gym. Steps execute ONLY with explicit native opt-in.

    Action: (index, continuous requested clock in nanoseconds).
    Index 0 edits only clock. Other indices edit one source-legal
    architecture request *and* optionally adjust clock on the same step.

    A source-legal candidate that fails original Vitis is a valid negative
    observation (reward -0.05); invalid source actions and contradictory
    auditor outputs fail before rewarding. No old case is overwritten.
    """
    metadata = {"render_modes": []}

    def __init__(self, model, samples: np.ndarray, root: str | Path, *,
                 part: str, resource_caps: tuple[int, int, int, int, int],
                 max_absolute_error: float, max_vitis_calls: int = 10,
                 initial_clock_ns: float = 14.0, max_parallel_units: int = 8,
                 vitis_executable: str = "vitis_hls",
                 timeout_seconds: int = 7200,
                 execute_native: bool = False,
                 fixed_clock_ns: float | None = None):
        if gym is None:
            raise RuntimeError("Install gymnasium>=1,<2 to use native hardware Gym")
        if type(execute_native) is not bool:
            raise ValueError("Native opt-in must be a typed boolean")
        if type(max_vitis_calls) is not int or not 1 <= max_vitis_calls <= 100:
            raise ValueError("Explicit 1..100 original Vitis calls required per episode")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 172800:
            raise ValueError("Explicit vendor timeout must be 1..172800 seconds")
        if not isinstance(root, (str, Path)):
            raise ValueError("Explicit experiment root path required")
        self.fixed_clock_ns = (
            _clock(fixed_clock_ns) if fixed_clock_ns is not None else None
        )
        if (self.fixed_clock_ns is not None
                and self.fixed_clock_ns != _clock(initial_clock_ns)):
            raise ValueError("Fixed clock must equal the original initial clock")
        self.root = Path(root).expanduser()
        if self.root.is_symlink():
            raise ValueError("Experiment root cannot be a symlink")
        from fpgai_research.architecture.resolver import TargetLimits
        self.initial = initial_state(
            model, part=part, clock_ns=_clock(initial_clock_ns),
            resource_caps=resource_caps, max_absolute_error=max_absolute_error,
            limits=TargetLimits(max_parallel_units=max_parallel_units),
        )
        # Validate the exact fixed original numerical cohort before the first
        # episode, without creating cases or launching any vendor process.
        self.samples = np.array(samples, copy=True)
        self.samples.setflags(write=False)
        self.source_plan = hardware_native.planned_case(self.initial, self.samples)
        graph = graph_observation(self.initial)
        if len(graph["nodes"]) > MAX_NODES or len(graph["edges"]) > MAX_EDGES:
            raise ValueError("Network graph exceeds explicitly supported Gym graph bounds")
        self.max_vitis_calls = max_vitis_calls
        self.vitis_executable = vitis_executable
        self.timeout_seconds = timeout_seconds
        self.execute_native = execute_native
        self.episode_index = 0
        self.current = None
        self.calls = 0
        self.best_latency_ns = None
        self.best_state_sha256 = None
        self.last_audit = None
        self.done = False
        self.visited = set()
        self.history: list[dict] = []
        self.offers: tuple[HardwareEdit | None, ...] = ()
        self._mask = np.zeros(MAX_ACTIONS, np.int8)
        self._episode_path = None
        self._episode_allocated = False

        self.clock_action_bounds_ns = (
            (self.fixed_clock_ns, self.fixed_clock_ns)
            if self.fixed_clock_ns is not None
            else (CLOCK_MIN_NS, CLOCK_MAX_NS)
        )
        self.action_space = spaces.Tuple((
            spaces.Discrete(MAX_ACTIONS),
            spaces.Box(low=self.clock_action_bounds_ns[0],
                       high=self.clock_action_bounds_ns[1],
                       shape=(), dtype=np.float64),
        ))
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(-1e6, 1e6, (MAX_NODES, NODE_FEATURES), np.float32),
            "node_mask": spaces.MultiBinary(MAX_NODES),
            "edge_index": spaces.Box(0, MAX_NODES - 1, (MAX_EDGES, 2), np.int32),
            "edge_mask": spaces.MultiBinary(MAX_EDGES),
            "action_features": spaces.Box(-1e6, 1e6, (MAX_ACTIONS, ACTION_FEATURES),
                                           np.float32),
            "action_mask": spaces.MultiBinary(MAX_ACTIONS),
            "action_target_node": spaces.Box(-1, MAX_NODES - 1,
                                              (MAX_ACTIONS,), np.int32),
            "history": spaces.Box(-1e6, 1e6, (MAX_HISTORY, HISTORY_FEATURES),
                                  np.float32),
            "history_mask": spaces.MultiBinary(MAX_HISTORY),
            "global": spaces.Box(-1e6, 1e6, (GLOBAL_FEATURES,), np.float32),
            "clock_bounds_ns": spaces.Box(CLOCK_MIN_NS, CLOCK_MAX_NS,
                                          (2,), np.float32),
        })

    def _observation(self) -> dict:
        state = self.current
        if state is None:
            raise RuntimeError("Gym reset required")
        graph = graph_observation(state)
        nodes = np.zeros((MAX_NODES, NODE_FEATURES), np.float32)
        node_mask = np.zeros(MAX_NODES, np.int8)
        for i, entry in enumerate(graph["nodes"]):
            row = nodes[i]
            node_mask[i] = 1
            row[0] = float(entry["type"] == "tensor")
            row[1] = float(entry["type"] == "operator")
            if entry["type"] == "tensor":
                row[2] = float(entry["constant"])
                shape = entry["shape"]
                row[12] = _ln(entry["element_count"])
            else:
                shape = entry["iteration_shape"]
                kind = entry["kind"]
                row[13 + (KIND_ORDER.index(kind) if kind in KIND_ORDER else 12)] = 1.
                req = entry["mapping"]
                for j, field in enumerate(("precision_bits", "integer_bits",
                                           "parallelism", "pes", "simd",
                                           "serialization")):
                    row[26+j] = _ln(req[field])
                row[32] = float(req["pipeline_ii"] is not None)
                row[33] = _ln(req["pipeline_ii"]) if req["pipeline_ii"] else 0.
                row[34] = float(bool(req["tile"]))
                row[35] = _ln(prod(req["tile"])) if req["tile"] else 0.
            row[3] = len(shape) / 16.
            for j, dim in enumerate(shape[:8]):
                row[4+j] = _ln(dim)
        edge_index = np.zeros((MAX_EDGES, 2), np.int32)
        edge_mask = np.zeros(MAX_EDGES, np.int8)
        for i, edge in enumerate(graph["edges"]):
            edge_index[i] = (edge["src"], edge["dst"])
            edge_mask[i] = 1

        # Every candidate is proposed by the original compiler, then source
        # checked; these masks indicate *source compilation only*.
        candidates = candidate_actions(state, max_actions=MAX_ACTIONS - 1)
        if len(candidates) >= MAX_ACTIONS:
            raise ValueError("Cannot expose a truncated action menu")
        source_mask = action_mask(state, candidates)["source_compilable"]
        self.offers = (None, *candidates)
        self._mask = np.zeros(MAX_ACTIONS, np.int8)
        action_features = np.zeros((MAX_ACTIONS, ACTION_FEATURES), np.float32)
        target_nodes = np.full(MAX_ACTIONS, -1, np.int32)
        # A fixed-clock experiment has no clock-only policy action.
        self._mask[0] = int(self.fixed_clock_ns is None)
        action_features[0] = _features_for_action(None)
        for i, (candidate, valid) in enumerate(zip(candidates, source_mask), 1):
            self._mask[i] = int(valid)
            action_features[i] = _features_for_action(candidate)
            if candidate.operator_index is not None:
                target_nodes[i] = len(state.model.tensors) + candidate.operator_index
        caps = list(state.resource_caps)
        previous = self.last_audit or {}
        timing = previous.get("hls_timing") or {}
        used = previous.get("original_hls_resource_consumed") or {}
        vals = [
            state.clock_ns / CLOCK_MAX_NS,
            _ln(state.limits.max_parallel_units),
            _ln(state.max_absolute_error),
            self.calls / self.max_vitis_calls,
            (self.max_vitis_calls-self.calls) / self.max_vitis_calls,
            float(self.best_latency_ns is not None),
            _ln(self.best_latency_ns) if self.best_latency_ns is not None else 0.,
            float(previous.get("qualified") is True),
            float(bool(previous)),
            _ln(previous["hls_estimated_latency_ns"])
                if previous.get("hls_estimated_latency_ns") is not None else 0.,
            float(previous.get("hls_estimated_latency_ns") is not None),
            _ln(timing["hls_estimated_clock_ns"])
                if timing.get("hls_estimated_clock_ns") is not None else 0.,
            float(timing.get("hls_estimated_clock_ns") is not None),
            _ln(previous["csim_max_absolute_error"])
                if previous.get("csim_max_absolute_error") is not None else 0.,
            float(previous.get("csim_max_absolute_error") is not None),
            *[_ln(value) for value in caps],
        ]
        assert len(vals) == GLOBAL_FEATURES
        history = np.zeros((MAX_HISTORY, HISTORY_FEATURES), np.float32)
        history_mask = np.zeros(MAX_HISTORY, np.int8)
        for i, row in enumerate(self.history[-MAX_HISTORY:]):
            report = row["audit"]
            measured = report.get("original_hls_resource_consumed") or {}
            timing = report.get("hls_timing") or {}
            latency = report.get("hls_estimated_latency_ns")
            cycles = report.get("raw_hls_worst_case_cycles")
            error = report.get("csim_max_absolute_error")
            history[i] = [
                row["action_index"] / MAX_ACTIONS,
                row["clock_ns"] / CLOCK_MAX_NS,
                float(report["qualified"]),
                float(row["reward"]),
                _ln(cycles) if cycles is not None else 0.,
                _ln(error) if error is not None else 0.,
                (float(timing["hls_estimated_clock_ns"]) / CLOCK_MAX_NS
                 if timing.get("hls_estimated_clock_ns") is not None else 0.),
                float(timing.get("hls_timing_feasible") is True),
                _ln(latency) if latency is not None else 0.,
                float(latency is not None),
                *[float(measured[name]) / max(1, cap)
                  if measured.get(name) is not None else 0.
                  for name, cap in zip(RESOURCE_NAMES, state.resource_caps)],
                float(report.get("evidence") == "E3-HLS-synthesis-report"),
                1.,  # original vendor result present
                _ln(row["best_ns"]) if row["best_ns"] is not None else 0.,
            ]
            history_mask[i] = 1
        return {
            "nodes": nodes, "node_mask": node_mask,
            "edge_index": edge_index, "edge_mask": edge_mask,
            "action_features": action_features, "action_mask": self._mask.copy(),
            "action_target_node": target_nodes,
            "history": history, "history_mask": history_mask,
            "global": np.asarray(vals, np.float32),
            "clock_bounds_ns": np.asarray(self.clock_action_bounds_ns,
                                          dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current = self.initial
        self.calls = 0
        self.best_latency_ns = None
        self.best_state_sha256 = None
        self.last_audit = None
        self.done = False
        self.visited = set()
        self.history = []
        self._episode_path = self.root / f"episode_{self.episode_index:06d}"
        self._episode_allocated = False
        if self._episode_path.exists() or self._episode_path.is_symlink():
            raise FileExistsError("Existing episode cannot be reused or overwritten")
        self.episode_index += 1
        observation = self._observation()
        return observation, {
            "schema": SCHEMA,
            "state_sha256": self.current.state_sha256,
            "source_actions": len(self.offers),
            "native_vitis_calls": 0,
            "reward": None,
            "source_mask_is_not_native_feasibility": True,
        }

    def preview_action(self, action) -> HardwareState:
        """Source-only admission before spending an original Vitis call.

        Return the exact complete compiler state, or reject a masked, no-op,
        previously synthesized or source-invalid proposal. This method is
        read-only: no case allocation, vendor call, reward, or history update.
        Both campaign admission and step() use this same implementation.
        """
        if self.current is None:
            raise RuntimeError("Gym reset required before step")
        if self.done:
            raise RuntimeError("Episode completed; no additional Vitis calls permitted")
        if (not isinstance(action, (tuple, list)) or len(action) != 2
                or type(action[0]) not in (int, np.int64, np.int32)):
            raise ValueError("Expected (source action index, floating requested clock)")
        idx = int(action[0])
        if not 0 <= idx < len(self.offers) or not self._mask[idx]:
            raise ValueError("Invalid/masked source architecture index")
        period = _clock(action[1])
        if self.fixed_clock_ns is not None and period != self.fixed_clock_ns:
            raise ValueError("Fixed-clock environment refuses requested clock changes")
        if not self.action_space.contains((idx, np.asarray(period, np.float64))):
            raise ValueError("Clock action outside Gym action-space bounds")

        state = self.current
        if idx:
            decision = transition(state, self.offers[idx])
            if decision.status != "SOURCE_ACCEPTED":
                raise ValueError("Source action rejected: " + str(decision.reason))
            state = decision.next
        if state.clock_ns != period:
            clock_change = transition(state, HardwareEdit("clock", "clock_ns", period))
            if clock_change.status != "SOURCE_ACCEPTED":
                raise ValueError("Clock action rejected: " + str(clock_change.reason))
            state = clock_change.next
        if state.state_sha256 == self.current.state_sha256:
            raise ValueError("No-op architecture-clock action; native case not started")
        if state.state_sha256 in self.visited:
            raise ValueError("Repeated complete hardware state; no new native case")
        return state

    def step(self, action):
        if not self.execute_native:
            raise RuntimeError("Native Vitis requires explicit execute_native=True")
        state = self.preview_action(action)
        idx, period = int(action[0]), _clock(action[1])
        source = hardware_native.planned_case(state, self.samples)
        case = self._episode_path / f"case_{self.calls:04d}"
        if case.exists() or case.is_symlink():
            raise FileExistsError("Native case exists; never overwrite")
        if not self._episode_allocated:
            if self._episode_path.exists() or self._episode_path.is_symlink():
                raise FileExistsError("Episode path has already been allocated")
            self._episode_path.mkdir(parents=True, exist_ok=False)
            self._episode_allocated = True
        elif self._episode_path.is_symlink() or not self._episode_path.is_dir():
            raise FileNotFoundError("Original episode directory is missing or unsafe")
        self.calls += 1
        self.visited.add(state.state_sha256)
        try:
            # Original Vitis executes once. Independently replay the stored
            # original generated code + CSim + synth reports without Vitis.
            first = hardware_native.execute_native_case(
                case, state, self.samples,
                vitis=self.vitis_executable, timeout_seconds=self.timeout_seconds,
            )
            audit = hardware_native.audit_native_case(case, state, self.samples)
            if first != audit:
                raise RuntimeError("Original native result and independent replay differ")
            _validated_native_result(audit, case=case, state=state, plan=source)
        except BaseException:
            self.done = True
            # Original case and invocation logs are preserved on any failure.
            raise
        previous = self.best_latency_ns
        qualified = audit["qualified"]
        if qualified:
            latency = audit["hls_estimated_latency_ns"]
            if self.best_latency_ns is None or latency < self.best_latency_ns:
                self.best_latency_ns = latency
                self.best_state_sha256 = state.state_sha256
        reward = latency_only_reward(
            previous, self.best_latency_ns, qualified=qualified,
        )
        self.current = state
        self.last_audit = audit
        self.history.append({
            "action_index": idx, "clock_ns": period, "audit": audit,
            "reward": reward, "best_ns": self.best_latency_ns,
        })
        self.done = self.calls >= self.max_vitis_calls
        try:
            observation = self._observation()
        except BaseException:
            self.done = True
            # An emitted native case cannot be retried after observation failure.
            raise
        truncated = self.done
        terminated = not truncated and not bool(np.any(observation["action_mask"]))
        info = {
            "schema": SCHEMA,
            "native_case": str(case.absolute()),
            "native_vitis_calls": self.calls,
            "qualified": qualified,
            "hls_estimated_latency_ns": audit["hls_estimated_latency_ns"],
            "best_hls_estimated_latency_ns": self.best_latency_ns,
            "hls_worst_case_cycles": audit["hls_worst_case_cycles"],
            "original_vitis_audit": audit,
            "source_mask_is_not_native_feasibility": True,
            "state_sha256": state.state_sha256,
        }
        # A separate episode-level record, not an edit to source or original
        # vendor-case artifacts. Fail rather than clobber any existing file.
        receipt = {
            "schema": SCHEMA, "state_sha256": state.state_sha256,
            "action_index": idx, "requested_clock_ns": period,
            "qualified": qualified, "reward": reward, "native_case": str(case),
            "best_hls_estimated_latency_ns": self.best_latency_ns,
            "original_case_audit": audit,
        }
        with (self._episode_path / f"step_{self.calls - 1:04d}.json").open(
                "x", encoding="utf-8") as f:
            json.dump(receipt, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
        return observation, float(reward), terminated, truncated, info

    def close(self):
        return None



def main(argv=None) -> int:
    """One-step hardware Gym smoke; default only inspects model/action mask.

    Does NOT launch online training, reuse prior native cases, or invent an
    initial qualified baseline. --execute-native is explicit, one case only.
    """
    import argparse
    import sys
    from fpgai_research.frontend.onnx import load_onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--samples", required=True, help="Existing float32 .npy cohort")
    parser.add_argument("--part", required=True, help="Exact target FPGA part")
    parser.add_argument("--resource-caps", required=True, nargs=5, type=int,
                        metavar="N", help="LUT FF DSP BRAM18K URAM task budgets")
    parser.add_argument("--max-absolute-error", type=float, required=True)
    parser.add_argument("--initial-clock-ns", type=float, default=14.00)
    parser.add_argument("--action-clock-ns", type=float,
                        help="Policy-selected continuous requested clock for one native step")
    parser.add_argument("--action-index", type=int, default=0,
                        help="0=clock-only; otherwise compiler-masked architecture choice")
    parser.add_argument("--max-parallel-units", type=int, default=8)
    parser.add_argument("--output-root", required=True,
                        help="Fresh experiment root; no old episode/case reuse")
    parser.add_argument("--execute-native", action="store_true",
                        help="Explicitly run exactly one original Vitis HLS case")
    parser.add_argument("--vitis", default="vitis_hls")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    try:
        model = load_onnx(args.onnx)
        cohort = np.load(args.samples, allow_pickle=False)
        environment = GymHardwareNativeEnv(
            model, cohort, args.output_root, part=args.part,
            resource_caps=tuple(args.resource_caps),
            max_absolute_error=args.max_absolute_error,
            initial_clock_ns=args.initial_clock_ns,
            max_parallel_units=args.max_parallel_units,
            execute_native=bool(args.execute_native), max_vitis_calls=1,
            vitis_executable=args.vitis, timeout_seconds=args.timeout,
        )
        observation, info = environment.reset()
        if args.execute_native:
            if args.action_clock_ns is None:
                raise ValueError("--action-clock-ns required for native step")
            observation, reward, terminated, truncated, info = environment.step(
                (args.action_index, args.action_clock_ns)
            )
            result = {
                "schema": SCHEMA, "status": "ONE_NATIVE_GYM_STEP_AUDITED",
                "reward": reward, "terminated": terminated,
                "truncated": truncated, "native_result": info,
                "original_native_calls": environment.calls,
            }
        else:
            from dataclasses import asdict
            result = {
                "schema": SCHEMA, "status": "SOURCE_ACTION_PREVIEW_ONLY",
                "state_sha256": info["state_sha256"],
                "original_native_calls": 0, "reward": None,
                "operator_count": len(model.operators),
                "graph_nodes": int(observation["node_mask"].sum()),
                "graph_edges": int(observation["edge_mask"].sum()),
                "source_offer_count": len(environment.offers),
                "source_accepted_mask_count": int(observation["action_mask"].sum()),
                "source_mask_is_not_native_feasibility": True,
                "first_action_choices": [
                    {"slot": i, "edit": asdict(a) if a is not None else None,
                     "source_legal": bool(observation["action_mask"][i])}
                    for i, a in enumerate(environment.offers[:16])
                ],
                "native_vitis_calls": 0,
                "training_started": False,
                "requested_clock_is_continuous": True,
                "native_clock_resolution_ns": .01,
                "physical_board_inference_verified": False,
            }
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, FileExistsError) as exc:
        print(f"Native hardware Gym stopped: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
