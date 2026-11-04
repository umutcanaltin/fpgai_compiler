"""Gymnasium interface for the original source-audited, HLS-only backend.

Graph observations are padded to the *currently compiled* 1..4-operator
MatMul/Gemm scope. Padding masks are explicit; no synthetic hardware rewards.
The clock is never a problem input: the actor outputs a floating period at\n1 ps resolution within internal safety bounds. Real HLS determines feasibility.\nEvery real step requires explicit foreground native opt-in.
"""
from __future__ import annotations

from math import log1p, prod
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from fpgai_research.architecture.resolver import iteration_shape
from .post_hls_env import PostHLSEnv
from .clock_actions import (CLOCK_MIN_NS, CLOCK_MAX_NS, action_bounds_ns,
                            normalize_clock_period_ns, validate_clock_action)


MAX_NODES = 4
NODE_FEATURES = 8
KNOBS = 8
CONTEXT = 30
MAX_HISTORY = 30
HISTORY_FEATURES = MAX_NODES * KNOBS + 9


class GymGraphEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(self, task, root, *, execute_native=False,
                 max_candidates=12, timeout_seconds=7200, vitis_executable="vitis_hls"):
        if gym is None:
            raise RuntimeError("Install optional dependency: pip install -e '.[rl]'")
        if not 1 <= len(task.model.operators) <= MAX_NODES:
            raise ValueError("Compiled research scope is 1..4 operations")
        if not 1 <= max_candidates <= 128:
            raise ValueError("Bounded action menu required")
        self.task = task
        self.root = Path(root)
        self.execute_native = bool(execute_native)
        self.max_candidates = max_candidates
        self.timeout_seconds = timeout_seconds
        self.vitis_executable = vitis_executable
        self.episode_index = 0
        self.backend = None
        self.offers = ()
        self.max_actions = max_candidates + 3
        # Discrete compiler-legal architecture and *floating* learned clock.
        low, high = action_bounds_ns(task)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.max_actions),
            spaces.Box(low, high, shape=(), dtype=np.float64),
        ))
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(-1e6, 1e6, (MAX_NODES, NODE_FEATURES), np.float32),
            "edges": spaces.Box(0, 1, (MAX_NODES, MAX_NODES), np.float32),
            "node_mask": spaces.MultiBinary(MAX_NODES),
            "current_arch": spaces.Box(0, 1, (MAX_NODES, KNOBS), np.float32),
            "best_arch": spaces.Box(0, 1, (MAX_NODES, KNOBS), np.float32),
            "global": spaces.Box(-1e6, 1e6, (CONTEXT,), np.float32),
            "action_features": spaces.Box(-1e6, 1e6,
                                         (self.max_actions, MAX_NODES * KNOBS), np.float32),
            "clock_bounds_ns": spaces.Box(CLOCK_MIN_NS, CLOCK_MAX_NS,
                                          (2,), np.float32),
            "action_mask": spaces.MultiBinary(self.max_actions),
            "history": spaces.Box(-1e6, 1e6,
                                  (MAX_HISTORY, HISTORY_FEATURES), np.float32),
            "history_mask": spaces.MultiBinary(MAX_HISTORY),
        })

    def _choices(self):
        b = self.backend
        if b.done:
            return ()
        proposals = list(b.candidates(max_candidates=self.max_candidates))
        for current in (b.current, b.catalog.baseline,
                        b.best_candidate[0] if b.best_candidate else None):
            if current is not None and current not in proposals:
                proposals.append(current)
        # A proposed architecture can be evaluated again at another floating
        # period. Fixed-clock ablation has no other period, so mask out pairs
        # already attempted instead of ever proposing a duplicate native case.
        fixed = self.task.fixed_clock_ablation_ns
        if fixed is not None:
            proposals = [candidate for candidate in proposals
                         if (candidate, fixed) not in b.tried_configs]
        return tuple(proposals[:self.max_actions])

    def _observation(self):
        task, b = self.task, self.backend
        self.offers = self._choices()
        nodes = np.zeros((MAX_NODES, NODE_FEATURES), np.float32)
        edges = np.zeros((MAX_NODES, MAX_NODES), np.float32)
        mask = np.zeros(MAX_NODES, np.int8)
        producer = {name: i for i, op in enumerate(task.model.operators)
                    for name in op.outputs}
        for i, op in enumerate(task.model.operators):
            shape = iteration_shape(task.model, op)
            a = task.model.tensor(op.inputs[0]).shape
            out = task.model.tensor(op.outputs[0]).shape
            nodes[i] = [
                float(op.kind == "MatMul"), float(op.kind == "Gemm"),
                *[log1p(float(x)) / 12 for x in shape],
                log1p(prod(a)) / 16, log1p(prod(out)) / 16,
                log1p(len(op.inputs)) / 4,
            ]
            mask[i] = 1
            for tensor in op.inputs:
                if tensor in producer:
                    edges[producer[tensor], i] = 1.
        remaining = task.max_vitis_calls - b.calls
        last = b.last_result or {}
        measured = last.get("hls_estimated_resources") or {}
        values = [
            *[log1p(v) / 20 for v in task.resources],
            log1p(task.max_absolute_error),
            log1p(remaining) / 6,
            log1p(b.calls) / 6,
            float(b.best_latency_ns is not None),
            log1p(b.best_latency_ns) / 20 if b.best_latency_ns else 0,
            float(last.get("qualified") is True),
            log1p(last["hls_worst_case_cycles"]) / 20
                if last.get("hls_worst_case_cycles") else 0,
            log1p(last["hls_estimated_clock_ns"]) / 10
                if last.get("hls_estimated_clock_ns") else 0,
            float(last.get("hls_estimated_clock_ns") is not None),
            log1p(len(task.model.operators)) / 5,
            *[v for name in ("lut", "ff", "dsp", "bram_18k", "uram")
              for v in (float(measured.get(name) is not None),
                        log1p(measured[name]) / 20 if measured.get(name) is not None else 0)],
            float(last.get("csim_max_absolute_error") is not None),
            log1p(last["csim_max_absolute_error"])
                if last.get("csim_max_absolute_error") is not None else 0,
            float(last.get("hls_timing_feasible") is True),
            log1p(last["requested_clock_ns"]) / 10
                if last.get("requested_clock_ns") else 0,
            log1p(b.best_candidate[1]) / 10 if b.best_candidate else 0,
        ]
        assert len(values) == CONTEXT
        action_features = np.zeros((self.max_actions, MAX_NODES * KNOBS),
                                   dtype=np.float32)
        action_mask = np.zeros(self.max_actions, np.int8)
        for slot, candidate in enumerate(self.offers):
            vector = action_features[slot]
            for i, indices in enumerate(candidate):
                for j, index in enumerate(indices):
                    vector[i * KNOBS + j] = (
                        index / max(1, len(b.catalog.options[i][j]) - 1))
            action_mask[slot] = 1
        def architecture_features(candidate):
            features = np.zeros((MAX_NODES, KNOBS), dtype=np.float32)
            if candidate is not None:
                for i, indices in enumerate(candidate):
                    for j, index in enumerate(indices):
                        features[i, j] = index / max(
                            1, len(b.catalog.options[i][j]) - 1)
            return features

        history = np.zeros((MAX_HISTORY, HISTORY_FEATURES), np.float32)
        history_mask = np.zeros(MAX_HISTORY, np.int8)
        for slot, (candidate, clock, result) in enumerate(b.history[-MAX_HISTORY:]):
            row = history[slot]
            row[:MAX_NODES * KNOBS] = architecture_features(
                candidate).reshape(-1)
            start = MAX_NODES * KNOBS
            error = result.get("csim_max_absolute_error")
            resources = result.get("hls_estimated_resources") or {}
            ratios = [float(resources[k]) / max(1, budget)
                      for k, budget in task.budgets.items()
                      if resources.get(k) is not None]
            timing = result.get("hls_estimated_clock_ns")
            latency = result.get("hls_estimated_latency_ns")
            row[start:] = [
                float(clock) / CLOCK_MAX_NS,
                float(result.get("qualified") is True),
                min(10., max(0., float(error) / task.max_absolute_error))
                    if error is not None else 0.,
                float(error is not None),
                min(10., max(ratios)) if ratios else 0.,
                float(timing) / CLOCK_MAX_NS if timing is not None else 0.,
                float(timing is not None),
                log1p(latency) / 20 if latency is not None else 0.,
                float(result.get("evidence") == "E3-HLS-synthesis-report"),
            ]
            history_mask[slot] = 1
        return {
            "history": history, "history_mask": history_mask,
            "nodes": nodes, "edges": edges, "node_mask": mask,
            "current_arch": architecture_features(b.current),
            "best_arch": architecture_features(
                b.best_candidate[0] if b.best_candidate else None),
            "global": np.asarray(values, dtype=np.float32),
            "action_features": action_features, "action_mask": action_mask,
            "clock_bounds_ns": np.asarray(action_bounds_ns(task),
                                          dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.root.mkdir(parents=True, exist_ok=True)
        episode = self.root / f"episode_{self.episode_index:06d}"
        self.episode_index += 1
        self.backend = PostHLSEnv(self.task, episode, seed=seed or 0)
        obs = self._observation()
        return obs, {"task_sha": self.task.task_sha, "original_vitis_calls": 0}

    def step(self, action):
        if self.backend is None:
            raise RuntimeError("Call reset() before step()")
        if (not isinstance(action, (tuple, list)) or len(action) != 2
                or type(action[0]) not in (int, np.int64, np.int32)
                or action[0] < 0 or action[0] >= len(self.offers)):
            raise ValueError("Invalid or masked architecture-clock action")
        # Gymnasium may sample arbitrary real periods; normalize them before
        # the original native backend, which requires exact canonical floats.
        decimals = 2 if self.task.curriculum_stage in ("L5", "L6") else 3
        period = validate_clock_action(
            self.task, normalize_clock_period_ns(action[1], decimals=decimals))
        if not self.action_space.contains((
                int(action[0]), np.asarray(period, dtype=np.float64))):
            raise ValueError("Clock action outside Gym action space")
        if not self.execute_native:
            raise RuntimeError("Native Vitis requires explicit execute_native=True")
        candidate = self.offers[int(action[0])]
        result = self.backend.step(
            candidate, clock_ns=period, execute_native=True,
            vitis_executable=self.vitis_executable,
            timeout_seconds=self.timeout_seconds,
        )
        obs = self._observation()
        truncated = self.backend.calls >= self.task.max_vitis_calls
        terminated = not truncated and not bool(np.any(obs["action_mask"]))
        info = {
            "original_vitis_calls": self.backend.calls,
            "task_sha": self.task.task_sha,
            "qualified": result["result"]["qualified"],
            "hls_estimated_latency_ns": result["result"]["hls_estimated_latency_ns"],
            "native_case": f"case_{self.backend.calls-1:04d}",
            "source_attributed_result": result,
        }
        return obs, float(result["reward"]), terminated, truncated, info
