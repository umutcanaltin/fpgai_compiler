"""Source-feedback-only competing search strategies over the SAME legal menu.

These are clearly named reference heuristics. The Parzen density-ratio
reference is not marketed as an independently tuned Optuna TPE benchmark.
No score is ever passed to the HLS environment as if it were measured QoR.
"""
from __future__ import annotations

import numpy as np

from .clock_actions import normalize_clock_period_ns


METHODS = ("random", "greedy", "evolutionary", "parzen")


class BaselineChooser:
    def __init__(self, method, *, seed):
        if method not in METHODS:
            raise ValueError("Unknown HLS baseline method")
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.history = []
        self.best = None
        self.best_clock_ns = None

    def choose(self, obs):
        valid = np.flatnonzero(obs["action_mask"])
        if not len(valid):
            raise ValueError("No legal unevaluated architectures")
        features = np.asarray(obs["action_features"])[valid]
        if self.method == "random":
            return int(self.rng.choice(valid))
        if self.best is None:
            # Feasibility-sparse experiments have no latency incumbent.
            # These are distinct named reference search policies, with
            # original negative outcomes, not four identical RNG fallbacks.
            if not self.history:
                return int(self.rng.choice(valid))
            attempted = np.stack([x[0] for x in self.history])
            distances = np.linalg.norm(
                features[:, None, :] - attempted[None, :, :], axis=-1)
            if self.method == "greedy":
                # Deterministic local continuity around the last attempted
                # configuration (no fabricated latency estimate).
                scores = distances[:, -1]
            elif self.method == "evolutionary":
                # Novelty search when selection by verified QoR is impossible.
                scores = -distances.min(axis=1)
            else:
                # Parzen-like kernel exclusion around known failures, not
                # a claim to independently tuned Bayesian/TPE optimization.
                bandwidth = .25
                scores = np.exp(
                    -(distances ** 2) / (2 * bandwidth ** 2)).mean(axis=1)
            tie = np.flatnonzero(scores <= scores.min() + 1e-9)
            return int(valid[int(self.rng.choice(tie))])
        if self.method == "greedy":
            target = self.best[0]
            scores = np.linalg.norm(features - target, axis=1)
        elif self.method == "evolutionary":
            elites = sorted((x for x in self.history if x[1] is not None),
                            key=lambda x: x[1])[:max(1, len(self.history) // 4)]
            if self.rng.random() < 0.25:
                return int(self.rng.choice(valid))
            target = elites[int(self.rng.integers(len(elites)))][0]
            scores = np.linalg.norm(features - target, axis=1)
        else:
            good = sorted((x for x in self.history if x[1] is not None),
                          key=lambda x: x[1])
            bad = [x[0] for x in self.history if x[1] is None]
            bad.extend(x[0] for x in good[len(good)//2:])
            good = [x[0] for x in good[:max(1, len(good)//2)]]
            if len(good) < 2 or len(bad) < 2:
                return int(self.rng.choice(valid))
            # Kernel density ratio on prior REAL-HLS observations; an
            # acquisition ranking only, never a synthetic latency receipt.
            bandwidth = 0.25
            def density(samples):
                diff = (features[:, None, :] - np.stack(samples)[None, :, :])
                squared = np.sum(diff * diff, axis=-1)
                return np.mean(np.exp(-squared / (2 * bandwidth**2)), axis=1) + 1e-10
            scores = -np.log(density(good) / density(bad))
        optimal = np.flatnonzero(scores <= scores.min() + 1e-9)
        return int(valid[int(self.rng.choice(optimal))])

    def choose_clock(self, obs):
        """Baseline may propose any bounded millips period, same as the policy."""
        low, high = (float(x) for x in obs["clock_bounds_ns"])
        if low == high:
            return normalize_clock_period_ns(low)
        if (self.best_clock_ns is not None and self.method != "random"
                and self.rng.random() < .75):
            suggested = self.best_clock_ns + float(self.rng.normal(0., 2.))
            suggested = min(high, max(low, suggested))
        else:
            suggested = float(self.rng.uniform(low, high))
        return normalize_clock_period_ns(suggested)

    def observe(self, selected_action_features, actual_feasible_latency_ns,
                *, clock_ns=None):
        feature = np.asarray(selected_action_features, dtype=np.float64).copy()
        measured = actual_feasible_latency_ns
        if measured is not None and (not np.isfinite(measured) or measured <= 0):
            raise ValueError("Baseline observations must be actual feasible HLS ns")
        self.history.append((feature, measured))
        if measured is not None and (self.best is None or measured < self.best[1]):
            self.best = (feature, measured)
            if clock_ns is not None:
                self.best_clock_ns = normalize_clock_period_ns(clock_ns)
