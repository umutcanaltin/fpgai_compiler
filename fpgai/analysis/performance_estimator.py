from __future__ import annotations

from typing import Any, Dict


def _cfg_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def estimate_performance(
    *,
    resource_estimate: Dict[str, Any],
    raw_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    clk_mhz = float(_cfg_get(raw_cfg, "targets.platform.clocks.0.target_mhz", 200.0))
    cpu_baseline_ms = float(_cfg_get(raw_cfg, "analysis.design_space.performance.baseline_cpu_latency_ms", 1.0))

    total_macs = int(resource_estimate["totals"]["total_macs"])
    dsp = max(1, int(resource_estimate["totals"]["predicted_dsp"]))
    lut = max(1, int(resource_estimate["totals"]["predicted_lut"]))
    bram = max(1, int(resource_estimate["totals"]["predicted_bram18"]))

    parallel_macs = max(1.0, min(dsp * 2.0, lut / 300.0))
    compute_cycles = total_macs / parallel_macs
    memory_cycles = 800.0 + 0.02 * lut + 8.0 * bram
    total_cycles = compute_cycles + memory_cycles

    latency_ms = total_cycles / (clk_mhz * 1e3)
    throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    speedup_vs_cpu = cpu_baseline_ms / latency_ms if latency_ms > 0 else 0.0

    return {
        "clock_mhz": clk_mhz,
        "predicted_parallel_macs": float(parallel_macs),
        "predicted_cycles": float(total_cycles),
        "predicted_latency_ms": float(latency_ms),
        "predicted_throughput_fps": float(throughput_fps),
        "predicted_speedup_vs_cpu": float(speedup_vs_cpu),
    }