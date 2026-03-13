from __future__ import annotations

import argparse
import sys

from fpgai.config.loader import load_config, print_summary, ConfigError
from fpgai.engine.compiler import Compiler
from fpgai.benchmark.pipeline import run_compile_correctness_benchmark


def parse_args():
    p = argparse.ArgumentParser("FPGAI compiler (v1)")
    p.add_argument("--config", required=True, help="Path to fpgai.yml")
    return p.parse_args()


def _benchmark_enabled(cfg) -> bool:
    raw = getattr(cfg, "raw", {}) or {}
    bench = raw.get("benchmark", {}) or {}
    return bool(bench.get("enabled", False))


def main():
    args = parse_args()

    try:
        cfg = load_config(args.config)
        print_summary(cfg)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)

    try:
        if _benchmark_enabled(cfg):
            print("\n=============== FPGAI Benchmark ===============")
            bench = run_compile_correctness_benchmark(config_path=args.config)
            print(f"Benchmark passed : {bench.passed}")
            print(f"Metrics JSON     : {bench.metrics_json}")
            print(f"Summary TXT      : {bench.summary_txt}")
            if bench.quant_metrics_json is not None:
                print(f"Quant JSON       : {bench.quant_metrics_json}")
            if bench.quant_summary_txt is not None:
                print(f"Quant Summary    : {bench.quant_summary_txt}")
            print("==============================================")
            if bench.passed:
                print("[OK] Benchmark completed successfully.")
            else:
                print("[WARN] Benchmark completed but did not pass thresholds.")
        else:
            compiler = Compiler(cfg)
            result = compiler.compile()
            print(result.summary())
            print(f"[OK] Wrote artifacts to: {result.out_dir}")
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()