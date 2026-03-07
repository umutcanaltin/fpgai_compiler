from __future__ import annotations

import argparse
import sys

from fpgai.config.loader import load_config, print_summary, ConfigError
from fpgai.engine.compiler import Compiler


def main() -> None:
    ap = argparse.ArgumentParser(description="FPGAI compiler entrypoint")
    ap.add_argument("--config", required=True, help="Path to fpgai.yml")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
        print_summary(cfg)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)

    comp = Compiler(cfg)
    result = comp.compile()
    print(result.summary())


if __name__ == "__main__":
    main()
