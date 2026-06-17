#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fpgai.experiments.config_materializer import probe_parameter_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe FPGAI config paths available for Sprint 8 materialization")
    parser.add_argument("--config", default="fpgai.yml")
    args = parser.parse_args()
    path = Path(args.config)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    print(json.dumps({"config": str(path), "existing_parameter_paths": probe_parameter_paths(data)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
