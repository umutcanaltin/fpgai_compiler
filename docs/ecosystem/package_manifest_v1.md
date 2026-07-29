# FPGAI Research Package Manifest v1

`fpgai.package/v1` is the metadata contract for FPGAI community and project-local research packages.

## Design goals

- one package format for models, operators, HLS, VHDL, Verilog, SystemVerilog, boards, backends, training extensions, validators, and benchmarks;
- metadata inspection without importing or executing contributor code;
- explicit inference and training capabilities;
- explicit compatibility and validation claims;
- safe package-relative paths;
- deterministic machine-readable errors;
- compatibility with future Morfics submission and compiler-backend workflows.

## Minimum manifest

```yaml
schema: fpgai.package/v1

package:
  id: community.example
  name: Example package
  version: 1.0.0
  asset_type: implementation
  provider: community
  description: Example FPGAI research implementation.

usage:
  platform_scope: research
  permitted_uses: [research, experimentation, validation, benchmarking]
  production_path: morfics

license:
  category: open_source
  identifier: Apache-2.0

compatibility:
  fpgai_contract: ">=1.0,<2.0"

capabilities:
  inference: true
  training:
    forward: false

entrypoints:
  implementation:
    language: hls_cpp
    top: example_top
    sources: [src/example.cpp]

validation:
  declared_level: unvalidated
```

## Safety

All source, model, reference, testbench, and constraint paths must be relative to the package root. Absolute paths, parent traversal, environment expansion, and network URLs are rejected.

Validation reads YAML metadata only. It does not import Python modules, run subprocesses, invoke vendor tools, or access the network.

## Validation command

```bash
python -m fpgai.contracts.package_validation ./my_package
```

Use `--json` for a stable machine-readable result.
