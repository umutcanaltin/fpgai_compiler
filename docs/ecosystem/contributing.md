# Contributing to the FPGAI Ecosystem

FPGAI has one extension system: the **FPGAI Ecosystem**. External research contributions are distributed as `fpgai.package/v1` packages and discovered through the existing ecosystem registry. There is no separate community subsystem.

## Contribution types

The ecosystem can represent models, operators/layers, hardware implementations, accelerators, boards, backends, optimizers, losses, datasets, memory policies, transports, runtime references, validation providers, reporters, benchmarks, system blocks, and adapters.

Contributor-friendly aliases are available for common hardware assets:

- `layer` -> `operator`
- `hls` / `hls_cpp` -> `implementation` with `hls_cpp`
- `vhdl` -> `implementation` with `vhdl`
- `verilog` -> `implementation` with `verilog`
- `systemverilog` -> `implementation` with `systemverilog`

## Create a contribution

```bash
fpgai ecosystem init \
  --type layer \
  --id example.layer.my_operator \
  --out packages/my_operator
```

```bash
fpgai ecosystem init \
  --type hls \
  --id example.impl.my_operator_hls \
  --out packages/my_operator_hls
```

```bash
fpgai ecosystem init \
  --type vhdl \
  --id example.impl.my_operator_vhdl \
  --out packages/my_operator_vhdl
```

```bash
fpgai ecosystem init \
  --type model \
  --id example.model.my_model \
  --out packages/my_model
```

Use `fpgai ecosystem types` to list canonical asset types.

## Validate without executing contributor code

```bash
fpgai ecosystem validate packages/my_operator
```

Validation checks the existing package contract, research/production boundary, paths, versioning, entrypoints, interfaces, and declared validation level.

## Discover through the existing ecosystem registry

```bash
fpgai ecosystem discover --project-root .
```

Discovery uses the existing package discovery, quarantine, conflict-resolution, dependency, and registry infrastructure. Contribution scaffolding does not create a second registry.

## Operator and implementation separation

An operator contribution describes semantics, shape/type behavior, frontend binding, reference behavior, and training capabilities. Physical implementations are separate ecosystem assets.

For example:

```text
example.operator.foo
example.implementation.foo_hls
example.implementation.foo_vhdl
```

This allows multiple HLS/VHDL implementations to target the same logical operator and lets users select implementations through the existing FPGAI implementation-selection system.

See also:

- `docs/ecosystem/adding_an_operator.md`
- `docs/ecosystem/adding_hardware_implementation.md`
- `docs/ecosystem/package_manifest_v1.md`
- `docs/ecosystem/package_discovery.md`
- `docs/ecosystem/registry_architecture.md`
- `docs/ecosystem/implementation_selection.md`
