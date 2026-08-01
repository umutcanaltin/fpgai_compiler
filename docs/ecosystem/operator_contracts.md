# FPGAI Logical Operator Contracts

FPGAI logical operator contracts describe what an operation means independently of any HLS, VHDL, Verilog, SystemVerilog, or software implementation.

FPGAI is an open research, validation, reproducibility, and benchmarking platform. A contract may be used to study and validate operator behavior locally. Commercial productization, managed deployment, hosted inference or training, and production support use Morfics.

## Separation of concerns

```text
Logical operator contract
  -> semantics, ports, attributes, ONNX bindings, capabilities

Concrete implementation
  -> HLS C++, VHDL, Verilog, SystemVerilog, simulator, or future backend
```

One logical operator can have multiple implementations. The compiler will later select an implementation according to explicit YAML choices and compatibility constraints.

## Contract contents

An operator contract declares:

- namespace-qualified operator ID
- canonical FPGAI operation type
- contract version
- category
- input and output tensor ports
- attributes
- ONNX domain, operation type, and opset ranges
- aliases
- inference and training capabilities
- shape and type inference capability
- numeric-reference capability
- canonicalization capability
- resource-estimation capability
- metadata entrypoints
- implementation requirements

E3A records entrypoint metadata but does not import or execute external Python code.

## Built-in compatibility

Existing `fpgai.layers.registry` data remains authoritative during the migration. `builtin_operator_contracts()` adapts it into logical contracts, and `builtin_operator_entries()` adapts those contracts into the common operator registry.

Compiler dispatch, ONNX import, reference execution, and HLS generation remain unchanged in E3A.
