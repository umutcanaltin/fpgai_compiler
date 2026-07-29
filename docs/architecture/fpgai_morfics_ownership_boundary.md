# FPGAI and Morfics Ownership Boundary

## Decision

FPGAI is the open research, experimentation, validation, reproducibility, and benchmarking platform. Morfics is the commercial productization, deployment, and operations platform.

Community members may contribute models, ONNX import rules, FPGAI IR operators, HLS implementations, VHDL/Verilog/SystemVerilog implementations, boards, backends, optimizers, losses, datasets, memory policies, transports, validators, reporters, and benchmarks to FPGAI.

Compiling and validating those contributions in FPGAI is for research, engineering evaluation, education, reproducible experimentation, and benchmarking. Production use is delivered through Morfics.

## FPGAI responsibilities

FPGAI owns open research contracts and implementations for:

- model and ONNX import research;
- FPGAI IR and compiler-pass research;
- inference and training semantics;
- operator and implementation registries;
- HLS and RTL research backends;
- local C simulation, RTL simulation, synthesis, implementation, and hardware validation;
- resource, timing, numeric, and behavior reports;
- reproducibility manifests and benchmark formats;
- versioned compiler requests, results, and artifact contracts that Morfics can invoke;
- local research use without a Morfics account.

FPGAI does not promise production availability, managed operation, customer support, service-level guarantees, deployment certification, or fleet operation.

## Morfics responsibilities

Morfics owns:

- user accounts, organizations, projects, and private workspaces;
- model upload and managed build APIs;
- hosted inference and training;
- managed HLS, RTL, synthesis, implementation, and hardware queues;
- deployment to Morfics hardware and customer-owned FPGAs;
- production runtimes, device integration, telemetry, and fleet management;
- commercial adapters and proprietary packages;
- security, entitlements, billing, support, certification, updates, and rollback;
- visual embedded, robotics, and cloud application composition.

## Integration rule

Morfics may use FPGAI as its compiler backend through stable, versioned, deterministic contracts. Morfics must not depend on private compiler functions, terminal-output parsing, implicit file locations, or silent implementation fallback.

Open package contracts may describe proprietary Morfics packages without publishing their implementation source. Production orchestration and commercial policy remain outside the FPGAI repository.
