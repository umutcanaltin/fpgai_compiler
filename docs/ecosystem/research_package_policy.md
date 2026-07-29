# FPGAI Research Package Policy

FPGAI packages are contributions to the open FPGAI research platform. They are intended for research, experimentation, validation, benchmarking, reproducibility, and education.

A package may contain a model, ONNX import extension, FPGAI IR operator, HLS or RTL implementation, board description, backend experiment, optimizer, loss, dataset, validator, reporter, benchmark, or research runtime reference.

The package manifest must declare:

```yaml
usage:
  platform_scope: research
  permitted_uses:
    - research
    - experimentation
    - validation
    - benchmarking
  production_path: morfics
```

This declaration describes the FPGAI platform boundary. It does not replace the package's legal license.

Commercial productization, hosted inference or training, production deployment, customer FPGA delivery services, managed runtime operation, telemetry, fleet management, security, billing, certification, and production support belong to Morfics.

FPGAI remains suitable as the deterministic compiler and validation backend consumed by Morfics, but Morfics production services and proprietary implementation code do not become part of the open FPGAI repository.
