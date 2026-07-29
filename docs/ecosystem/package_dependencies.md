# FPGAI research-package dependencies

FPGAI package dependencies are metadata-only declarations used for research reproducibility, benchmarking, and deterministic compiler preparation. They do not install packages, access the network, import plugin code, or invoke vendor tools.

## Declaration

```yaml
dependencies:
  - package: fpgai.operator.conv2d
    version: ">=1.0,<2.0"
    required: true

  - package: community.axi_stream_utils
    version: "^2.1"
    required: false
```

Supported version expressions include exact versions, PEP 440 ranges, caret ranges, tilde ranges, and `*`.

## Resolution rules

Resolution operates only over package roots explicitly supplied by the caller. It selects the highest available version satisfying all required constraints, reports missing required and optional dependencies separately, rejects conflicting duplicate identities, detects cycles, and produces deterministic dependency-first ordering.

No package silently replaces another package with the same ID and version but different manifest content.

## Lock manifest

A successful resolution can produce `fpgai.package-lock/v1`. Each locked package records its package ID, exact version, source class, manifest SHA-256 hash, and dependency declarations. The lock is designed for repeatable FPGAI research builds and future Morfics-managed compilation.

The lock file does not contain Morfics credentials, entitlement data, deployment state, or production-runtime configuration.

## Research and production boundary

Dependency resolution belongs to open FPGAI because reproducibility and community research packages require deterministic metadata. Package hosting, private access control, commercial entitlement, build queues, deployment, managed inference or training, and production operation belong to Morfics.
