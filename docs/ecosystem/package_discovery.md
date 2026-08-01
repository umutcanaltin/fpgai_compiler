# Safe package discovery

FPGAI discovers research packages from project-local and explicitly configured directories without importing package Python modules or executing package code.

The default project location is `<project_root>/packages`. Configured directories are supplied explicitly by the caller. Discovery reads and validates `fpgai.yaml`, calculates manifest hashes, creates metadata-only registry entries, and produces deterministic reports.

```python
from fpgai.discovery import DiscoveryRequest, discover_packages

result = discover_packages(DiscoveryRequest(
    project_root="./project",
    configured_directories=("./shared_packages",),
    include_builtin=True,
))
```

Invalid packages are quarantined in permissive mode. Strict mode fails the discovery operation when a package is invalid, conflicting, or a required configured search root is missing.

FPGAI discovery is for research, validation, benchmarking, and reproducibility. Morfics later supplies approved package locations to FPGAI for managed compilation and owns production access control, private registries, deployment, and operations.
