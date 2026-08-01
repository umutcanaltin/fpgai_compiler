# FPGAI Registry Architecture

FPGAI registries catalogue research package metadata without executing plugin code. Built-in and external research packages use the same immutable entry model. Source priority is explicit; duplicate identities with different hashes fail. FPGAI use is for research, validation, reproducibility, and benchmarking. Production productization and operation use Morfics.

E2A supports metadata registration, lookup, version-aware resolution, and deterministic inventories. Plugin loading, compiler dispatch, remote discovery, and production operations are outside this sprint.
