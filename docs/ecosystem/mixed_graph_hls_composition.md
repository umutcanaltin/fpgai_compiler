# Mixed-graph external HLS composition

FPGAI E4C composes explicitly selected external HLS implementations into the normal inference HLS project. Logical operator loading and implementation selection remain separate. The resulting immutable composition plan binds each external graph node to one implementation contract, stages package-owned source files read-only, and lets the ordinary graph emitter generate one top kernel containing built-in and external calls.

The first composition profile is sequential `flat_array_v1`: one runtime input, one runtime output, static shapes, equal flattened element counts, and no unresolved branch liveness. Unsupported topology is rejected rather than silently lowered incorrectly. Node-specific implementation preferences take precedence over logical-operator preferences, followed by the global selection policy.

Reports record node provenance, package versions and hashes, selected implementations, conversion-buffer requirements, staged artifacts, and the FPGAI research/Morfics production boundary.
