# Built-in Registry Migration

The existing `fpgai.layers.registry` remains authoritative during E2A. `fpgai.registries.builtin_layers` adapts its capability records into immutable operator registry entries. Compiler dispatch is unchanged. Later sprints may migrate execution selection only after compatibility tests and approval.
