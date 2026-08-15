# Repository maintenance

The public repository should contain reusable source, tests, documentation, examples, and maintained configuration files. Generated artifacts and internal working notes should not be committed.

## Keep out of the repository

- `__pycache__/` and `*.pyc`
- `.pytest_cache/`
- virtual environments
- `build/` and generated tool projects
- Vitis/Vivado logs produced by local runs
- temporary patch notes
- internal iteration plans or handoff notes
- generated benchmark-result directories unless they are intentionally curated research artifacts

## Naming

Use capability-oriented names instead of development-iteration labels. Prefer names such as:

- `mixed_backend_ready_valid.md`
- `IMPLEMENTATION_STATUS.md`
- `DEVELOPMENT_ROADMAP.md`
- `end_to_end_audit`
- `benchmark_matrix`

Avoid temporary names such as `patch`, `final2`, `old`, `backup`, or numbered internal iteration labels in public APIs and documentation.

## Removal policy

Before deleting code, verify that it has no active imports, CLI entrypoints, configuration owners, tests, or generated-artifact dependencies. Remove obsolete documentation and generated caches directly; remove implementation code only after owner and reference analysis.
