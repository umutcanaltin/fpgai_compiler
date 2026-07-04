# Cleanup candidates

This document starts U0-lite cleanup tracking. It is intentionally conservative: uncertain files should be listed here before deletion.

## Current U0-lite policy

- Do not delete uncertain scripts or old experiment artifacts yet.
- Prefer migrating YAML examples/configs to canonical keys first.
- Keep legacy alias support in the compiler until W0/W strict schema work.
- Remove only obvious duplicates in later cleanup patches after grep/import/test checks.

## Candidate categories for later U0/W0

- Old configs that still use deprecated aliases after Q0 examples are complete.
- Sweep templates that should live under a single documented sweep format.
- Historical paper experiment outputs that should move out of the public package or be marked archival.
- Detached scripts that should become library/CLI commands or be removed if unused.
- Oversized tests that can be split only after coverage-equivalent fixtures exist.
