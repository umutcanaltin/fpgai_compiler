# P7 Public CLI Documentation Cleanup

## Goal

Make the repository present FPGAI as a professional CLI/package-based compiler instead of a collection of scripts.

## Changes

- Root `README.md` now uses `fpgai` commands for normal workflows.
- Added `docs/CLI_WORKFLOWS.md` to define public command patterns.
- Added `scripts/README.md` to mark `scripts/` as transitional/internal rather than the primary user interface.

## Public commands

- `fpgai inspect --config ...`
- `fpgai compile --config ...`
- `fpgai benchmark --config ...`
- `fpgai sweep inspect --config ...`
- `fpgai sweep run --config ...`
- `fpgai evidence inspect --config ...`

## Policy

Normal user documentation should not instruct users to run internal scripts directly. If a workflow is important, it should become a `fpgai` CLI command.
