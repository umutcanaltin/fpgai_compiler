# R1 Professional Validation Naming Refactor

## Purpose

Start replacing informal reporting/product names with professional software terminology while preserving compatibility for existing tests and scripts.

## New canonical module

- `fpgai.reporting.paper_validation`
- CLI: `python -m fpgai.reporting.paper_validation ...`
- Output artifact kind: `paper_validation_trace`

## Legacy compatibility kept for now

- `fpgai.reporting.paper_evidence`
- `tests/test_paper_evidence_chain.py`

These are intentionally left in place so existing commands continue to work. They should be migrated in a later compatibility-removal sprint after all users and scripts call the new validation module.

## Canonical terminology

Use:

- validation
- verification
- artifact status
- implementation status
- support status
- build result
- synthesis result
- implementation result
- deployment result
- runtime result
- compliance
- traceability

Avoid product/reporting names based on:

- truth
- evidence
- unsafe wording

## Field rename map

- `paper_evidence_chain` -> `paper_validation_trace`
- `truth_level` -> `validation_level`
- `evidence_levels` -> `validation_status`
- `paper_safe` -> `paper_ready`
- `safe_wording` -> `paper_statement`
- `unsafe_wording` -> `pending_validation`
- `missing_evidence` -> `required_validation`
- `paper_safe_claim_level` -> `support_level`

## Next steps

1. Update acceptance commands to call `fpgai.reporting.paper_validation`.
2. Rename old tests once the full suite is stable.
3. Add compatibility shims only where needed.
4. Remove legacy names only after repository-wide references are migrated.
