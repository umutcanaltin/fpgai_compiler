# FPGAI developer tools

This package contains maintained repository diagnostics and developer-only
workflows. File names must state the action and object clearly.

## Repository maintenance

- `repository_audit.py`: reports generated artifacts, ambiguous names, caches,
  and oversized source or test modules. It never changes files.
- `clean_generated_artifacts.py`: previews or removes generated outputs and
  cache directories. The default mode is a dry run.

## Existing compiler diagnostics

- `audit_knob_materialization.py`: verifies that selected YAML knobs affect
  generated implementation artifacts.
- `canonical_hls_source_audit.py`: checks generated HLS source ownership.
- `contract_source_audit.py`: checks source-level implementation contracts.
- `end_to_end_audit.py`: coordinates broader implementation audits.
- `hls_source_effect_audit.py`: checks whether configuration changes affect HLS.
- `probe_config_schema.py`: inspects configuration-schema behavior.

Do not add temporary one-use scripts here. A reusable product workflow belongs
in the public CLI. A maintained developer diagnostic belongs here with tests and
a descriptive module name.
