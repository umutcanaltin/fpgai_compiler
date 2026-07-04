# FPGAI YAML schema audit — W0-lite

FPGAI is moving toward one canonical public YAML format. W0-lite is an audit sprint, not the final breaking cleanup sprint.

The compiler now emits:

```text
reports/config_contract.json
reports/config_contract.md
```

Those artifacts classify YAML paths in the current config as:

```text
canonical
  Preferred public key. It must affect generated artifacts/reports/runtime or reject clearly.

deprecated_alias
  Temporarily accepted legacy spelling. It must show its replacement path in the report.

legacy_or_internal
  Historical/internal section accepted by the loader but not selected as a public canonical key yet.

section_container
  Parent/container section where child keys carry the actionable contract.

sweep_template
  Sweep/materialization YAML key, not a direct compiler config key.

paper_artifact_spec
  Paper/report aggregation YAML key, not a direct compiler config key.

unclassified_known_section
  Known top-level section, but this exact key still needs a canonical/deprecated/rejected decision.

unknown_top_level
  Top-level section is not part of the accepted v1 section list.
```

W0-lite does **not** globally reject all unknown keys yet. Later YAML cleanup sprints will make this stricter after examples, migration documentation, and tests are complete.

## Priority rule

```text
manual YAML override > policy default > compiler default
```

No policy/default should silently override a manual YAML key.

## Canonical direction

Use direct, grouped keys:

```yaml
data_movement:
  inputs:
    interface: m_axi
    transport: ps_runtime
    tiled:
      enabled: false
      tile_size: 64
  outputs:
    interface: axi_stream
    transport: dma

optimization:
  pipeline:
    style: balanced
    ii: 1
  parallel:
    pe: 1
    simd: 1
    partition_factor: 1

weights:
  mode: embedded

training:
  optimizer:
    type: sgd
    learning_rate: 0.001
  loss:
    type: mse
  batch:
    size: 1
  accumulation:
    steps: 1
    mode: none
```

Avoid older nested or sprint-era aliases in new examples.

## Evidence requirement

Every canonical key must either:

```text
1. affect generated HLS C++ / Vivado / runtime artifacts,
2. appear in a validation/report artifact proving the decision,
3. reject clearly with a real reason,
4. or be implemented in the next sprint.
```

The config contract report is only the YAML audit layer. It does not replace movement, runtime, Vivado, board-fit, or numeric validation reports.

## W0-lite repository-level YAML audit

W0-lite also adds a repository scan over `configs/` and `examples/` so the current tree can be audited before production examples are created.

Generated audit artifacts:

```text
docs/YAML_REPO_AUDIT.json
docs/YAML_REPO_AUDIT.md
```

The repo audit is intentionally audit-only. It separates direct compiler config keys from sweep templates and paper aggregation specs, then reports the remaining deprecated aliases and unknown/unclassified keys. Later W0/W sprints will turn these findings into migration patches and stricter unknown-key rejection.

After the second W0-lite classification pass, the provided repo snapshot has zero unknown/unclassified YAML paths in `configs/` and `examples/`; the remaining actionable queue is deprecated aliases plus legacy/internal compatibility paths.

## W0-lite/Q0 foundation update

The Q0 foundation examples use canonical public keys for production examples under
`examples/inference`, `examples/training`, `examples/build`, and `examples/boards`.
The reference template under `examples/reference/full_options_reference.yml` is valid
YAML and documents alternatives, but it is not intended to enable every option at once.

Additional canonical paths classified in this update:

- `weights.*`
- `build.existing_hls_ip`
- `codegen.*`
- `training.batch.size`
- `training.batch.epochs`
- `memory.*_storage`

Artifact validation should now include report inspection, not only pytest. Use
`python -m fpgai.reporting.artifact_smoke <compile-out-dir> [...]` to summarize
whether generated C++, runtime package, movement reports, HLS truth reports, Vivado
truth reports, and bitstream reports are present and what evidence level they support.

## Batch 2 example expansion

Batch 2 extends Q0 examples while keeping W0-lite audit mode. Production examples under `examples/inference`, `examples/training`, `examples/boards`, and `examples/build` are expected to use canonical YAML keys and pass `load_config`. Reference and paper sweep examples are parsed and audited but are not direct compiler configs until materialized.

The repository audit currently distinguishes direct compiler configs from sweep templates and paper artifact specifications. `docs/YAML_REPO_AUDIT.md` is the migration queue for the remaining deprecated aliases in older configs.
