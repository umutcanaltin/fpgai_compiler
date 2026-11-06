# Compiler-paper publication cleanup — source and file impact map

**6 November 2026 | Research branch `research/first-paper-source-memory-bridge-20261106`, draft PR #2. This is an editorial, distribution-hygiene and read-only analysis change set, NOT a new compiler or native HLS implementation and NOT an authorized public release.**

## Grounded original inventory and exact change baseline

GitHub's full recursive tree at **author-verified original C1/C2 execution SHA** `637f2745721b32d6f5a7e02c4659b0623bc9eee7` contained **880 Git-tracked blobs** and **911 total tree entries**, with no tracked `.gitignore`, GitHub community issue/PR templates or root operative LICENSE. The tracked tree included **425 files under `src/fpgai_research/`**, **242 tests**, **101 `paper/compiler_ir/` documents**, **60 `rl_environment/` files**, and **25 `docs/` files**, plus other roots/examples. The filename-level scan did not find tracked `.onnx`, `.npy`, `.npz`, `.pt`, `.pth`, `.bit`, `.dcp`, `.rpt`, `.log`, or obvious `build/` outputs in that snapshot. **This is NOT a content-level or full-Git-history credential/IP audit.**

The direct GitHub compare from that original SHA to the cleanup branch head (before this report) showed **19 changed files**: **8 added**, **11 modified**, **0 deleted or renamed**. No original C1/C2 study outputs were ever committed; their original author-run `build/` source outputs remain local and immutable.

## Precise file-level impact, dependencies and connectivity

| Changed file | Why it exists/changed | Dependencies and immediate impact | Source-evidence status |
|---|---|---|---|
| **ADDED** `.gitignore` | Protect future local original studies, test caches, generated ONNX/weights, Vitis/Vivado reports/projects, credentials and release approvals from accidental Git tracking | Git's default ignore behavior for *untracked* files only; original committed history not removed. Preserves `build/` original C1/C2/R4 studies | Static source check only; verify with `git check-ignore` and local tests |
| **ADDED** `.github/ISSUE_TEMPLATE/bug_report.md` | Reproducible public-facing bug template with source revisions, evidence levels and no secret/model uploads | GitHub issue UI; `scripts/prepare_public_preview.PUBLIC_FILES` already expected it; `.github` is **not** a workflow | Template only, no Actions |
| **ADDED** `.github/ISSUE_TEMPLATE/feature_request.md` | Source/hardware/experiment acceptance, license and safety controls | GitHub issue UI; existing preview allowlist | Template only, no Actions |
| **ADDED** `.github/PULL_REQUEST_TEMPLATE.md` | Require detailed file/folder change-impact records, target tests and scientific interpretation | GitHub PR UI; existing preview allowlist; source compilers unaffected | Template only, no Actions |
| **ADDED** `docs/RESEARCH_PUBLICATION_STATUS.md` | Public-safe canonical research/publication overview, original 72-pass C1/C2 outcome, experiment SHA identity, compiler-versus-paper scope, unknown physical latency and release blockers | Linked from `README.md` and `docs/README.md`, included in curated `MANIFEST.in` and `scripts/prepare_public_preview.PUBLIC_FILES`. Links to archival paper on GitHub via web URL so compiler-only preview does not include `paper/` | Original C1/C2 results author-verified; C2 CPU per-method timing summary still UNKNOWN |
| **ADDED** `src/fpgai_research/experiments/cst_paper_c2_original_readonly_analysis.py` | Strict SHA-pinned independent interpretation of **existing** 324-row C2 original report; calculates paired direct/lazy/eager median CPU times, paired wins, compile-call totals and excluded setup | Reads only `build/paper_c2_fair_source_cost_20261106_01/{report.json,artifact_sha256.json}` from original experiment source revision `637f274...`; no `unified_transition`, HLS, vendor or new report writes. Exporter original left intact | NEW SOURCE ANALYZER, tests/workstation original invocation pending |
| **ADDED** `tests/test_cst_paper_c2_original_readonly_analysis.py` | Synthetic negative/positive tests for 324 paired-row SHA analysis and forged/partial result rejection | Imports new read-only analysis only; does not run original C2 compiler or vendor | New tests NOT yet author-run |
| **ADDED** `tests/test_repository_publication_hygiene.py` | Guard `.gitignore`, compiler preview allowlist, Markdown link closure, truthful no-license/no-E4 claims and absence of GitHub Actions | Calls existing `scripts/prepare_public_preview.plan` and `scripts/check_public_docs.audit_source_links`; no native tool calls | New tests NOT yet author-run |
| **MODIFIED** `AGENTS.md` | Add current research-branch instruction and mandatory per-file change-impact rule above dated, preserved historical engineering notes | Guides future maintainers only; no compiler import change | Documentation only |
| **MODIFIED** `CHANGELOG.md` | Log original observed 72/486/324 source results and cleanup scope, original evidence preservation, release blockers | Public chronology; original older entries retained | Documentation only |
| **MODIFIED** `MANIFEST.in` | Include `docs/RESEARCH_PUBLICATION_STATUS.md` and previously linked PYNQ pilot user guide in curated Python sdist | Source package **still excludes** archival `paper/`, RL source and private native outputs | Packaging metadata change; build/install still untested |
| **MODIFIED** `README.md` | Add one clear link to research/paper/release status and no-operative-license disclosure, while retaining synthetic ONNX user quickstart | Public root entrypoint; no CLI/production source changes | Documentation only |
| **MODIFIED** `docs/OPEN_SOURCE_RELEASE_CHECKLIST.md` | Record no-license/rights/full-history and separate paper-artifact gate despite technical cleanup | Rights-holder review; no automatic release or visibility change | Documentation only |
| **MODIFIED** `docs/README.md` | Add research publication status in user documentation index | Both public doc and curated preview link targets available | Documentation only |
| **MODIFIED** `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md` | Incorporate original 54/486 C1, 324-row C2 data and strict limits; make no CPU/FPGA latency superiority claim | Relies on original source revision, 2 report SHA digests, signed earlier HLS/routed evidence and separate benchmark methodology | Manuscript editing only; no new experiment |
| **MODIFIED** `paper/compiler_ir/FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md` | Replace original planned C1/C2 metrics with author-verified results, add report identity and new untested C2 analysis gate | Active manuscript, original source reports and independent original report SHA audits | Evidence-index editing only |
| **MODIFIED** `paper/compiler_ir/README.md` | Replace stale 'not yet run' status and mistaken 'not public' assertion with author-verified original results and publicly-viewable/no-license distinction | Navigational paper index; archival notebooks still reachable and untouched | Documentation only |
| **MODIFIED** `paper/compiler_ir/SCIENTIFIC_STRENGTHENING_C1_C2_PROTOCOL_20261106.md` | Record original C1/C2 success SHA provenance, 72 focused tests, zero vendor calls and read-only C2 analysis yet untested | Original R3j/R3f/R3h source study lineage and C1/C2 experiment drivers left unchanged | Documentation only |
| **MODIFIED** `scripts/prepare_public_preview.py` | Explicitly allowlist new research status guide and formerly missing `docs/PYNQ_Z2_BOARD_DMA_PILOT.md`, which six existing public docs already reference | Supports `scripts/check_public_docs` local link integrity, `docs/RELEASE_PREVIEW.md`, and existing preview tests. **The PYNQ pilot is still BOARD_NOT_QUALIFIED** | Preview allowlist metadata only, no upload or publication |

**After this file is added**, there is one extra documentation file in `docs/maintainers/`. This file is deliberately excluded from the clean compiler sdist and normal compiler-only preview.

## What we inspected and what remains unresolved

- **Current recursive tracked filenames:** inspected at SHA `637f274...`; no obvious binary native/model artifact suffixes present. This does **not** inspect contents, past branches/tags/commit blobs, local untracked files or third-party licensing.
- **Public relative Markdown links:** source review found six links to the existing PYNQ-Z2 pilot guide missing from the release allowlist. Added the guide to `MANIFEST.in` and preview policy rather than deleting historical board evidence or stripping valid links. Full local `scripts/check_public_docs.py --profile compiler` remains the qualifying check.
- **Release templates and `.gitignore`:** previously required by `scripts/prepare_public_preview.plan` but missing from the active source tree; now created. No Actions workflow added.
- **Packaging:** source-only compiler `sdist` is not a full scholarly reproducibility package because it deliberately excludes archival `paper/` evidence and RL research. A paper artifact will need a separate rights-cleared curated protocol and signed evidence manifest; the user's local original `build/` data are not in Git.
- **Known release blockers:** no operative root LICENSE, no signed institutional/third-party rights clearance, no verified all-ref Git-history secret scan, no clean-room package-install regression, no independent complete first-paper scientific review. Do NOT interpret an ignore rule as deleting a previously exposed secret.
- **Known scientific blockers:** executed matched competitor (Allo) comparison, generated executable source-event refinement, C2 cost interpretation and demonstrated benefit beyond trivial catalog cached selection, timing-qualified physical latency. Current paper remains MAJOR REVISION.

## Safe workstation publication-hygiene and result inspection checks

All commands below are SOURCE ONLY; no native HLS/FPGA calls. They do not modify original results. A separate full regression and an independent rights/security release review are still required.

```bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m compileall -q \
  scripts/prepare_public_preview.py \
  src/fpgai_research/experiments/cst_paper_c2_original_readonly_analysis.py \
  tests/test_cst_paper_c2_original_readonly_analysis.py \
  tests/test_repository_publication_hygiene.py
python -m pytest -q \
  tests/test_cst_paper_c2_original_readonly_analysis.py \
  tests/test_repository_publication_hygiene.py \
  tests/test_public_release_preview.py \
  tests/test_public_docs_links.py \
  tests/test_public_distribution_surface.py
python scripts/prepare_public_preview.py --plan --profile compiler
python scripts/check_public_docs.py --profile compiler
python scripts/check_open_source_readiness.py
python -m fpgai_research.experiments.cst_paper_c2_original_readonly_analysis \
  --study build/paper_c2_fair_source_cost_20261106_01
git status --short
```

`check_open_source_readiness.py` correctly reports `ready_to_switch_visibility=false` irrespective of local file hygiene: legal, history and clean-room human approvals are separate. This is a positive fail-closed behavior, not a passing release authorization.

**STOP if any new test fails.** Preserve original saved C1/C2 output under `build/` and do not rerun, overwrite or purge it. If the new read-only analyzer disagrees with the author's frozen C2 report hash or observed row population, treat it as a source/provenance issue, not permission to change original measurements.
