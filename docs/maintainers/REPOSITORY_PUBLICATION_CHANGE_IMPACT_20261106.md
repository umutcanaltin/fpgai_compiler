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

## Continuation — source connectivity and independently executable C1 gate

**Scope verification:** GitHub's source-commit compare from the user's original C1/C2 result revision `637f2745721b32d6f5a7e02c4659b0623bc9eee7` to research cleanup head `434737ba4d50b89d897be97abcd834a5411b8359` contains **28 changed paths over 41 commits**: **14 newly added files, 14 modified files, zero deletions or renames**. This baseline includes the earlier 19-file non-destructive cleanup plus the additional research source-map and E1 validation work. A subsequent commit adding this ledger update will advance the head again. No original R3/R4 local source/native case receipts were modified, and no GitHub Actions/workflows were created.

| Newly added or updated path | Owner/import and behavioral impact | Testing/reporting state |
|---|---|---|
| **ADDED** `scripts/audit_source_inventory.py` | Read-only Git-tracked file inventory; uses Python `ast` to map module imports, direct importers, declared top-level symbols, subsystem owners, byte hashes and duplicate-byte candidates. No production compiler import, code deletion or dynamic execution. Create-only ignored `build/` inventory is optional; CLI summary otherwise read-only. | New local test required; source inventory is a static **approximation**, not a whole call graph |
| **ADDED** `tests/test_audit_source_inventory.py` | Synthetic Git fixture checks actual static importers, relative imports, clean/dirty worktree, SHA export, syntax errors, symlinks and non-Git rejection | Six source-only tests added, **not author-workstation-run** |
| **ADDED** `src/fpgai_research/experiments/cst_paper_c1_heldout_cpp_bridge.py` | NEW source E0→actual emitted E1 study. Reads source/SHA-locked original 54-profile C1 report; re-generates same canonical bounded source events under current separate source revision; uses **existing** `emit_allocated_fanout_cpp` and `verify_allocated_gcc` to compile+run 54 matched disjoint/reused actual host-GCC sources per layout (108 executions planned). Verifies actual scratch address witness and all intermediate/final NumPy numeric output; clean Git source gate, create-only per-case C++ and stdout SHA manifest, fail receipt | **Not yet run**; existing R2h/R2r source and E0 original results untouched |
| **ADDED** `tests/test_cst_paper_c1_heldout_cpp_bridge.py` | Synthetic original C1 E0 population and SHA checks, stale source event mismatch, optional genuine GCC two-profile scalar/mixed-`Phi` paired smoke, output hash drift, failure preservation | Six targeted tests created; genuine GCC tests run only when `g++` installed, **not author-run** |
| **ADDED** `paper/compiler_ir/C1_E0_E1_HELDOUT_CPP_BRIDGE_GATE_20261106.md` | Standalone preregistered E0→E1 science protocol, accepted/negative results, exact original cohort digests, staged safe `--quick` and full output names | Protocol only, **no experiment evidence** |
| **MODIFIED** `Makefile` | Adds `research-inventory` (read-only), `research-audit-tests` (source tests) and `research-c2-review` (SHA-pinned read-only 324-record CPU postanalysis). No Vitis/Vivado or Actions targets | Make parsing/targets need author `make -n` or direct execution |
| **MODIFIED** `CONTRIBUTING.md` | Explains code owner/location discovery before adding/moving modules and requires per-file dependency/test impact report in PR | Documentation only |
| **MODIFIED** `MANIFEST.in`, `scripts/prepare_public_preview.py`, `docs/README.md`, `docs/RESEARCH_PUBLICATION_STATUS.md` | Explicitly include new source-inventory utility in curated compiler release preview/sdist; index source inventory and unverified C1 E1 bridging protocol, without placing internal paper journals in the normal compiler preview | Link/package preview checks still need local execution |
| **MODIFIED** `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md`, `paper/compiler_ir/FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md`, `paper/compiler_ir/README.md` | Record earlier separately verified R2r emitted-C++ source mutation coverage and distinguish **new, unexecuted** matched 54-profile C1 source-to-GCC bridge from original 486 abstract traces | No new hardware or source GCC measurements claimed |
| **MODIFIED** `paper/compiler_ir/R3I_ISOLATED_ALLO_CAPABILITY_RUNBOOK_20261104.md` | Correct dated external-pinned Allo experiment working-directory examples from obsolete separate `fpgai_research` checkout to active `fpgai_compiler` branch. Isolation, pinned Allo SHA and `NOT EXECUTED` gate preserved | External verifier still **not executed**; unsupported code/imports are not weakness evidence |

**Missing next evidence:** Author-workstation targeted pytest for new scripts, `make research-inventory`, `make research-audit-tests`, `make release-links`, C2 original SHA-pinned read-only analysis, two-profile real GNU GCC E1 smoke and independent read-only audit. If those pass, a separate **new output** full 54-profile E1 trial can be considered. A full original author-workstation `pytest -q` and original source rights/history review are required before actual release. The paper remains `MAJOR REVISION` until external semantic comparison and proof scope are independently reviewed.

## Current 6-Nov direct-CLI defect and scientific novelty correction

**Author workstation supplied facts:** The author pulled source HEAD `d0b72b4a00309ff01f8f159d035390c43c35789c`; source-only targeted suite **55/55 passed in 2.55s**; the read-only inventory reported **894 tracked files, 718 modules, 0 Python parse errors, 0 byte-duplicate groups, clean tracked tree**. The subsequent `make release-links` invoked `python scripts/check_public_docs.py --profile compiler` and failed with **`ModuleNotFoundError: No module named 'scripts'`**. Because the group used `set -e`, subsequent original C2 read-only CPU review and C1 real-GCC E1 smoke were **NOT run**. This is a direct-script sys.path issue and not evidence of a C1/C2 scientific fault.

| File changed | Dependency/why and future effect | Validation at time of commit |
|---|---|---|
| `scripts/check_public_docs.py` | Switch to `from prepare_public_preview ...` only when `__package__` is empty (direct script mode); keep `from scripts.prepare_public_preview ...` under module import. Supports both documented `python scripts/...` and module `python -m scripts.check_public_docs`. No added `sys.path` hacks, no user data or vendor calls. | Implementation committed, **new direct CLI not yet author-workstation-verified** |
| `Makefile` | `release-links` now uses `$(PYTHON) -m scripts.check_public_docs --profile compiler` so the parent source checkout is a reliable Python import root. All existing source targets retained; no GitHub Actions. | Await new workstation `make release-links` test |
| `tests/test_public_docs_links.py` | Adds real subprocess regression of both CLI invocation modes with identical `public_release_authorized=false` receipts; targets the author-reproduced error. | New test unverified; prior 55/55 targeted tests used the **old** test version |
| `paper/compiler_ir/SCIENTIFIC_REVIEWER_ADVERSARIAL_CLAIM_MATRIX_20261106.md` (ADDED) | Falsifiable A–G semantic controls, value/declared `Phi` order/data hazard distinctions, fair Allo/CODO external capability classifications, C2 CPU/call budget and strict physical latency gate | Preregistration, **not a performed benchmark** |
| `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md` | Corrects prior-art claim in favor of **Allo PLDI 2024 Section 5.2/Fig. 6**: existing Allo detects a two-MatMul FIFO producer/consumer order mismatch. Narrows any novel remainder to combined per-reader lifetime, scratch alias, operator-local `Phi` and actual emitted-code evidence, subject to external comparison. | Evidence synthesis based on public published paper/tutorial; **external Allo environment not executed** |
| `paper/compiler_ir/FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md` | Adds explicit C1 novelty STOP rule and C2 possible downgrade to enabling compiler infrastructure; no firstness or unverified E4/latency claim. | Editorial, all original source evidence retained |
| `paper/compiler_ir/README.md` | Indexes new review/falsification matrix, distinguishing protocol from verified original experiment | Editorial |
| `paper/compiler_ir/R3I_ISOLATED_ALLO_CAPABILITY_RUNBOOK_20261104.md` | Updates pre-existing pinned Allo comparison instructions with true PLDI 2024 FIFO mismatch example and separate fair capability dimensions, without claiming an Allo weakness from a missing/failed import | External verification remains **not executed** |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | Adds public-safe explicit Allo overlap warning and links adversarial reviewer challenge matrix | Documentation only |
| `CHANGELOG.md` | Preserves exact traceback, source pass counts and novelty overclaim correction | Documentation only |

**Scientifically decisive next evidence (in order):** (1) `make release-links` and the added direct script regression; (2) unchanged source original C2 `make research-c2-review` — interpret all 324 actually saved CPU timings including eager losses; (3) staged new C1 real E1 GCC smoke with both scalar/mixed source maps against original 54 E0 SHA-anchored population; (4) if successful, entirely new 54-profile/108-GCC original E1 source refinement; (5) isolate/pin and actually execute matched external Allo check; (6) independently assess whether an operator-local `Phi`/per-reader lifetime/actual scratch ownership contract is nontrivially distinct from Allo/CODO. Source experiment pass counts alone cannot close these P0 gates. Strict original routed E4 and physical latency remain null/unknown.

**No change in access/security:** Main branch unchanged, PR #2 stays draft, no source/native evidence deleted/overwritten, no native Vitis/Vivado reruns, no GitHub Actions, no license/rights authorization. Exact latest head and future code/test outcomes must be reverified in the author's checkout rather than inferred from a GitHub commit.
