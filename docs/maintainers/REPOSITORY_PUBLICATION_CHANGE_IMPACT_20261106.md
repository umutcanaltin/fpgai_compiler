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

## Full 54-profile original C1 E1 results and read-only effect-size tool — latest change inventory

**Original author workstation evidence (new immutable study):** source revision `3bd1e36ff49864d89fa4c676e61da8ba0b936fce`, 17/17 focused tests passed, original E0 report SHA `8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978` verified read-only, **54/54 source profiles**, **108/108 actual GCC C++ compilations/executions**, **54/54 positive source slot saving**, **271** independently hash-audited C++/stdout/case/summary files, full manifest SHA `a79e700208e5d7cada93dd1f488809b6622f52cf6b14fa7b7d7b05a8d624d83e`. No source compiler/vendor call in the audit, no qualified physical clock or board latency. Original output `build/paper_c1_e1_cpp_bridge_full_20261106_01` is **FROZEN**.

| Added/changed file | Responsibility and dependency impacts | Current verification |
|---|---|---|
| **ADDED** `paper/compiler_ir/C1_FULL_54_PROFILE_E1_GCC_ORIGINAL_RESULTS_20261106.md` | Author exact full C1 original SHA provenance, E0/E1 denominators, 54/54 source slot signs, distinct E1 actual GCC witness and no-physical-latency claims. Points to the existing original generated-source emitter/validator and earlier R2r mutation evidence. | Original user-provided stdout and separate original `--study` audit support the cited values. |
| **ADDED** `src/fpgai_research/experiments/cst_paper_c1_original_e1_readonly_descriptives.py` | Read-only analysis imports **existing** `cst_paper_c1_heldout_cpp_bridge.audit`, same E0 SHA, and exact original E1 manifest SHA; verifies 54/108/271 original row population and 3×3×2×3 source strata; reports source slot/numerical/address-witness summaries and per-profile SHA relationships. **No new compiler/GCC/Vitis/Vivado calls, no original file modifications.** | New source-only analyzer **NOT YET workstation-tested**; do not claim exact slot distributions until verified. |
| **ADDED** `tests/test_cst_paper_c1_original_e1_readonly_descriptives.py` | Full 271-artifact synthetic SHA fixture and negative mutation tests covering manifest drift, source tampering, duplicate profiles, invalid slot deltas, witness counts, NaN/inf numerical error, read-only preservation. Uses existing source verifier but no GCC/FPGA calls. | Synthetic test suite **NOT YET workstation-tested**. |
| **MODIFIED** `Makefile` | New explicit `research-c1-e1-review` runs only SHA-pinned full-original read-only analysis; `research-audit-tests` adds its targeted regression. | New target **NOT YET workstation-tested**. |
| **MODIFIED** `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md`, `paper/compiler_ir/FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md` | Active abstract, implementation evidence and submission gate now use *actual* full 54-profile/108-GCC E1 source evidence. Preserve original failed routed R4 BRAM/LUT/FF/WNS caveats and negative C2 lazy-vs-eager result. | Editorial; no new native/hardware results claimed. |
| **MODIFIED** `paper/compiler_ir/C1_E0_E1_HELDOUT_CPP_BRIDGE_GATE_20261106.md`, `paper/compiler_ir/SCIENTIFIC_STRENGTHENING_C1_C2_PROTOCOL_20261106.md`, `paper/compiler_ir/README.md`, `docs/RESEARCH_PUBLICATION_STATUS.md` | Supersede current-facing `full 54 pending` labels while clearly retaining historical pre-fix checkpoints and original two-profile smoke chronology. Index full original report and new separate read-only analyzer. | Documentation only, original E0/E1 outputs unchanged. |
| **MODIFIED** `CHANGELOG.md`, `docs/maintainers/REPOSITORY_PUBLICATION_CHANGE_IMPACT_20261106.md`, GitHub PR #2 | Track all added/updated/deleted files, ownership and import callers. | PR stays DRAFT, `main` untouched, no Actions. |

**Scientific caveat:** 486 E0 traces contain nine selected event mutations per original static model; the 108 E1 source runs cover only *canonical* emitted source under two scratch layouts, NOT all E0 mutation traces under generated code. Source slots are source units, not BRAM tiles or hardware latency. Allo/CODO matched semantic coverage, clean-DRC timing-qualified physical evidence, publication rights and full repository release tests remain open.

## Postrun science release update — verified negative C2 and original two-profile C1 E1

**User workstation facts:** At revision `5bc8e9c1ec749c1c3dcf09bf85bace91503ee0c7` the author passed **30/30** focused tests (1.60s), verified **168/168** public relative links across 31 documents, and created a **472-of-895 tracked file** nonpublishable compiler preview plan. `make release-links` import-path issue is **FIXED/VERIFIED**; no legal release approval. The frozen source revision `637f274…` C2 benchmark report SHA `be2adaa4acf30196a2ccb17114f2caa0af3e6679dc2685872e27cb086ca86ec1` was independently analyzed read-only: **324 method rows / 108 paired conditions**, lazy CPU faster than eager in **108/108**, with original compiler calls 198 lazy, 864 eager, 1476 direct, 48 separate excluded preflight. The original 2-profile E1 smoke at new revision `5bc8e9c…` had 4 actual GCC compilations/executions and 11 SHA-attested artifacts, positive source scratch savings 2/2; the 54-profile full E1 remains **unexecuted**.

| Changed file or dependency | Purpose and connectivity | Runtime verification |
|---|---|---|
| **ADDED** `paper/compiler_ir/C1_E1_SMOKE_C2_SOURCE_COST_ORIGINAL_RESULTS_20261106.md` | Full source-locked six-condition descriptive CPU timing table, complete per-method transition counts/paired wins, provenance and original C1 E1 two-profile SHA audit; links active paper and source-latency limits | Values from author stdout; no new original source/compiler/vendor calls |
| **UPDATED** `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md` | Abstract/table/C1 E1 discussion now use actually observed lazy-vs-eager negative CPU and 4 authentic GCC results instead of `unanalyzed`/`unexecuted` placeholders; C2 may be enabling infrastructure | Editorial only; source and original reports unchanged |
| **UPDATED** `paper/compiler_ir/FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md` | Frozen SHA-backed provenance, honest lower C2 utility claim and full 54 E1 still-open gate | Editorial only |
| **UPDATED** `paper/compiler_ir/C1_E0_E1_HELDOUT_CPP_BRIDGE_GATE_20261106.md` | Stages already passed original 2-profile g++ source smoke separately from next new 54/108 run; original SHA manifest and source revision are recorded | Smoke and audit verified at author workstation |
| **UPDATED** `paper/compiler_ir/SCIENTIFIC_STRENGTHENING_C1_C2_PROTOCOL_20261106.md`, `paper/compiler_ir/README.md`, `docs/RESEARCH_PUBLICATION_STATUS.md`, `CHANGELOG.md`, GitHub draft PR #2 | Remove stale method-status claims, expose negative C2 results, register measured E1 source evidence and next scientific checks | Documentation/index only; compiler behavior unchanged |

**Non-destructive change policy:** No original E0 or C2 output was overwritten, source E1 smoke is immutable and full output must be a new path. No GitHub Actions, Vitis, Vivado, RL training or original R4p/R4u reruns; public license/rights review remains unresolved. Strict timing-qualified E4 and real board latency remain UNKNOWN. A repeated compile/run test with synthetic fixtures cannot be pooled with original signed source studies.

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

## Verified original R4w new-Vivado processes; postrun audit repaired without touching evidence

**Observed in author's pasted workstation log:** after pull to source SHA `a0d4db117247f08b0c85849f3fbd5d8c0cdac34a`, 40 targeted tests passed, original C1 full 54-profile E1 read-only numerical source-slot report succeeded, original source-only Vitis+Vivado cross-stage SHA join succeeded. Six newly staged R4w Vivado processes returned exit=0 (full/row at 16, 20, 24ns) with no original R4p/R4u output overwrite. The original postrun `make research-r4w-audit` failed in `audit_prepared` before parsing physical reports with **`R4w unexpected RTL/ROM files under source root`**, and `make research-r4x-feasible-latency` failed identically. **No six-route physical qualification, latency winner or DRC status is yet established.**

| Change | Reason, deps and impact | Verification |
|---|---|---|
| `src/fpgai_research/experiments/cst_paper_r4w_feasible_clock_sweep.py` MODIFIED | Original prep SHA manifest requires immutable RTL/ROM and Tcl; postrun Vivado executes `cd $case_dir/rtl`, generating `.Xil` and journal/usage output files not in source manifest. A new `_valid_native_attempt_lock()` allows benign **non-design** sidecars *only* under native workdir after exact immutable attempt lock. Any extra `.v/.sv/.dat/.mem/.xdc/.tcl/.xci/.dcp` design asset, symlink, additional canonical source, changed expected source hash, missing source or lock still rejected; `_case_native` performs native receipt and report SHAs independently. No changes to vendor invocation or classification semantics. | Fix committed; new tests unexecuted locally. |
| `tests/test_cst_paper_r4w_feasible_clock_sweep.py` MODIFIED | Adds 4 synthetic positive/negative postrun cases: `.Xil`/journal/usage sidecar permitted only after immutable lock, extra HDL/ROM/constraint/IP refused, canonical source extras refused, symlink refused. All 6 original actual outputs remain untouched. | New tests pending. |
| `paper/compiler_ir/R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md` MODIFIED | Records six new native exits and explains audit checker failure as distinct from routed feasibility; no new WNS/DRC/physical latency asserted. | Editorial only. |
| `paper/compiler_ir/C1_FULL_54_PROFILE_E1_GCC_ORIGINAL_RESULTS_20261106.md` MODIFIED | Fills original 54-profile E1 effect size after the successful 40-test source audit: median 15.5 slots, 50% per-profile savings, max errors ~2.4e-7, 6 reader/Φ strata. Not mapped BRAM. | Source original post-run audit verified; no new vendor calls. |
| `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md` MODIFIED | Separates validated source-level E1 effect sizes from six native exit-0 processes without physical audit/latency, so claims don't confuse source/gcc and RTL results. | Editorial only. |
| `docs/RESEARCH_PUBLICATION_STATUS.md`, `CHANGELOG.md`, `docs/maintainers/REPOSITORY_PUBLICATION_CHANGE_IMPACT_20261106.md` MODIFIED | Maintainer-facing state, bug cause, file/impact dependencies and no-rerun rule. | Editorial only. |

**NEXT SAFE CHECK:** fetch branch; run `python -m pytest -q tests/test_cst_paper_r4w_feasible_clock_sweep.py tests/test_cst_paper_r4x_feasible_only_core_latency.py`; then `make research-r4w-audit` and `make research-r4x-feasible-latency`, all read-only. Stop on a failed test/audit and report the exact per-case status. **Never rerun the six vendor cases or remove the one-time attempt lock**; all original report files and hash receipts remain untouched.

## New valid-only Vitis/Vivado hardware and latency study — full change-impact ledger

**User's scientific requirement:** compare **VALID timing-qualified** full versus row-storage designs and actual latency (core + end-to-end), not merely source scratch units, Vivado BRAM, HLS cycle counts, or routed WNS. The already completed original R4p 12ns results have negative WNS in both arms; they are frozen descriptive negative evidence, NOT valid hardware performance wins. This new code is an unexecuted new pilot. The previous original `cst_paper_c1_original_e1_readonly_descriptives` CLI test failure was corrected by passing the existing immutable manifest SHA into `analyze()` explicitly; no original experiment or report hashes were changed, and the fix requires the author's local retest.

| File | Added/modified | Owner/dependencies and change consequence | Local verification state |
|---|---|---|---|
| `src/fpgai_research/experiments/cst_paper_r4v_vitis_vivado_joint_compare.py` | ADDED | Joins existing original R4l HLS E2/E3 SHA and R4q/R4r source-attested Vivado routed resource/DRC/WNS hashes; computes clearly labeled E3-only latency proxy, retains `physical_E4=null` | NOT workstation tested; zero native calls by design |
| `tests/test_cst_paper_r4v_vitis_vivado_joint_compare.py` | ADDED | Synthetic paired original HLS/Vivado receipts and invalid source SHA/resource/DRC/timing/cycles negative checks | NOT workstation tested |
| `src/fpgai_research/experiments/cst_paper_r4w_feasible_clock_sweep.py` | ADDED | Original `r4p._original_r4o` exact Vitis-RTL/ROM/source evidence preflight; creates six NEW full/row isolated Vivado candidate cases for 16/20/24ns at 1.5ns uncertainty; source/ROM cross-copy SHA, literal Tcl constraints, create-only summary and manifest, **one-time opt-in native tool lock**; parses postroute WNS/DRC/resources and reuses `toolchains/vivado/advisory_observation.py` for recognized noncritical DSP advisory detection WITHOUT E4 promotion | New code only; no vendor calls or author tests yet |
| `tests/test_cst_paper_r4w_feasible_clock_sweep.py` | ADDED | Synthetic original R4o source and ROM, six create-once clocks, TCL and source drift checks, native dry run and permanent attempt-lock rejection; mocks original signed evidence, not a real vendor result | NOT workstation tested |
| `src/fpgai_research/experiments/cst_paper_r4x_feasible_only_core_latency.py` | ADDED | Read-only original HLS E2/E3 + new R4w native qualification reader. In fixed equal-clock pairs only, **strict routing + WNS>=0 + zero DRC** allows an HLS cycles × clock source time proxy. No pair = no winner/latency claim; actual core, board, DMA, throughput remain NULL. No vendor call | New code only, not workstation tested |
| `tests/test_cst_paper_r4x_feasible_only_core_latency.py` | ADDED | Synthetic matched clock valid pair, one arm invalid, original SHA/cycle mismatch, incomplete evidence, no-proxy gates | NOT workstation tested |
| `paper/compiler_ir/R4V_VITIS_VIVADO_STAGE_PARETO_AND_REAL_LATENCY_GATE_20261106.md` | ADDED | Exact existing original 22,954/22,602 cycles and 7/4 routed BRAM source-PPA table, negative WNS, HLS latency estimate and no E4; corrected DPIP/DPOP as **DSP pipelining**, not RAM-port errors | Documentation only |
| `paper/compiler_ir/R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md` | ADDED | Reproducible dry-run/prep/one-time vendor/audit commands, strict valid-design criterion, narrow DSP advisory diagnostic, proposed 2×2 storage × DSP-pipeline intervention and measured core/board/DMA latency protocol | Protocol not original result |
| `Makefile` | MODIFIED | Adds `research-hls-vivado-review`, `research-r4w-audit`, `research-r4x-feasible-latency`, includes new synthetic tests in `research-audit-tests`. These three research targets READ ORIGINALS, do not call vendors; NEW native pilot executable only via explicit two-flag module CLI after create-only preparation | NOT workstation tested |
| `src/fpgai_research/experiments/cst_paper_c1_original_e1_readonly_descriptives.py` | MODIFIED | Corrects reported summary-only **test-only monkeypatched manifest digest** by explicitly passing constant into `analyze()`, preserving original source SHA policy | Failing original test observed by author, fix NOT yet retested |
| `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md`, `paper/compiler_ir/README.md`, `docs/RESEARCH_PUBLICATION_STATUS.md`, `CHANGELOG.md`, `docs/maintainers/REPOSITORY_PUBLICATION_CHANGE_IMPACT_20261106.md` | MODIFIED | Explicitly separate previously verified source GCC E1 from independently original Vitis/Vivado PPA, negative 12ns results, unexecuted candidate clock pilot, actual board time still UNKNOWN; updated dependency/change map | Editorial only, not new hardware evidence |

**No deletes/renames, no production compiler arithmetic/scheduling changes, no original R4k/R4l/R4p/R4u vendor replay, no GitHub Actions, no merge to main.** The original DPIP/DPOP are warning-only **DSP input/output pipelining recommendations**, so any design fix should be a new controlled HLS DSP-pipeline variant (and a matched full storage control), not a fictitious RAM-collision 'fix' or silencing warnings. The original R4 12ns timing is still negative even if performance-only warnings were deemed acceptable. Actual inference timing will require safe routed designs integrated into bitstreams and measured on a qualified board; a HLS-cycle × target-period product is **never a measurement**.

## 6 November 2026 — R4w actual results reconciliation and R4x read-only sensitivity extension

| Modified path | Ownership, callers and affected interface | Verification status and impact |
|---|---|---|
| `src/fpgai_research/experiments/cst_paper_r4x_feasible_only_core_latency.py` | Existing experiment analyzer and CLI; calls original R4l/R4w source-attested audits; extends JSON with separate non-promoting advisory-only timing sensitivity and routed resource fields, plus one-time attempt lock/DRC/clock guards. No change to strict E4 gate or null measured latency. | New source changes not yet workstation tested; no compiler/vendor invocation. |
| `tests/test_cst_paper_r4x_feasible_only_core_latency.py` | Existing synthetic regression owner; adds both-arms timing-only checks, conditional 20/24ns arithmetic, no-mutation and nine forged-source/qualification checks. | New tests not yet workstation-run. The 32 original passing tests preceded this edit. |
| `paper/compiler_ir/R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md` | Research methods and original 6-run results; includes the warning-only descriptor, 20/24ns conditional sensitivity and explicit non-hardware limits. | Original author console log; original native artifact data untouched. |
| `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md` | Active two-contribution manuscript; corrects stale native-pending/audit-failed status, claims only verified timing and no E4 success. | Editorial; first paper still MAJOR REVISION. |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | Public research entrypoint; corrects outdated audit-failure claims and separates advisory sensitivity from physical evidence. | Editorial, no release authorization. |
| `CHANGELOG.md` | Chronology of author's real R4w results and new source changes. | Editorial. |
| This maintainer impact file | Tracks every changed path, imports, evidence relationships and tests. | Editorial. |

**Added/removed/renamed paths:** none. **Original R4k/R4l/R4o/R4p/R4u/R4w original study outputs and locks modified:** none. **New GitHub Actions:** none. **Main branch:** untouched. **Outstanding:** independent local source tests, zero-DRC feasible matched designs, measured RTL/board core cycles/II, board DMA timings, 2×2 DSP pipeline study if performance claims pursued, external novelty comparison and rights approvals.

## 6 November 2026 — conservative-clock planning amendment

| Changed path | Owner/interface and connectivity | Verification / outstanding gate |
|---|---|---|
| `src/fpgai_research/experiments/cst_paper_r4x_feasible_only_core_latency.py` | Extends existing original R4l+R4w SHA-attested read-only `qualify()` JSON with `prospective_clock_planning_NOT_physical_qualification`: minimum paired WNS≥1.0ns and fastest surviving candidate, 24ns. Existing primary strict E4/board null outputs and native calls unchanged. | New heuristic is **post-hoc**, not safety proof. New source-only code not yet author-run. |
| `tests/test_cst_paper_r4x_feasible_only_core_latency.py` | New synthetic regression cases for matched margin, 24ns selection, absent eligible cases, no clean-DRC promotion; reuses existing fixtures. | Tests not yet author-workstation-run. Prior verified **43 tests** belong to earlier revision `066a57c`. |
| `paper/compiler_ir/R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md` | Extends first-paper method with independent future all-path/hold/DRC/bitstream validation and prospective 2×2 24ns source experiment/3-graph generalization. | Preregistration proposal only, no new Vitis/Vivado results. |
| `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md` | Active manuscript adds clearly distinct post-hoc clock choice, physical failure gates and keeps C1/C2 novelty central. | Editorial only, draft not submission ready. |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | Public-scoped research status explains 24ns clock selection and constraints. | No release approval. |
| `CHANGELOG.md` | New source/planning and original author 43-pass checkpoint. | Documentation. |
| This change-impact ledger | Tracks all paths, callers and unrun tests; no duplicate experiment implementation. | Updated this commit. |

**Added, deleted, renamed, moved:** none. **Original source/ROM/vendor reports/attempt locks modified:** none. **New GitHub Actions, native executions or main merge:** none. **Further gates:** full source-only regression, signed hold/constraint reports, documented DRC severity, new 2×2 controlled source variations, separate physical/board measured latency and external C1/C2 novelty evaluation.

### R4k/R4o source generator dependency audit

The existing `cst_r4k_matched_row_schedule_storage.py` packages both emitted storage variants with the **original `r4a.CLOCK_NS` / `r4a.UNCERTAINTY_NS`**, and `cst_r4o_matched_vivado_ooc_prepare.py` fixes its historical OOC `CLOCK_NS=12.0`. `toolchains/vitis_hls/package.py` does support explicit package clock parameters, but no verified 24ns DSP-pipeline source generator exists yet. Existing row and full HLS source emitters certify source-event order/scratch edits, NOT hardware DSP pipeline behavior. Correct path is separate create-only 24ns source packages with a deliberate verified pipeline knob, new E2/E3 and fresh E4 receipts; never patch frozen R4k/R4o or infer HLS cycle compatibility from a relaxed Vivado clock.

These are **read-only code inspection findings**, not original experiment observations. Only `paper/compiler_ir/R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md` and this maintainer ledger changed in this small follow-up; **no source, test, native output, or lock** was altered.

## R4y source-only controlled 24ns hardware mechanism preparation — 6 November 2026

| File | Operation | Ownership/connectivity | Local verification status |
|---|---|---|---|
| src/fpgai_research/experiments/cst_paper_r4y_24ns_pipeline_source_prepare.py | ADDED | Experimental pure plan, create-only prepare, SHA/provenance-locked audit. Imports existing canonical full/row emitters, R4k and R4l original source auditors, R4a input and Vitis package API. | Not yet author-run; no GCC/Vitis/Vivado. |
| tests/test_cst_paper_r4y_24ns_pipeline_source_prepare.py | ADDED | Synthetic tests: 4-arm plan, exact pragma mutation, no-source-drift, original SHA mismatch, create-only, symlinks, source/testbench/clock tamper, no vendor calls. | Not yet author-run. |
| paper/compiler_ir/R4Y_24NS_FOUR_ARM_SOURCE_PREPARATION_20261106.md | ADDED | Prospective method, exact four-arm source matrix, commands and E2/E3/E4 evidence limits. | Documentation. |
| Makefile | MODIFIED | research-r4y-plan/prepare/audit and new regression inclusion in research-audit-tests; reuses canonical R4 source root variables. | Source-only, not yet author-run. |
| paper/compiler_ir/R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md | MODIFIED | R4y cross-reference for source intervention, no older original run rewrites. | Documentation. |
| CHANGELOG.md | MODIFIED | R4y source-only historical checkpoint and verification status. | Documentation. |
| docs/maintainers/REPOSITORY_PUBLICATION_CHANGE_IMPACT_20261106.md | MODIFIED | This owner/connection/dependency audit for every changed path. | Documentation. |

No files deleted, moved or renamed; no previous HLS/RTL generator changed. Original original-source experiments and one-time locks untouched. Draft PR remains open and main unchanged. New 24ns native Vitis/Vivado work is UNIMPLEMENTED and original 47-pass tests predate these changes.

**R4y serialization follow-up:** updated the new R4y source-only auditor to compare JSON-normalized canonical source metadata, because Python tuple source-slot maps become JSON lists on disk. This fixes a potential false-negative on unchanged manifests. Added a synthetic prepare→audit regression test and updated the R4y paper protocol. No other compiler semantics, original study roots, source hashes, vendor calls or test labels changed. New tests still need workstation execution.

## 6 November 2026 — R4y actual author failure and repair file impacts

User's first post-implementation local run: **44 passed, 6 failed**, all from legitimate prepared-status being compared to proposal-status in R4y read-only audit. New real source preparation then **failed before allocation** on a mistaken R4k-only `source_only_E1_verified` requirement applied to R4l `audit_native`. The subsequent new source audit had no directory to inspect; no source native results exist. Corrected only the R4y read-only source implementation and test fixtures; the older 47 passing tests were on pre-R4y code.

| File | Operation | Connection and exactly what changed | Current verification |
|---|---|---|---|
| `src/fpgai_research/experiments/cst_paper_r4y_24ns_pipeline_source_prepare.py` | MODIFIED | Original preflight now uses **actual R4l native audit schema** (E1 matched, 2 arms, E2/E3 qualified counts, original staged SHA, no new/native audit calls) rather than absent R4k-only summary field; audit compares all frozen plan fields except intentional status transition, with separate exact prepared status check; validates model/solver source and package symlinks before reading. | **Pending local test**; no vendor calls. |
| `tests/test_cst_paper_r4y_24ns_pipeline_source_prepare.py` | MODIFIED | Synthetic fixture mirrors real R4l response; adds positive real-schema source package round-trip and negative count/schema/receipt/clock/status/false-E2 controls, preserving existing source SHA and injected-file tests. | **Pending local test**; earlier 6 failures are expected to be addressed. |
| `paper/compiler_ir/R4Y_24NS_FOUR_ARM_SOURCE_PREPARATION_20261106.md` | MODIFIED | Adds author-observed failures, exact cause and preflight repair, source-only rerun commands and evidence limitations. | Editorial grounded in uploaded terminal. |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | MODIFIED | Public-facing short status of failed first R4y preparation and pending fix verification; no premature 24ns native claim. | Editorial. |
| `CHANGELOG.md` | MODIFIED | Chronological R4y repair checkpoint; records 44/6, original failed preparation, no old outputs changed. | Editorial. |
| This repository impact ledger | MODIFIED | Keeps architecture and file history, calls & dependency map complete. | Editorial. |

**No new file or folder additions/deletions/renames in this repair.** The existing R4y source output was NOT created by the author's attempt, and no original R4k/R4l/R4o/R4p/R4w source or vendor result/attempt lock was modified. Research PR stays draft, main untouched, no Actions.

## 7 November 2026 — R4z file-level change/dependency report

**Verified earlier source checkpoint:** author passed 60 focused tests at `dde56d4`, independently prepared and audited 4 source-only R4y arms. R4z has not been vendor executed or author-tested. All original R4w locks, Vitis reports and study output remain unmodified.

| Path | Change | Owner and dependency/connectivity | Verification / evidence |
|---|---|---|---|
| `src/fpgai_research/experiments/cst_paper_r4z_24ns_four_arm_native.py` | ADDED | New R4z study; calls `r4y.audit` to revalidate original SHA roots, stages four source packages, freezes 17 SHA entries, owns permanent one-time vendor attempt lock, optionally calls canonical `run_vitis`, and read-only `summarize_case` / `qualify_case`. Exports `prepare`, `audit_prepared`, `execute`, `audit_native`, CLI. | Source change not yet author-run. All default paths are vendor-free; native requires two flags. |
| `tests/test_cst_paper_r4z_24ns_four_arm_native.py` | ADDED | No-vendor tests for source identity/tamper, original-root overlap, exact 17 files, no-autostart, lock before invocation, interrupted irreversible lock, E2/E3 fail-closed, physical null, matched timing and factorial comparisons. | Not yet author-run, synthetic only. |
| `paper/compiler_ir/R4Z_24NS_NATIVE_E2_E3_FOUR_ARM_PROTOCOL_20261107.md` | ADDED | Main study protocol, author original source checkpoint, commands, scientific acceptance criteria and limitations. | Editorial. |
| `Makefile` | MODIFIED | Registers R4z tests in research-audit-tests and ONLY tool-free `research-r4z-native-prepare`, `research-r4z-native-dry-run`, `research-r4z-native-audit`. | Not yet author-run. |
| `paper/compiler_ir/R4Y_24NS_FOUR_ARM_SOURCE_PREPARATION_20261106.md` | MODIFIED | Completes R4y author source checkpoint and links successor R4z. | Editorial. |
| `paper/compiler_ir/R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md` | MODIFIED | Updates original timing protocol with clearly separated R4z prospective native HLS gates. | Editorial. |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | MODIFIED | Original 60-passed source result and honest R4z pending status. | Editorial. |
| `CHANGELOG.md` | MODIFIED | Chronology of author verified source-only stage versus new unverified R4z code. | Editorial. |
| This maintainer impact ledger | MODIFIED | File ownership, interfaces, tests, original/output scope. | Editorial. |

No existing native compiler/RTL/original study source files modified. No file deletions, moves, renames or GitHub Actions. No main merge. No native tool calls were made by these source commits. Next milestone is the author's local R4z focused regression, create-only native staging and no-vendor audit; real Vitis remains separately opt-in and must preserve adverse outcomes.

## 7 November 2026 — Overleaf scientific manuscript edit and file impact

| Path | Operation | Function and dependency | Verification |
|---|---|---|---|
| `paper/compiler_ir/FPGAI_EXECUTABLE_SOURCE_CONTRACTS_MANUSCRIPT_20261107.tex` | ADDED | Standalone ~103 kB article LaTeX. Literature from existing manuscript, fixed source-contract methods, original SHA-audited C1 GCC 54-profile/108-run effect sizes, original C2 direct/lazy/eager 108 paired source-CPU workloads, adverse HLS/physical routed reports, and strictly planned 24 ns storage × producer-pipeline factorial, external semantic comparator, tested-clock frontier and board latency. No new empirical outputs. | Compiled twice locally with pdfLaTeX in article fallback: **21 pages, no unresolved citations or refs**, 64 unique labels, no missing bibliography keys. Rendered multipage preview inspected. |
| `paper/compiler_ir/README.md` | MODIFIED | Adds discoverable manuscript pointer, publication/evidence boundary and Overleaf instructions. | Editorial. |
| This maintainer ledger | MODIFIED | File addition, dependencies, author-facing access; no generated hardware outputs touched. | Editorial. |
| `CHANGELOG.md` | MODIFIED | Manuscript editorial milestone, source-only science clarification and claim limitations. | Editorial. |

**No original experiment, tool runner, original LaTeX text, native attempt lock, branch base, licensing file or GitHub Action was altered.** All R4z actual native hardware observations remain not executed; R4y remains the last author-verified source-only experiment. Same draft research PR; main remains unchanged. The GitHub-hosted source is an alternate to a direct `main.tex` download for the author. This edit does not issue rights or reuse permission.

## 9 November 2026 — bounded graph-wide Phi issue-order regression

A source-only, prospective checker was added to the **existing** independent Phi source-order reference owner, not a new parallel contract implementation. It computes closed-form MatMul source iteration ranks and owners without importing the source ordering generator or resolver coordinate function. The graph-wide certificate now requires both the existing value/lifetime replay and this additional mapped-read issue check. It distinguishes a value-safe permutation of two MatMul consumer reads from a source-order conforming implementation, and fails closed if a common-mode generator ordering bug is introduced. Scope is **only MatMul READ tokens from a produced intermediate**, within the static rank-two admitted graph family; no scalar/inter-operator schedule, RTL, HLS timing, or actual vendor evidence is implied.

| Changed file | Operation and responsibility | Callers/dependencies and impact | Verification |
|---|---|---|---|
| `src/fpgai_research/model_ir/independent_phi_issue_reference.py` | MODIFIED, appended independently calculated graph MatMul issue-order oracle with typed read population and counterexample reporting | Uses existing graph admission and `GraphEvent` type, but calculates issue rank and owner without `mapped_matmul_order`, source emitter or `resolver.coordinate`; preserves existing two-reader APIs | Local isolated Python syntax compilation; 8,100 synthetic shape/tile/PE/SIMD/serial source-loop comparisons (not repository pytest) |
| `src/fpgai_research/model_ir/graph_phi_contract.py` | MODIFIED, graph certificate requires independent mapped-read issue check after value/lifetime validation | `graph_contract` lazily imports the independent checker, returns exact checked scope/count; `verify_graph_events` remains value-only by design | Static integration reviewed, new repository tests authored but not run here |
| `tests/test_cst_graph_phi_independent_issue.py` | ADDED, independent graph regression (baseline, mixed Phi, read swap, common-mode corruption, fail-closed budgets) | Uses synthetic `sample_graph` fixtures and current source validator; no vendor or GCC calls | Python syntax compilation only; **repo pytest requires local checkout** |
| `docs/maintainers/REPOSITORY_PUBLICATION_CHANGE_IMPACT_20261106.md` | MODIFIED, complete file/dependency and evidence-impact record | No original R3/R4 outputs or locks touched | Editorial |
| `CHANGELOG.md` | MODIFIED, release chronology and bounded source checker scope | No runtime import effects | Editorial |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | MODIFIED, current research source-validation status and explicit non-evidence note | No new native artifacts or original results | Editorial |

**Research boundaries:** This is a compiler-source contract strengthening, not a new empirical paper result or an independent full semantic specification; graph admission remains shared. No GitHub Actions, Vitis, Vivado, native cases, original source receipts, or draft PR merge. Existing source-revision-pinned experimental claims are unchanged. The full targeted and full repository test suites remain **not executed** by this connected-GitHub-only change.


## 10 November 2026 — public Markdown link regression repaired

**Reported original user verification:** On the research branch, `python -m pytest -q` completed with **2,351 passed, 3 skipped and 2 failed**. The failures were `tests/test_public_docs_links.py::test_direct_script_and_module_invocations_match_from_checkout` and `tests/test_repository_publication_hygiene.py::test_compiler_preview_markdown_links_resolve_without_internal_paper_files`. Both independently reported the identical single unresolved link `CHANGELOG.md:77` pointing to `paper/compiler_ir/C1_E1_SMOKE_C2_SOURCE_COST_ORIGINAL_RESULTS_20261106.md`, which is excluded from the compiler-only source preview by policy.

| Path | Change and purpose | Dependencies and impact | Verification |
|---|---|---|---|
| `CHANGELOG.md` | MODIFIED: turned its one historical internal-paper relative Markdown link into an absolute GitHub link targeting the same research-branch file. Added chronology documenting the author's observed failure. | Public compiler preview can resolve every **local** Markdown reference while paper artifacts remain excluded. No change to the linked source evidence or its permission status. | Confirmed the invalid relative target was unique, GitHub repository contains the linked paper file, and the existing checker intentionally excludes `https://` URLs from local-path assertions. Full pytest still awaits author rerun. |
| `docs/maintainers/REPOSITORY_PUBLICATION_CHANGE_IMPACT_20261106.md` | MODIFIED: records source and test impact without modifying any original study. | Documentation tracking only; no compiler modules, APIs, result receipts, native attempt locks or GitHub Actions changed. | Static inspection of current checker and user failure log. |

The fix **does not weaken** `scripts/check_public_docs.py`, alter `scripts/prepare_public_preview.py`, add a new release-file exception, grant a software license or authorize release. Both original failing tests remain in the suite.

## 10 November 2026 — actual GCC Phi issue-proof manifest binding and feature audit

| Path | Operation / owner and effect | Verification / evidence boundary |
|---|---|---|
| `src/fpgai_research/model_ir/graph_phi_contract.py` | MODIFIED: graph bounded source certificate now returns independently calculated MatMul READ order signature. | Existing model admission unchanged; requires local tests |
| `src/fpgai_research/codegen/hls/graph_phi.py` | MODIFIED: includes issue status, checked reads/operators and signature in emitted C++ manifest. | Generated HLS source only; no vendor calls |
| `src/fpgai_research/validation/cpp_graph_phi_trace.py` | MODIFIED: recomputes independent source-order digest from observed GCC execution and fails closed if receipt coverage or SHA mismatches. | Host C++ evidence only, not cycle-/port-level proof; GCC tests pending |
| `tests/test_cst_graph_phi_source.py` and `tests/test_cst_graph_phi_independent_issue.py` | MODIFIED: assert full certificate/manifest/executed source signature agreement and forgery rejection. | Tests authored, not yet locally executed |
| `Makefile` | MODIFIED: `research-audit-tests` includes the two graph Phi source/GCC modules. | Remains explicit source-only command |
| `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | ADDED: real feature coverage, bounded limitations, tests and scientific/legal gates. | Documentation only, not results |
| `paper/compiler_ir/README.md`, `CHANGELOG.md`, `docs/RESEARCH_PUBLICATION_STATUS.md`, this impact record | MODIFIED: implementation pointers, test status and release limits. | Documentation only |

The author previously reported 2,351 passed / 2 documentation failures / 3 skipped, then 13/13 fixed documentation tests and 168 resolved links. Those checks do NOT qualify this new source commit. No original R3/R4 outputs, attempt locks, actions, source results, GitHub Actions, distribution visibility or operative license were changed. Local verification required: `python -m pytest -q tests/test_cst_graph_phi_independent_issue.py tests/test_cst_graph_phi_source.py tests/test_cst_graph_edge_witness.py`; then `python -m pytest -q` and `make release-links`. No claim of full publication readiness or physical speedup.

## 10 November 2026 — user workstation confirmation of source implementation

The author pulled source revision `156778309e92f90d10bf82242dbf57862a228936` and executed:
- Focused graph Phi/GCC tests: `20 passed in 2.38s`.
- Full suite: `2354 passed, 3 skipped in 242.70s` (0 failures).
- `make release-links`: 168/168 checked Markdown references resolve (31 documents).
- `make release-preview-plan`: 478 included / 916 tracked; excludes 438 (no staging/clean-room test).
- `make release-check`: no suspect tracked filename suffixes; not licensed or release cleared. Source distribution, full-history scan, IP/rights/third-party approval, dependency audit, and clean-room runtime remain OPEN.

This is author's local stdout observed in conversation, now recorded in the source impact ledger. It is a **source-only test result**; no new native Vitis, Vivado, Allo, PYNQ board, hardware latency, Fmax, or main-branch merge. No claim of physical qualification or authorization. This documentation change does not alter source behavior, original experiment files, or any compiler APIs.

## 10 November 2026 — staged compiler preview attested; new optional isolated install smoke

**Author-observed stage:** On the research branch at `fdfac755662a8904682c3597f103b20d8f242dda`, `prepare_public_preview.py --profile compiler --output build/public_compiler_preview_20261110_01` created **478 selected files**, `ready_to_publish=false`. Independent `check_public_docs.py --profile compiler --root build/public_compiler_preview_20261110_01` verified manifest file inventory and **168 local links** across 31 Markdown files (0 unresolved). This does not certify package import/install, dependency license, history scan, original research artifacts or publication rights.

| Path | Operation and ownership | Dependency/effect and verification |
|---|---|---|
| `scripts/verify_compiler_preview.py` | ADDED: release-maintainer isolated installed-package smoke helper, read-only by default, explicit `--execute` required | Uses existing `scripts.check_public_docs` staged manifest audit, fresh copied preview, separate `venv` and `pip install .[onnx,test]`, `fpgai-research` CLI source smoke and gallery source smoke with optional GCC. Does not execute vendor tools, modify staged preview, or grant publication approval. New helper not yet locally run. |
| `tests/test_verify_compiler_preview.py` | ADDED: safety/regression for no-install plan, missing/forged manifest and extra files, create-only build output constraints and explicit user install option | Uses only synthetic fixture preview; does not install packages or call native tools. Tests authored, not yet run. |
| `docs/RELEASE_PREVIEW.md` | MODIFIED: verified stage result and explicit opt-in command | Existing preview scripting and 478-file allowlist not modified; checkout-only helper deliberately excluded from distributable preview. |
| `docs/RESEARCH_PUBLICATION_STATUS.md`, `CHANGELOG.md`, `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED: evidence-qualified stage/installation status and release blockers | No algorithm, source contract, original R3/R4 result, native lock or public LICENSE changes. |
| This maintainer change-impact ledger | MODIFIED: file effects and subsequent verification required | Documentation only. |

**Important:** The staged preview is a potential distribution candidate only after rights/legal, full all-ref history/security, dependency-license and clean-room review. This opt-in install workflow only tests selected source entry points, not all possible models and board or RTL constraints. GitHub Actions remain disabled and PR #2 remains draft.

## 10 November 2026 — isolated install smoke verified; curated preview test runner added

**Author-observed positive evidence:** On commit `ab9947f6f7be9c31aa70b7e28b05e45502c3c2f9`, 4/4 `test_verify_compiler_preview.py` tests passed. The user executed the real isolated `--execute` installed preview smoke. The exact staged manifest SHA-256 was `5d95008e19806471184f11071d439621e9e9efac451442ac391d3845e4823c39`; the tool reported `INSTALLED_PREVIEW_SOURCE_SMOKE_VERIFIED` with independent site-packages import, toy source SHA `33ff3faf720d5038506aca63ff8b6d4bef963c634c78ad2565bfeccc7f93b77d`, CNN source SHA `528a5b2c80ac893685f5db17122aae901fbf02107143a72685b56ea7beb4c96c`, all CNN stages lowered and host GCC checks passed. The author did **not** run preview curated pytest, native vendors or an independent rights/security assessment.

| Path | Operation / canonical ownership | Dependencies, effects and testing |
|---|---|---|
| `scripts/verify_preview_test_suite.py` | ADDED: maintainer-only curated pytest runner over an existing isolated venv and staged source COPY | Imports prior staged smoke audit and curated `PUBLIC_TESTS`; checks selected file hashes against original 478-file stage, requires original install receipt and import path inside isolated venv, `--plan` has no subprocess, `--execute` invokes only those 9 pytest modules and deselects exactly the Git-checkout-only direct/module link audit. Writes create-only `curated_preview_pytest_receipt.json`, retains incomplete rights/physical status. **Not executed by author.** |
| `tests/test_verify_preview_test_suite.py` | ADDED: synthetic staged/isolated environment mock test fixture | Verifies fail-closed changed source/receipt, read-only plan, exact deselection and create-only receipt. No real installation/pytest subprocess/vendor calls in unit tests. **Not executed by author.** |
| `docs/RELEASE_PREVIEW.md`, `docs/RESEARCH_PUBLICATION_STATUS.md`, `CHANGELOG.md`, `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED: actual installed-host source observations, separate next curated test gate and bounded interpretation | No compiler, original R3/R4 study, license or GitHub workflow edits |
| This maintainer impact ledger | MODIFIED: preserve file ownership and evidence source revision | Documentation only |

**Original source/experiment effects:** None. The curated tests are intentionally **not added** to the compiler preview allowlist; the runner is a release-maintainer-only checkout tool. The staged preview itself is untouched. Publication remains blocked by operative LICENSE, IP and rights-holder decisions, history/secrets and dependency review, external compiler baseline, missing strict physical/board evidence for general performance claims. PR #2 remains draft; main branch unchanged.

## 10 November 2026 — curated source preview verified, distribution wheel guard added

**Observed original author output:** At branch revision `4e82046667126ae2b8c6ce7d133480669ed1fc25`, `tests/test_verify_preview_test_suite.py` returned **5 passed (0.38s)**. The user then executed `python -m scripts.verify_preview_test_suite --execute` on the already installed preview, whose manifest was `5d95008e19806471184f11071d439621e9e9efac451442ac391d3845e4823c39`; result `CURATED_PREVIEW_SOURCE_TESTS_PASSED`, **65 passed, zero skipped, one checkout-only Git-index test deselected**. A separate `make release-links` passed with **168/168** links. No original R3/R4 study, native tool, GitHub Action or main branch was changed.

| Path | Change and canonical owner | Connections and validation impact |
|---|---|---|
| `scripts/verify_compiler_wheel.py` | ADDED; source-distribution technical artifact gate, read-only default `--plan`, explicit `--execute` | Requires `verify_preview_test_suite` curated receipt; builds a wheel from previously copied source in a create-only ignored workspace; validates exact `src/fpgai_research/**/*.py` contents, metadata, archive-member safety, creates second isolated venv, installs wheel with ONNX dependency and generates checked toy HLS C++. This is no license/release step. **Not author executed.** |
| `tests/test_verify_compiler_wheel.py` | ADDED; synthetic zip tests for expected modules, forbidden paper/weights/tests paths, missing modules, no-write planning and explicit switch | Does not build wheel, network or invoke vendors during tests. **Not author executed.** |
| `docs/RELEASE_PREVIEW.md`, `docs/RESEARCH_PUBLICATION_STATUS.md`, `CHANGELOG.md`, `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED; record original user 65-pass result, staged source SHA, wheel gate still prospective | No compiler library API or frozen source/physical/native evidence altered |
| This maintainer file-impact ledger | MODIFIED; trace dependencies and test scope, prevent duplicate implementation | Documentation only |

**Boundaries:** Wheel build and its checks may download Python build requirements only on deliberate `--execute`. No third-party code ownership, historical secrets/redistribution consent, clean DRC/FPGA runtime or exact same-workload external compiler capability is asserted. Keep PR #2 draft; `main` remains unchanged.

## 10 November 2026 — wheel NOTICE false-positive repair and precise file impact

**Observed:** Author ran 9/9 focused `tests/test_verify_compiler_wheel.py` at `2fb3e10`. Plan passed; actual `--execute` wheel build reached ZIP audit and rejected only `fpgai_research-0.3.0.dev0.dist-info/licenses/NOTICE.md`. The real wheel install/CLI test was not reached. The build's create-only `_01` workspace remains on the author's workstation and must not be overwritten. The prior 65/65 source-preview tests remain attested; no native vendor run or original study mutation occurred.

| Path | Operation and effect | Dependencies and qualification |
|---|---|---|
| `scripts/verify_compiler_wheel.py` | MODIFIED: exact `.dist-info/licenses/NOTICE.md` exception only when original staged NOTICE is allowed, wheel bytes match staged file and `License-File: NOTICE.md` is declared; reject unknown `License-Expression` and all other unreviewed license members. Audit receipt distinguishes NOTICE from operative LICENSE. | Reads `NOTICE.md` from original staged preview and uses existing source population/manifest checks. Does not modify packaging config, model code, original studies, Git history or release permission. New checks require author local rerun. |
| `tests/test_verify_compiler_wheel.py` | MODIFIED: synthetic accepted NOTICE plus unknown, missing, forged/altered, extra license and metadata mutation controls. | No network, native tools or actual wheel installation in unit tests; **new tests not yet executed**. |
| `CHANGELOG.md`, `docs/RELEASE_PREVIEW.md`, `docs/RESEARCH_PUBLICATION_STATUS.md`, `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED: fail-closed qualification status, real failure point and safe `_02` create-only rerun. | Documentation only; explicitly no license grant or "publishable" claim. |
| This maintainer impact ledger | MODIFIED: source/test dependency inventory, impact, and pending verification. | Documentation only. |

**Critical release boundaries:** A wheel can contain a NOTICE.md describing no operative source code license while remaining entirely unlicensed for third-party redistribution. Nothing in the new audit promotes the wheel to distributable. No GitHub Action or branch main merge, no original research evidence alteration, no native HLS/Vivado/board or prior art execution.

## 10 November 2026 — completed compiler wheel smoke, documentation-only attestation

**Author-local evidence (10 November 2026, research branch `2078a6b041e7504e85003173090990d17d5a795f`):** After 19/19 wheel verifier unit tests passed in 0.10s, the author used fresh create-only output `build/public_compiler_wheel_smoke_20261110_03` and completed the wheel audit and installed CLI smoke. The tool reported `WHEEL_CONTENTS_AND_INSTALLED_CLI_VERIFIED` for `fpgai_research-0.3.0.dev0-py3-none-any.whl`: exactly **420 of 420 staged Python modules present**, **426 total wheel members**, **0 unexpected members**, exact staged `NOTICE.md` SHA-256 `3ba6f83964ea9e92d187093534754311b3a2e002e4ec7ff1241ce832be17d20e`, wheel SHA-256 `327d1fdff569726667d0218de07f124061ebf34e1e643153dcdaeba6d5635732`. Wheel independently installed into a new Python 3.12 virtual environment, and the wheel-installed CLI emitted synthetic toy HLS C++ with matching manifest SHA-256 `33ff3faf720d5038506aca63ff8b6d4bef963c634c78ad2565bfeccc7f93b77d`. The staging manifest remained `5d95008e19806471184f11071d439621e9e9efac451442ac391d3845e4823c39`; prior curated-preview test receipt reported 65 passing tests. **No native vendor tools executed, no original studies reproduced, and the NOTICE remains nonoperative as a software license.**

| Path | Operation and dependencies | Verification and impact |
|---|---|---|
| `CHANGELOG.md` | MODIFIED: release-test chronology and exact wheel digest | Author-verified stdout; documentation only |
| `docs/RELEASE_PREVIEW.md` | MODIFIED: successful wheel/NOTICE gate and remaining release blockers | No changes to wheel builder, staged preview policy or source artifacts |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | MODIFIED: technical source packaging milestone versus legal/scientific qualification | No new original experiment claim |
| `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED: claim-to-code readiness matrix with actual wheel proof and bounded scope | No scientific source data altered |
| This impact ledger | MODIFIED: full path/change rationale and source-revision provenance | No compiler code, tests, native tools, workflow or API changed |

`main` unchanged, research PR stays draft, GitHub Actions unused, original studies untouched. This commit only records observed local verification and should not be mistaken for a new full-suite regression.

## 10 November 2026 — broaden bounded C1 NCHW semantic checks (file/dependency audit)

| Path | Action and ownership | Impact / required verification |
|---|---|---|
| `src/fpgai_research/validation/nchw_access_oracle.py` | MODIFIED existing independent NumPy graph-access reference; adds pure arithmetic owner rule for Conv output tile/channel, PE, replica, SIMD reduction lane and serialization fold | References resolved point data and independently produced NumPy index arrays; **does not** import `resolver.coordinate` or the canonical source-event generator. Returns independent owner multiset hash/population and rejects shared coordinate bugs. Test after source revision pull. |
| `src/fpgai_research/model_ir/nchw_graph_contract.py` | MODIFIED existing graph E0 verifier/certificate; checks modeled intermediate-input READ completeness *before* COMMIT, then requires independent E0 NumPy read/owner oracle | Preserves event schema and existing lifetime/alias rejection. Lazy independent validation import avoids a circular runtime dependency. Adds bounded source receipt fields; no new physical storage implementation. |
| `src/fpgai_research/validation/cpp_nchw_conv_graph_trace.py` | MODIFIED GCC-observed Conv source-tap attestation to require independent source/owner oracle verdict and expose exact owner digest/count | Source trace and original C++ hash still required; checks actual Conv intermediate reads, **not** all scalar READ/COMMITs or hardware schedule. GCC test execution is pending. |
| `tests/test_cst_nchw_graph_contract.py` | MODIFIED existing canonical NCHW regression file, not a duplicate test owner | Adds mapped positive/negative owner reference, five mapping-variant corrupt-owner cases, shared common-mode coordinate mutation and three early-COMMIT source-order mutants. Tests are authored, **not yet executed** by connected-GitHub editing. |
| `Makefile` | MODIFIED source-only `research-audit-tests` target to include existing NCHW regression module | Run locally; may invoke real host GCC but never proprietary Vitis/Vivado. |
| `CHANGELOG.md`, `docs/RESEARCH_PUBLICATION_STATUS.md`, `docs/SUPPORTED_FEATURES.md`, `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED engineering/research capability and evidence documentation | New source behavior must not be conflated with earlier author's 2,354-pass source tests, 65 clean-room curated tests or verified wheel of an older source revision. |
| This maintainer impact document | MODIFIED change dependency and scientific limitations ledger | Historical entries preserved rather than rewritten. |

**File linkage and debugging:** `nchw_graph_contract.nchw_graph_events` -> `verify_nchw_graph` -> independent `validation.nchw_access_oracle.independent_nchw_access`; the actual GCC `cpp_nchw_conv_graph_trace.trace_nchw_conv_sources` -> same independent oracle for observed Conv reads. If a logical value-address/owner mismatch appears, inspect the independent NumPy mapping, the canonical `_source_reads`, the local output/reduction tile formula, and the emitted Conv trace printer before touching recorded research artifacts.

**Evidence and branch guardrails:** Original R3/R4 studies, frozen SHA receipts, board native runs and release-preview artifacts unchanged; no new external Allo execution, no Vitis/Vivado, no GitHub Actions, no `main` merge and no approved LICENSE. Full local pytest is a necessary next gate but does not alone establish manuscript submission readiness.

## 10 November 2026 — author-observed NCHW regression and targeted serialization repair

**Observed regression:** After pulling `ce5970b`, the user executed focused, full-suite, research-audit and public documentation checks. Focused: **15 failed / 20 passed**; complete suite: **46 failed / 49 errors / 2297 passed / 3 skipped**; research audit: **15 failed / 158 passed**; public links: **168/168**. Analysis of full user-provided terminal log traced 122 listed FAIL/ERROR entries to `numpy.int64` owner evidence JSON serialization, with the remaining three instances representing the same stale earlier-rejection expectation repeated across runs. The counts must not be silently converted to passes. The previous `ce5970b` source change is not yet regression-qualified.

| Path | Operation / owner | Effects / verification |
|---|---|---|
| `src/fpgai_research/validation/nchw_access_oracle.py` | MODIFIED existing independently calculated NCHW Conv ownership oracle | Constrains integer channel inputs, normalizes NumPy `np.int64` and Python `int` to built-in Python integers, and returns native-int PE/replica/SIMD/fold owner tuples for portable JSON SHA. No change to output owner arithmetic or C1 acceptance for supported valid mappings. New tests pending actual run. |
| `tests/test_cst_nchw_graph_contract.py` | MODIFIED canonical NCHW contract test file | Updates expected removed-intermediate-read reason to `COMMIT_BEFORE_MODELED_INPUT_READS` (strict early rejection), and adds NumPy-int digest stability, invalid channel scalar and separate missing external-sink completeness tests. No safety check is weakened. |
| `CHANGELOG.md`, `docs/RESEARCH_PUBLICATION_STATUS.md`, `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED status/evidence records | Explicitly records failed terminal verification and that subsequent repair is not yet author-qualified. |
| This maintainer impact ledger | MODIFIED engineering file inventory, failure triage and dependency history | No separate duplicate oracle, emitter, CI workflow or native-study mutation introduced. |

**Dependencies:** original source `np.unravel_index` -> `_independent_conv_owner` -> `expected_owner` `Counter` -> `owner_rows` JSON SHA -> `nchw_graph_contract` and actual GCC NCHW validation -> HLS source-stage planners. Focused regression first, then full `pytest`, research-audit and release-links required on the new exact SHA. No Vitis/Vivado, GitHub Actions, `main` merge, original experiment overwrite or licensing/IP authorization.

## 10 November 2026 — successful author regression at NCHW recovery revision

**Verified source revision:** `2689745607d512a4dc2f349baf2075a991d8c952`. The author's terminal output reports `38 passed in 3.80s` for focused NCHW/graph tests; `2395 passed, 3 skipped in 244.87s` for full development pytest; `176 passed in 54.85s` for `make research-audit-tests`; and `168/168` local public documentation links across 31 markdown sources.

| Path | Operation / responsibility | Effect and evidence |
|---|---|---|
| `CHANGELOG.md` | MODIFIED: successful regression timeline | Preserves earlier failures and relates fixed implementation and exact verified source SHA |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | MODIFIED: research evidence tier qualification | Local host E0/E1 verified; never promotes native/peer/release evidence |
| `docs/SUPPORTED_FEATURES.md` | MODIFIED: replace obsolete NCHW "pending verification" wording with author-qualified source-only test numbers | Does not broaden admitted operator or hardware claims |
| `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED: C1 claim-to-code contract gate marked tested on exact source SHA | Peer/physical/rights requirements remain open |
| This `docs/maintainers/REPOSITORY_PUBLICATION_CHANGE_IMPACT_20261106.md` | MODIFIED: required file-level dependency and test documentation | No source code, tests, original experiments, native tools or CI workflow modified |

**Release artifact note:** The author-verifed earlier 478-file stage / isolated tests / wheel originated from source before this NCHW change. A new *nonpublishable* stage must use a fresh ignored build directory and new SHA-based evidence; past successful wheel/build artifacts must remain unchanged. PR #2 remains draft, no GitHub Actions, `main` unchanged, no license/IP authorization.


## 11 November 2026 — strict E4 campaign qualification and explicit mainline merge criteria

**Grounded starting state:** The author's most recent `2689745` source regression passed **2,395 full tests / 3 skipped**, **176 research tests**, and **168/168 public docs links**. The PR #2 research branch was observed 390 commits ahead of `main`, no submitted PR reviews, and still draft. The additional E4 patch and tests below have not been executed at the author's workstation.

| Path | Action and responsibility | Connected caller/evidence and verification impact |
|---|---|---|
| `src/fpgai_research/experiments/suite_native_batch.py` | MODIFIED canonical physical E4 qualification, shared by campaign and independent native E4 audits | Enforces zero total DRC violations and explicit clean status alongside already required CSim, OOC route, WNS, utilization, source/RTL and no-critical/no-unconstrained gates. Prevents warning-only E4 promotion in other callers. Synthetic regression tests added; unrun. |\n| `src/fpgai_research/experiments/reproduction_audit.py` | MODIFIED existing source-pinned campaign evidence aggregator; no parallel E4 classifier | Imports the single canonical `suite_native_batch._qualify` predicate (now extended to require `drc.violations==0` and `drc.clean is True`). Keeps `campaign_complete_for_required_evidence` and `paper_physical_results_complete` conservative, adds per-case E4 blockers and exact strict-E4 count. No vendor execution; tests unrun. |
| `tests/test_campaign_strict_e4_qualification.py` | ADDED one canonical synthetic E4 campaign-regression test owner | Positive balanced eight-case baseline/axes, adversarial zero-critical-but-nonzero DRC, missing/unclean DRC, incomplete routing, no CSim, negative WNS, wrong OOC mode, incomplete utilization and E3-only evidence, plus source-hash tampering. Fixtures do not represent actual executed Vivado cases. |
| `Makefile` | MODIFIED `research-audit-tests` source-only target | Includes focused strict campaign E4 tests; still never invokes native Vitis/Vivado. |
| `docs/RESEARCH_PUBLICATION_STATUS.md`, `docs/SCIENTIFIC_REPRODUCIBILITY.md`, `paper/compiler_ir/FEATURE_TO_IMPLEMENTATION_PUBLICATION_AUDIT_20261110.md` | MODIFIED scientific status and method-scope docs | Prevents E4-from-timing-only promotion; no source evidence, comparator results or board measurements changed. |
| `AGENTS.md` | MODIFIED engineering rule for technical `main` merge | Still disallows draft PR merge and GitHub Actions; requires exact-head verification, full 390-commit scope review, intentional ready-for-review transition and guarded merge. |
| `CHANGELOG.md` and this impact ledger | MODIFIED complete change and dependency trace | No experimental results/RTL/source model/synthetic weights altered. |

**Paper and release guard:** Code-only merge is separate from open-source licensing and scientific acceptance. Main remains unchanged by this patch. Test execution and independent review are required before merging; no authority to claim prior-art novelty, measured board latency, clean DRC in unrun hardware, or IP/redistribution approval.

## 11 November 2026 — 2,409-pass author verification and PR #2 full diff inventory

**Verified user terminal output on code revision `f34f80f93101b80f87fbf88d491db1881322b379`:** focused strict E4/paper 23 pass (0.18s), full regression 2,409 pass + 3 skipped (246.51s), research audit 190 pass (55.02s), doc links 168/168 across 31 Markdown files. Read-only preview 478 selected / 923 tracked / 445 excluded. Rights/history/license gate remains `ready_to_switch_visibility=false`.

**GitHub change inventory:** PR #2 draft, 391 commits ahead/zero behind, no reviews, 154 paths changed: 125 added and 29 modified. Top-level counts: 46 `src`, 44 `tests`, 41 `paper`, 7 `docs`, 6 `scripts`, 3 GitHub community templates and 7 other root-level configuration/docs files. No files deleted or renamed; no PR-added GitHub Actions workflow. All 41 paper changes are outside the separate compiler-only distribution preview. Filepath status is **not** a confidentiality/security/history or rights audit.

| Path | Operation / owner | Effect on dependencies, evidence and next validation |
|---|---|---|
| `docs/maintainers/MAIN_MERGE_READINESS_20261111.md` | ADDED: canonical main-merge risk and technical qualification report | Grounds PR diff size, public/default-branch visibility exposure and verified source tests; identifies must-review paper, compiler, configuration/API and legal/history boundaries. No compiler API or artifact mutation. |
| `CHANGELOG.md` | MODIFIED: actual `f34f80f` successful tests and merge inventory | Supersedes prior "tests pending" only for the exact verified source revision; frozen native studies remain unchanged. |
| `docs/RESEARCH_PUBLICATION_STATUS.md` | MODIFIED: strict E4 source tests now passed and public-release blockers distinguished | Links to maintainer gate; no claim of original Vivado or peer comparisons. |
| This maintainer impact map | MODIFIED: file audit and provenance ledger | No C1/C2 source, test files, R3/R4 result blobs, GitHub workflow or main branch changed. |

**Merge decision:** currently not ready. A complete human review of 154 files, confidential/IP and historical scanning, stable API/config compatibility and deliberate review/ready transition remain necessary before any guarded integration to `main`. Source-code merge permission is not paper scientific acceptance or approved redistribution.
