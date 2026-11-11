# FPGAI PR #2: `main` merge readiness and exact-scope audit

**Audit date:** 11 November 2026
**Candidate research source SHA:** `f34f80f93101b80f87fbf88d491db1881322b379`
**Base branch:** `main`
**Branch:** `research/first-paper-source-memory-bridge-20261106`
**Decision:** **BLOCKED — additional review required before changing the default branch.**
**Scope:** Technical integration only. Merging code is *not* a paper acceptance decision or a redistribution license.

## Author-observed verification at the candidate source revision

| Check | Author's observed outcome | Qualification |
|---|---:|---|
| E4/paper focused tests | 23 passed in 0.18 seconds | Source-only synthetic / existing paper-gate tests; no Vivado |
| Full development checkout regression | 2,409 passed, 3 skipped, 0 failed in 246.51 seconds | This exact research code revision |
| `make research-audit-tests` | 190 passed, 0 failed in 55.02 seconds | Includes strict zero-DRC campaign mutants |
| `make release-links` | 168/168 links across 31 Markdown files | Source-document reference closure only |
| `make release-preview-plan` | 478 selected of 923 tracked; 445 excluded | Read-only inventory, **not** a staged fresh preview |
| `make release-check` | `ready_to_switch_visibility=false` | No approved license/IP/history scan/dependency audit |

The separate **older** 478-file staged preview and successful 420-module wheel were audited at a prior source revision, **not** at this candidate. Do not transfer their clean-room attestation to `f34f80f`.

## Full PR diff and integration exposure

GitHub's live `main...research/first-paper-source-memory-bridge-20261106` comparison at candidate SHA shows:

- **391 commits ahead, 0 behind**, no reported merge conflict; the PR is still **draft** and currently has **no submitted reviews**.
- **154 changed files**: **125 added, 29 modified, 0 deleted/renamed**; 12,656 added lines and 16 deleted lines as reported by the GitHub diff.
- By top-level owner: `src/` **46**, `tests/` **44**, `paper/` **41**, `docs/` **7**, `scripts/` **6**, `.github/` **3** (issue/PR templates, no workflows), and seven other root-level configuration/documentation files.
- The 41 `paper/compiler_ir/` changes include manuscript drafts, mathematical/research claims, original-results interpretation, and native-hardware protocols; they are intentionally **excluded** from the compiler-only preview. Merging the entire PR into an already public default branch may make these files easier to discover. Visibility of the existing public branch does not establish their rights or confidentiality clearance.
- At the **changed-filename level only**, there are no obvious model/weight/checkpoint/bitstream artifacts or GitHub Actions workflow files, and no tracked removals. This is not an all-ref secret scan, semantic content review, private-data clearance, or confirmation that every file is suitable for the default branch.
- Current compiler Python packaging `pyproject.toml` is not in the 154-file PR diff; `MANIFEST.in`, `Makefile`, `.gitignore`, and contributor/release configuration **are** changed and require integration inspection.

## Technical integration gate

The *tested compiler source* passes the author's local tests. Before merging **all 154 files**:
1. Review the exact PR patch and modified public/default-branch surface, especially all 41 paper files, the 46 compiler files, native-study helpers, the build/package rules, and the 44 affected test modules. Resolve any institutional/private-data/review concerns. A filepath scan is not enough.
2. Confirm compatibility with `main` package/import conventions, old CLI behavior, maintainer workflows and source-pinned R3/R4 evidence; perform `git diff --check origin/main...HEAD` and inspect `git diff --name-status origin/main...HEAD` locally. No frozen source/native evidence may be altered or represented as new runs.
3. If a **public default-branch merge** is permitted after human IP/confidentiality assessment, record that approval separately; the local `release-approval.json` must stay ignored/uncommitted and the public repository currently has **no operative software LICENSE**.
4. Run a fresh source-only release preview from **this** revision when reviewing what can be separately distributed; optional isolated install/wheel verification must use a new create-only `build/` path. A preview or successful wheel is a technical check, never permission to redistribute.
5. Examine PR review comments and obtain the owner/maintainer's technical signoff on the final exact SHA. Then intentionally mark PR #2 ready for review and perform a guarded squash/merge only if remaining blockers are explicitly resolved. Do **not** enable GitHub Actions.

**Not required to merge technically scoped, accurately described source:** a full paper novelty acceptance or a clean-board E4 performance claim. Those scientific gates remain separately open and must not be upgraded by a merge.

## Remaining research gates (do not misrepresent)

- The pinned external Allo/CODO comparison lacks actual matched, independently executed peer-tool evidence.
- Original C1/C2 and native results have distinct revisions/cohorts, and still require appropriate independent reproduction/rightsholder decisions.
- The matched R4 physical comparisons lack strict clean-DRC qualified evidence required for timing/latency superiority. No actual qualified board-level inference latency has been demonstrated here.
- Institutional/source ownership, dependency-license, all-ref secrets/history and actual approved source-distribution licensing remain blocked.

## Recommended integration strategy

Keep PR #2 as the canonical source-pinned research history for now. After rights and scope review, prefer a **single reviewed squash merge** to `main` rather than importing 391 granular research commits. If paper files are not cleared for default-branch prominence, first split an independently tested compiler-code-only integration PR and leave paper material on the research branch. **Neither merge path should be automated or inferred from unit-test success.**

**Audit limitations:** This is a repository diff/metadata and owner-provided test-log assessment; the local workstation, full Git history, third-party artifacts and proprietary FPGA tools were not independently run by this audit.
