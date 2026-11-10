# Curated source-distribution workflow

This publicly viewable repository contains **scientific study history and material requiring separate rights and confidentiality review before licensed redistribution**. Simply moving a document from the root to an `archive/` folder, or deleting it from `main`, **does not remove it from Git history**.

For this reason, separately licensed distribution uses a two-stage process:

1. **Prepare and review a clean-room source preview** assembled only from an explicit allowlist of files.
2. **After author/IP/license/history approval**, decide separately how and where to publish. No tool in this repository makes that decision or changes GitHub visibility.

The repository does not currently publish an operative source-code license. A source preview supports technical and rights review; it is not an authorization to redistribute code. See [Licensing](LICENSE_DECISION.md).

## Read-only plan

```bash
python scripts/check_open_source_readiness.py
python scripts/prepare_public_preview.py --plan --profile compiler
```

This reports selected and omitted file counts without exporting code, invoking vendors or contacting GitHub. It cannot prove that the underlying source is free of confidential material.

## Create a non-publishable preview

```bash
python scripts/prepare_public_preview.py \
  --profile compiler \
  --output build/public_compiler_preview_v1
```

The destination must not already exist. Under the research checkout, previews may be created **only inside ignored `build/`**. An optional `--profile full-research` includes the RL implementation, research compiler experiment programs and test Python sources but still excludes historical journals, `paper/` files, GitHub workflows, model weights and Git history. The default compiler-only profile is smaller and may not contain the full research regression suite.

### Audit the isolated preview before installing it

A staged preview has **no Git history**. The documentation checker can
independently read its own manifest and verify **every** expected and
unexpected file, rather than borrowing the parent private repository index:

```bash
python scripts/check_public_docs.py \
  --profile compiler \
  --root build/public_compiler_preview_v1
```

It fails if the preview has a missing file, an unexpected internal file,
a symbolic link, a broken local Markdown link, a mismatched profile,
or a forged claim that publication has been approved.

A successful preview audit is still **not** a legal/IP/security clearance.
Then evaluate the staged files in an isolated environment and repeat the
synthetic CNN/MLP examples using new output directories.

A preview writes `PREVIEW_NOT_FOR_PUBLICATION.json` with `ready_to_publish=false`. **Never upload this automatically**. The preview does not contain a reviewed release LICENSE. It is not an authorized open-source release.

## Required independent release review

Before a separately licensed release or package:

- Obtain written institutional/employer and coauthor/third-party IP authorization.
- Select and approve a license for the exact collection of distributable source files; record authorship and dependency obligations.
- Run a competent **all-ref Git-history** secret/confidentiality review of the existing repository. If history cannot safely be released, use a **new clean repository without inherited Git objects**, after expert review. Rotate any compromised credentials.
- Scan **the preview's content** for private model constants, student/user identifiers, absolute paths, internal URLs, credentials, generated HDL/IP and restricted third-party code.
- Decide whether experimental RL code belongs in the same public project; the compiler-only preview deliberately omits it.
- Install the preview in a new isolated environment, run user examples, compile it and execute the applicable tests. Fix broken imports, missing data and unsupported examples. Record exact versions/results.
- Review native-vendor licensing and forbid public CI from launching licensed HLS/Vivado or uploading private generated source.
- Review public documentation, security policy, contribution rights and final author list; only then authorize any release.

See [Open-source release checklist](OPEN_SOURCE_RELEASE_CHECKLIST.md) and [License decision](LICENSE_DECISION.md).

## Scope of safe current work

Maintaining documentation and preparing a local preview do not change visibility or confer distribution rights. Publishing a licensed release, deleting source-pinned evidence or rewriting Git history requires separate authorization.

### Next step after a clean source checkout regression

The author's 10 November 2026 full-source regression passed 2,354 tests with 3 skipped at source commit `156778309e92f90d10bf82242dbf57862a228936`. The 478-file compiler preview reported by `--plan` was **not generated or clean-room tested**. A safe, create-only local stage and manifest/link check is:

```bash
python scripts/prepare_public_preview.py \
  --profile compiler \
  --output build/public_compiler_preview_20261110_01
python scripts/check_public_docs.py \
  --profile compiler \
  --root build/public_compiler_preview_20261110_01
```

Do not reuse a directory that exists; use a fresh ignored output name instead. The staged audit checks the selected file population, absence of unexpected/symlinked files and relative link integrity. It is **not** a fresh-env dependency installation or a complete clean-room test. Independently install/test a rights-reviewed copy in a separate environment only after examining source and dependency provenance. Neither test grants a license, authorizes publication or changes Git history.

### 10 November 2026 — staged preview verified; independent installation next

The author successfully created `build/public_compiler_preview_20261110_01` from the research branch at `fdfac755662a8904682c3597f103b20d8f242dda`, selecting **478 files**. The staged-preview inventory and relative-link checker independently accepted **168 local links across 31 Markdown files**, with zero unresolved; publication authorization correctly remained false.

This validates the **staged files and relative links**, not a fresh virtual-environment install, actual CLI execution, package dependency isolation, or research reproduction. To perform an optional local installed-package smoke test without modifying the audited preview, use the checkout-only helper:

```bash
python -m scripts.verify_compiler_preview \
  --preview build/public_compiler_preview_20261110_01 --plan

python -m scripts.verify_compiler_preview \
  --preview build/public_compiler_preview_20261110_01 \
  --execute --output build/public_compiler_installed_smoke_20261110_01
```

The default `--plan` checks only the manifest and link closure. **`--execute` explicitly creates a new isolated virtual environment and runs `pip install .[onnx,test]`, which may download third-party dependencies**. It copies the preview into a new ignored build workspace before installation, then invokes the installed compiler from a separate working directory to generate toy and CNN HLS C++ source; if GNU `g++` exists, the CNN synthetic source is also checked against NumPy intermediates. It never invokes Vitis/Vivado or touches original research study outputs. The output receipt is `build/public_compiler_installed_smoke_20261110_01/installed_preview_smoke_receipt.json`. It will never claim rights clearance or release approval, and a failure leaves the workspace intact for debugging. Use a fresh output path if this one already exists.

The helper and its unit tests live in the research checkout and are deliberately **not** part of the 478-file public preview; this is a maintainer release check. After running it, separately review dependency licenses and complete rights/history/security review. Even a successful smoke does not qualify every graph, model or vendor design.

### 10 November 2026 — installed preview smoke verified; curated preview pytest next

The author ran `tests/test_verify_compiler_preview.py` on commit `ab9947f6f7be9c31aa70b7e28b05e45502c3c2f9`: **4 passed in 0.05 seconds**. The opt-in `scripts.verify_compiler_preview --plan` verified the previously staged 478-file manifest (SHA-256 `5d95008e19806471184f11071d439621e9e9efac451442ac391d3845e4823c39`) and 168/168 Markdown links. The author then executed `--execute`, genuinely installing the preview in a fresh Python 3.12 virtual environment and reporting `INSTALLED_PREVIEW_SOURCE_SMOKE_VERIFIED`: the installed `fpgai_research` import came from that virtual environment, the installed CLI generated correct-manifest toy MatMul/ReLU HLS source (SHA-256 `33ff3faf720d5038506aca63ff8b6d4bef963c634c78ad2565bfeccc7f93b77d`), and an independent-architecture CNN/dense pipeline generated source with SHA-256 `528a5b2c80ac893685f5db17122aae901fbf02107143a72685b56ea7beb4c96c`, all stages lowered and actual host GCC intermediate-value verification passed. Zero native vendor calls; neither the full staged-preview pytest selection nor licensing/security review occurred.

A further opt-in checkout-only verifier now runs **all nine curated public preview test modules** with the isolated preview's already installed Python environment and no dependency reinstall. It records one intentionally deselected Git-checkout-only link-audit test: a staged preview deliberately has no `.git` directory. It checks all selected copied source bytes against the original staged files; runtime-generated pip/pytest caches in the copy do not count as source changes. Default `--plan` performs no subprocess/test execution; `--execute` writes a create-only test receipt under the existing isolated workspace:

```bash
python -m scripts.verify_preview_test_suite \
  --workspace build/public_compiler_installed_smoke_20261110_01 --plan

python -m scripts.verify_preview_test_suite \
  --workspace build/public_compiler_installed_smoke_20261110_01 --execute
```

The **new curated test runner has not yet been executed by the author**, and the earlier full 2,354-test development checkout suite is not equivalent to a curated source-suite run from the staged copy. This activity is nonpublishable: no licensing/IP/full-history security review, external compiler comparison, clean DRC physical timing, or verified board measurements have been established.

### 10 November 2026 — curated preview regression passes; wheel check prepared

The author executed the curated regression on the staged compiler-source copy using its isolated Python 3.12 environment: **65 passed, 0 skipped, 1 Git-checkout-only test deselected**. The audit checked the same `478` selected source files against staging SHA-256 `5d95008e19806471184f11071d439621e9e9efac451442ac391d3845e4823c39`. The runner's own five tests passed; `make release-links` again resolved all 168 local links. The `curated_preview_pytest_receipt.json` in `build/public_compiler_installed_smoke_20261110_01` records the exact test scope. This closes the nine-module **staged-source** regression for that revision, not wheel artifact packaging or full research reproducibility.

The next checkout-only, **explicitly opt-in** verification builds a wheel into a new ignored directory, checks that every staged Python package module is included, refuses unsafe archive members and unreviewed payloads, and installs/executes the wheel from a completely new virtual environment. The `--plan` mode does not write files or run subprocesses:

```bash
python -m pytest -q tests/test_verify_compiler_wheel.py
python -m scripts.verify_compiler_wheel \
  --workspace build/public_compiler_installed_smoke_20261110_01 \
  --output build/public_compiler_wheel_smoke_20261110_01 --plan
python -m scripts.verify_compiler_wheel \
  --workspace build/public_compiler_installed_smoke_20261110_01 \
  --output build/public_compiler_wheel_smoke_20261110_01 --execute
```

`--execute` may download Python build requirements/dependencies. It builds an unlicensed **local test artifact only**; no PyPI/GitHub upload, native FPGA tool, external competitor or original research run. The target output must not exist. A failing module-inclusion check should lead to a packaging fix, not relaxation of the audit. The code and tests above **have not yet been author-executed**.

No wheel passing status, public redistribution right, timing closure or board-performance claim is implied by the previously passing source-preview tests.

### 10 November 2026 — NOTICE wheel metadata gate correction

The author ran **9/9 original wheel verifier tests** and a successful read-only wheel plan. The real wheel build proceeded to wheel member inspection, then the original audit rejected the standard generated `fpgai_research-0.3.0.dev0.dist-info/licenses/NOTICE.md` member. The exception occurred **before** the separate wheel installation or CLI smoke; those are not yet verified. This is a packaging-auditor false positive, not evidence of ONNX compiler or numerical failure.

The patched wheel audit permits only that exact `NOTICE.md` metadata member, byte-for-byte matched to the original nonpublishable staged source `NOTICE.md`, and requires its `License-File` declaration. It rejects added/surprise license files and any `License-Expression` claim. **NOTICE.md grants no source redistribution rights.** No project license or rights approval was added.

The original failed `build/public_compiler_wheel_smoke_20261110_01` is intentionally retained. After pulling the new commit, first run the expanded focused tests. If desired, audit the previously built wheel without another wheel build by invoking `inspect_wheel` with the original stage notice and selected manifest; then run the **full** opt-in wheel+installed CLI gate with an entirely new create-only destination:

```bash
python -m pytest -q tests/test_verify_compiler_wheel.py
python -m scripts.verify_compiler_wheel \
  --workspace build/public_compiler_installed_smoke_20261110_01 \
  --output build/public_compiler_wheel_smoke_20261110_02 --plan
python -m scripts.verify_compiler_wheel \
  --workspace build/public_compiler_installed_smoke_20261110_01 \
  --output build/public_compiler_wheel_smoke_20261110_02 --execute
```

These checks are **not yet rerun on the author's machine**. `--execute` may fetch Python build/dependency packages; it performs no native FPGA tool invocation, license grant, source publication, Git history rewrite or original-study change.
