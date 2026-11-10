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
