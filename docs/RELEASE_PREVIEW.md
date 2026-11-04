# Preparing a public source preview (not a release)

The current research repository contains **scientific study history, unreviewed attribution/license questions, and potentially confidential prior Git objects**. Simply moving a document from the root to an `archive/` folder, or deleting it from `main`, **does not remove it from Git history**.

For this reason, publication is a two-stage process:

1. **Prepare and review a clean-room source preview** assembled only from an explicit allowlist of files.
2. **After author/IP/license/history approval**, decide separately how and where to publish. No tool in this repository makes that decision or changes GitHub visibility.

The current intended license direction is **academic, non-commercial
research-only**, under [proposed terms](ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md).
These terms are **not effective**, and they are not OSI open-source
terms. The clean-room preview contains only the proposal, not an
approved license grant.

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

Before a real public repository or package:

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

Editing the private repository, maintaining a clean external-facing README, reviewing the allowlist and producing a preview are allowed without changing visibility. Publishing, adding an unapproved license, deleting source-pinned results or rewriting Git history are **not** implicitly authorized by this work.
