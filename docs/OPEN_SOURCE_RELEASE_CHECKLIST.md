# Source-available research release checklist — BLOCKED until cleared

This repository is currently PRIVATE. The release process must be safe for
academic/employer/third-party intellectual property. The goal is to make the
research reproducible; this document is not authorization to publish.

## Release-critical gates (currently unresolved)

- [ ] Confirm ownership and obtain written institutional/employer/IP release
      approval for all code, data, generated hardware source and paper-related
      material. Explicitly separate the independent academic compiler from
      any proprietary commercial Morfics or third-party materials.
- [ ] Obtain written approval for a **research-only, non-commercial source-available LICENSE** compatible with all contributor, institutional and dependency rights. The [proposed academic research terms](ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md) are **not effective** and there is currently **NO authorized LICENSE**. Do not describe the intended field-of-use-restricted terms as OSI open source.
- [ ] Scan **full Git history, branches and tags**, not just working tree,
      for API keys, passwords, personal data, university NDA material,
      unpublished weights, student contributions, proprietary IP and
      third-party code. Review every flagged historical object; if anything
      must be removed, use a carefully coordinated history rewrite and
      rotate exposed credentials before publication.
- [ ] Audit the dependency and generated-script toolchain license boundaries.
      Vitis/Vivado are proprietary external tools and are **not** bundled.
      Never redistribute vendor binaries, license files, private tool logs,
      unpublished FPGA parts, or derived restricted material.
- [ ] Ensure `build/`, native case outputs, NPZ/PT/ONNX model weights,
      hidden data and test cohorts stay untracked and private, including
      prior commits and release archives.
- [ ] Check private paths, internal URLs, identifiable author/student emails,
      sensitive hosts, hardcoded local home paths, material from NDAs, and
      unpublished research claims in every public document and example.
- [ ] Run a clean virtualenv installation, targeted and complete tests,
      static/safety checks, reproducible non-native smoke examples,
      and verify no GitHub CI starts licensed HLS tools automatically.
- [ ] Verify public README, quick start, operator support, limitations,
      security policy, community contributions, citation, changelog,
      dataset/source provenance and reproducibility.
- [ ] Obtain the user's explicit final approval before changing visibility.
      Do not treat a failed/passing script as permission to publish.

## Local read-only checks

Run:

```bash
python scripts/check_open_source_readiness.py
python scripts/prepare_public_preview.py --plan --profile compiler
python scripts/check_public_docs.py --profile compiler
git status --short
git ls-files
git rev-list --all --count
```

Use an independently reviewed secret/history scanner such as Gitleaks
over **all refs**, not just `HEAD`; scanner invocation/version may vary,
so retain a review record instead of assuming a particular command is
universally available.

For a nonpublishable allowlisted source preview, see [Release preview](RELEASE_PREVIEW.md). A successful documentation link audit cannot establish license ownership, history cleanliness, a safe new repository or scientific validity.\n\nThe script is intentionally fail-closed: local approval details belong
in a gitignored `release-approval.json` (never share it) and a human
must review the complete historical/security/license findings.

## Public release success criteria

A finalized research-only source-available LICENSE approved by every relevant rightsholder; clean history scan,
resolved third-party attribution, full passing source tests, truthful
research limitations, private experimental data excluded from repository
and archives, and the owner's explicit publication authorization.

**Current status:** NOT READY TO MAKE PUBLIC. No visibility changes made.
