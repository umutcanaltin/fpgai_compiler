> **6 November 2026 cleanup checkpoint:** The active first-paper research branch now protects future local evidence with `.gitignore`, provides issue/PR templates without Actions, and includes linked public user guidance in a curated preview. The original C1/C2 author-run source reports remain local, immutable and SHA-pinned. **No source license, institutional approval, full-history security scan or completed clean-room release test is asserted.** A compiler-only `sdist` intentionally excludes `paper/` experiment notebooks; a peer-reviewed research artifact will need an independently rights-approved, reproducible source/protocol release rather than merely copying the ordinary Python package. The current GitHub repository is already publicly viewable, which is not permission for redistribution or proof that Git history is clean.
# Source-distribution readiness checklist

This repository is publicly viewable but currently has no operative source-code license. A separately licensed source distribution requires review of institutional, contributor and third-party rights. This checklist does not grant permissions.

## Release-critical gates (currently unresolved)

- [ ] Confirm ownership and obtain written institutional/employer/IP release
      approval for all code, data, generated hardware source and paper-related
      material. Explicitly separate the independent academic compiler from
      any proprietary commercial Morfics or third-party materials.
- [ ] Select and obtain written rights-holder approval for definitive source-code licensing terms compatible with institutional, contributor and third-party rights. The [historical academic distribution option](ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md) is a working document, not an effective license. Use restrictions are not OSI open source.
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
- [ ] Obtain explicit authorization before publishing licensed source or changing repository visibility. Passing a script is not licensing approval.

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

A definitive LICENSE approved by every relevant rightsholder; clean history scan,
resolved third-party attribution, full passing source tests, truthful
research limitations, private experimental data excluded from repository
and archives, and the owner's explicit publication authorization.

**Current licensing status:** No operative license published; public GitHub visibility is separate from a separately authorized source distribution.
