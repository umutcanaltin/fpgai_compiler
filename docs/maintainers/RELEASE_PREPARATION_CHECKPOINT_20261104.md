# Public-source preparation checkpoint — 4 November 2026

**Working branch:** `main`, in the **private** research repository. This is an internal maintainer checkpoint, **not** a release approval or public-facing user manual.

## Completed

- Replaced the root's dated research timeline with a concise source-compiler overview, synthetic ONNX quickstart and stable documentation links.
- Replaced the RL-centric contributor entry point with scoped compiler contribution instructions. Preserved both former documents under `docs/maintainers/`.
- Moved root-level extended scientific-review notes under `docs/maintainers/` without deleting the content; full Git history was not rewritten.
- Added user guide, architecture explanation, supported capability matrix, examples, FAQ, scientific evidence guide, research roadmap, public changelog and documentation index.
- Preserved original source/native research evidence and all existing compiler/RL code; no native vendor tools or experimental agents were invoked.
- Curated `MANIFEST.in` so a potential source distribution does not automatically bundle internal paper histories, RL development journals, tests, study outputs or vendor tool artifacts.
- Added source-only scripts `scripts/prepare_public_preview.py` and `scripts/check_public_docs.py` with focused pytest coverage.
- The preview allows selected GitHub issue/PR templates but **no GitHub workflows** or source-pinned paper archives.
- A separate remote-source documentation inspection resolved **72/72** relative links among 18 public Markdown files. This is a static authoring inspection, not an executed local script run.
- Confirmed via GitHub repository metadata: `private=true`, default `main`, no GitHub license. No license, visibility, branch, source compiler, RL code or original build artifacts were changed.

## Safeguarded, not complete

- The original research notebook collection under `paper/compiler_ir/` remains in the **private research repository**, preserving references and provenance. It is **not** part of the curated clean-room preview. A future public GitHub repository should be created from an **independently reviewed clean-room snapshot** if history is not safe to disclose; never simply flip the private research repository to public.
- There is **no approved LICENSE** or recorded authorization from all rights holders.
- A full-history scan and third-party code/IP/data attribution review have **not** been completed.
- The new docs/preview source has **not yet undergone a full clean-room Python installation, pytest or staged-preview audit** on the author's workstation.
- Existing GitHub Actions workflows remain tracked in the private repository, including a manually dispatched native Vitis workflow, but the public preview excludes them all.
- A release tag, PyPI publication, DOI and public repo publication were not created or requested.
- Public research claims must remain bounded; this work does not certify physical implementation or independent external novelty.

## Next workstation verification

```bash
git pull --ff-only origin main
python -m pip install -e ".[onnx,test]"
python -m compileall -q src/fpgai_research scripts tests
python -m pytest -q tests/test_public_release_preview.py tests/test_public_docs_links.py
python -m pytest -q
python scripts/check_public_docs.py --profile compiler
python scripts/prepare_public_preview.py --plan --profile compiler
python scripts/check_open_source_readiness.py
python scripts/prepare_public_preview.py --profile compiler --output build/public_compiler_preview_v1
```

These are local source-only steps; **no vendor tools**. Run the create-only preview into a new destination; use a fresh suffix if the folder already exists.

## Release decision

**NOT APPROVED FOR PUBLICATION.** First confirm compilation and preview integrity on the workstation. Then resolve rights/license/security/history blockers with the relevant institutional owners and only after that request the repository owner's explicit final go-ahead for actual distribution.
