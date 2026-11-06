# R3i: Isolated pinned Allo equivalence-capability comparison

> **6 November 2026 source-location clarification:** This protocol belongs to the active `fpgai_compiler` GitHub research branch. The older working-directory examples that referenced the separate `fpgai_research` checkout have been updated. An external Allo comparison must run in its own isolated Python/MLIR environment; the current protocol and synthetic tests do **not** establish equivalent semantic capability or justify publication priority. Status remains **NOT EXECUTED** until author logs and original SHA receipts are verified.

**Status 4 November 2026: NOT EXECUTED.** The author has only run `--protocol` and verified that `external_artifact_executed=false`. This document describes an optional future **source-artifact experiment in a different Python/LLVM environment**. It does not mark the paper submission-ready or claim that Allo lacks any FPGAI contract.

## Exact external source, scope and prerequisite

- External repository: https://github.com/cornell-zhang/allo
- Exact pinned commit: `abc270c7a1919c780df2b7d7e99de45e81c175b9`
- Public official installation guide: https://cornell-zhang.github.io/allo/setup/index.html
- At the observed pinned commit, `requirements.txt` specifies a **Linux x86-64 CPython 3.12** wheel for `past-0.7.2`, the source-to-source equivalence verifier binding. Its full dependency set includes LLVM/MLIR; a plain Python-only `pip install` is **not guaranteed** to work. The official guide requires a matching LLVM/MLIR environment or its supplied Docker environment.
- `allo/verify.py` implements `allo.verify(s1,s2)` by building Vitis-HLS-oriented C source and invoking PAST. Existing pinned Allo tests have both positive and negative source equivalence checks. We should **never** report their verification as absent.

## Installation isolation and source pinning

Do **not** modify the validated `~/Desktop/github_projects/fpgai_compiler/.venv`. Choose a separate external-artifact workspace. Obtain the Allo source and submodules using normal Git in that workspace, then pin and record the exact source:

```bash
mkdir -p ~/Desktop/external_compiler_artifacts
cd ~/Desktop/external_compiler_artifacts
git clone --recursive https://github.com/cornell-zhang/allo.git allo_r3i_pinned
git -C allo_r3i_pinned checkout --detach abc270c7a1919c780df2b7d7e99de45e81c175b9
git -C allo_r3i_pinned submodule update --init --recursive
git -C allo_r3i_pinned rev-parse HEAD
git -C allo_r3i_pinned status --short
```

Prepare Allo in an **isolated Docker container or distinct conda/venv**, using the official installation documentation. To allow the pinned PAST wheel, confirm `python --version` is compatible with CPython 3.12 (Linux x86-64). **Do not install dependencies into the compiler-paper's validated existing Python environment.** Record the external environment's toolchain and dependency versions, container image digest (if Docker), `git rev-parse HEAD`, and `python -c "import allo, past; print(allo.__file__); print(past.__file__)"` as separate experiment metadata.

The scientific driver refuses a different Allo source revision, a dirty checked-out source, or an import from outside the supplied exact checkout. The external probe writes Allo's temporary generated C files into a separate private temporary directory rather than the research source working tree.

## Safe preflight and isolated execution

From the active `fpgai_compiler` research checkout, checking the protocol requires **no** external compiler:

```bash
cd ~/Desktop/github_projects/fpgai_compiler
python -m fpgai_research.experiments.cst_r3i_external_allo_capability --protocol
```

Inside the verified external Python environment (where `import allo` resolves to the pinned checkout), expose **only the Python source tree** of the existing compiler research package, then opt in:

```bash
export FPGAI_RESEARCH_ROOT=~/Desktop/github_projects/fpgai_compiler
export PYTHONPATH="$FPGAI_RESEARCH_ROOT/src"
cd "$FPGAI_RESEARCH_ROOT"
python -m fpgai_research.experiments.cst_r3i_external_allo_capability \
  --allo-checkout ~/Desktop/external_compiler_artifacts/allo_r3i_pinned \
  --output build/r3i_external_allo_capability_v1 \
  --confirm-isolated-external
```

**Stop and inspect errors** if the experiment reports `ERROR_NOT_A_NEGATIVE_CAPABILITY_RESULT`, a missing PAST binding, a version mismatch, or an unsupported transformation. An error is not equivalent to `EXECUTED_FALSE`. Do not rerun into an existing create-only output folder.

## Prespecified comparisons

1. Same safe two-reader, two-dimensional float32 MatMul fanout program verified against itself (positive equivalence control).
2. Safe program vs a program that swaps the order of two independent readers of the shared producer H (semantic-equivalence control).
3. Safe program vs a program that overwrites H before the final reader, corrupting the final output (negative whole-output equivalence control).

A successful positive/negative result would establish *observable source result equivalence detection* for that Allo version. It would **not** establish or refute a hidden tensor-element last-reader event certificate, per-operator `Phi` source issue-order trace, or policy-neutral E0/UNKNOWN/REJECTED action interface. The stronger same-capability comparison remains a separate scientific gate.

The Allo source functions are hand-translated probe programs, not line-for-line generated from the FPGAI IR; the two compilers use different languages and implementation strategies. Do not compare CPU runtime or generated HLS area and attribute it to the IR. A fair hardware benchmark would require separately matched settings, target constraints and functionality.

## Decisions and possible outcomes

| Observed outcome | Permitted conclusion |
|---|---|
| Safe vs safe `EXECUTED_TRUE`; safe vs overwrite `EXECUTED_FALSE` | Pinned Allo verifier distinguishes output-equivalent and output-changing small programs |
| Reordered safe readers `EXECUTED_TRUE` | This small dataflow is output-equivalent despite different source order |
| Any `ERROR_NOT_A_NEGATIVE_CAPABILITY_RESULT` | External evidence is **UNKNOWN / broken environment / unsupported probe**, not that Allo is unsound |
| Allo actually exposes comparable typed lifetime/issue proof through another API | The proposed FPGAI claim must acknowledge overlap and narrow its contribution |
| No comparable capability demonstrated after full artifact exploration | Only describe *a demonstrated specific artifact/API difference*; never generic priority without broader survey |

After completing the external experiment, pin the generated report, environment, artifact SHA and capability interpretation before revising manuscript claims. Keep the original R3c–R3h cohorts frozen and **do not** combine R3i external-equivalence results with native HLS performance estimates.
