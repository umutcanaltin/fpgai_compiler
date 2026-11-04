# FPGAI compiler-IR research: workstation validation runbook

**IMPORTANT: do not run `git pull` in a checkout while the existing source-pinned L0 RL Vitis campaign is running.** The original campaign was launched from `5221094af585bd501e842deca94d19c2289bb628`; its source and native receipts remain immutable. The independent read-only study audit should finish before moving that checkout to newer `main`. You may instead work in a **separate, clean clone** for compiler development; RL and its code remain intact there too.

## Safe compiler-only source smoke (0 Vitis calls)

After the original RL run + independent audit have completed, or from a separate compiler-only clean clone:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
export PYTHONPATH="$PWD/src:$PWD"
python -m rl_environment.runtime_contract  # source-binding check only; no RL training
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_architecture_interfaces.py \
  tests/test_scopes.py tests/test_refinement_legality.py \
  tests/test_cpp_phi_trace.py tests/test_hls_loop_phi_oracle.py
python -m pytest -q
git status --short
```

The `runtime_contract` check guards against accidentally importing a stale sibling research package; it does **not** initiate RL work. The new interface-inspection tests have been committed to remote `main` but were **not yet run on the user's workstation**. Any failure must be investigated before a compiler experiment is reported.

## First E0 inspection, from Python only

```python
from dataclasses import replace
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.architecture.resolver import resolve
from fpgai_research.architecture.interfaces import inspect_array_interfaces

f = FunctionalModel(
    tensors=(
        Tensor("x", (1, 3)),
        Tensor("w1", (3, 4), data=(0.125,) * 12),
        Tensor("h", (1, 4)),
        Tensor("w2", (4, 2), data=(0.125,) * 8),
        Tensor("y", (1, 2)),
    ),
    operators=(
        Operator("mm1", "MatMul", ("x", "w1"), ("h",)),
        Operator("mm2", "MatMul", ("h", "w2"), ("y",)),
    ),
    inputs=("x",), outputs=("y",),
)
base = ArchitectureRequest()
architecture = resolve(f, base, {
    "mm1": replace(base, pes=2, simd=3),
    "mm2": replace(base, pes=3, simd=2),
})
receipt = inspect_array_interfaces(architecture)
assert receipt["status"] == "SOURCE_ARRAY_CONTRACT_COMPATIBLE"
assert receipt["original_vitis_calls"] == 0
print(receipt["status"], len(receipt["edges"]))
```

This tests what the current HLS **source-array** contract can express. It does not claim actual concurrent stream communication, interconnect circuitry, mixed-precision support or synthesized physical realization. Do not label this diagnostic as the new paper's completed contribution; C2 still requires substantial implementation.

## Paper milestones

1. Independent CPU suite green at new `main` (above).
2. Concretely specify and implement an architectural edge contract beyond current functional dependencies.
3. Positive and negative two-operator composition tests with actual lowering and independent host execution.
4. Restricted uniform-mapping baseline and deterministic expressiveness campaign.
5. Small original Vitis CSim/CSynth campaign with *new* source-pinned artifact roots; never mix with RL native directories.
6. Write manuscript results only after verified numeric/source evidence exists.

**Preservation backup:** `backup/rl-environment-frozen-2026-11-01` at `5221094af585bd501e842deca94d19c2289bb628`. This is a remote Git snapshot, not a disaster-recovery/offline export.


## New CST-IR compiler-only test commands (zero Vitis calls)

**Only from a different clean compiler checkout, or after the running L0 study and independent audit have finished.** Keep active L0 checkout on \`5221094\`, without \`git pull\`.

\`\`\`bash
export PYTHONPATH="$PWD/src:$PWD"
python -m compileall -q src/fpgai_research tests

python -m pytest -q \
  tests/test_architecture_interfaces.py \
  tests/test_compositional_model_ir.py \
  tests/test_cst_ir_campaign.py \
  tests/test_cst_row_lifetime.py \
  tests/test_cst_row_campaign.py \
  tests/test_scopes.py \
  tests/test_refinement_legality.py \
  tests/test_cpp_phi_trace.py \
  tests/test_hls_loop_phi_oracle.py

python -m pytest -q

python -m fpgai_research.experiments.cst_ir_campaign \
  --output build/compiler_ir/array_v1 --max-cases 96 --host-check-pairs 4

python -m fpgai_research.experiments.cst_row_campaign \
  --output build/compiler_ir/row_lifetime_v1 --max-pairs 18 --host-pairs 4
\`\`\`

The last two commands create directories; change the output name if an earlier run already occupies a path, rather than overwriting it. \`host-check-pairs\` runs GCC and numerical reference evaluation, not Vitis. The row-interleaved backend is opt-in, requires two MatMul operators, one internal nondegenerate matrix, constant weights, compatible row schedule and one input/output. No universal Conv/stream graph support is implied.

**Scientific caution:** \`E0\` typed/witness results and \`E1\` compiled-host source are evidence of this subset only. No HLS native or physical results have been measured by these scripts. Do not confuse row scratch *source declaration* with physical BRAM utilization.

**Novelty review:** [Prior-art gap matrix](PRIOR_ART_NOVELTY_GATE.md).


## Source-pinned original Vitis HLS experiment (explicitly opt-in, *not* run here)

**When L0 has finished and its read-only audit is complete**, from a clean compiler checkout with an original supported AMD Vitis HLS install:

\`\`\`bash
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest -q \
  tests/test_compositional_model_ir.py \
  tests/test_cst_row_lifetime.py \
  tests/test_cst_ir_mutations.py \
  tests/test_cst_pipeline.py \
  tests/test_cst_ir_campaign.py \
  tests/test_cst_row_campaign.py \
  tests/test_cst_native_campaign.py

python -m fpgai_research.experiments.cst_native_campaign prepare \
  --output build/compiler_ir/native_cst_v1 \
  --max-cases 12 \
  --part xck26-sfvc784-2LV-c \
  --clock-ns 32 \
  --clock-uncertainty-ns 8.64 \
  --tolerance 0.05

python -m fpgai_research.experiments.cst_native_campaign dry-run \
  --study build/compiler_ir/native_cst_v1 --max-calls 3

# Only when explicitly choosing to spend native original HLS calls:
python -m fpgai_research.experiments.cst_native_campaign execute \
  --study build/compiler_ir/native_cst_v1 \
  --max-calls 3 --execute-native \
  --vitis-executable vitis_hls

# Then audit read-only, including missing/infeasible cases:
python -m fpgai_research.experiments.cst_native_campaign audit \
  --study build/compiler_ir/native_cst_v1
\`\`\`

**Source pinning:** \`execute\` refuses a checkout revision different from the study's recorded SHA. \`audit\` intentionally accepts an older case root from a later checkout, but still validates all original source/Tcl/testbench hashes. Separate studies for each compiler source change; never overwrite \`native_cst_v1\`.

**Timing limitations:** the audit reports only original source-attributed CSim and bounded HLS cycles, plus \`cycles × requested period\` when available; it **does not** assert the actual target clock is achievable or that a row-buffer source declaration reduced synthesized BRAM. Native timing/resource feasibility needs a separate explicit comparison of qualified report fields and their constraints. This command sequence does not initiate or change any RL study.


## Read-only matched native report

After **any** native campaign (or before, to verify unmeasured cases remain blank):

\`\`\`bash
python -m fpgai_research.experiments.cst_native_report \
  --study build/compiler_ir/native_cst_v1 \
  --output build/compiler_ir/native_cst_scorecard_v1
\`\`\`

The report creates \`paired_hls.csv\` and \`report.json\` without running Vitis; it preserves \`None\` for missing original synthesis data and reports only HLS cycle ratios **without** claiming achieved target clock or routed speedup. Use a new output path for every updated scorecard.

The native test matrix alternates uniform and heterogeneous per-operator PE/SIMD/serialization in matched 3-policy triplets. Additional compiler-only test files: \`tests/test_cst_native_report.py\`, \`tests/test_cst_ir_mutations.py\`, \`tests/test_cst_pipeline.py\` and fan-out coverage in \`test_compositional_model_ir.py\`.


## Multi-operator row-chain CST Model IR — CPU-only experiment

From a clean compiler checkout (not the frozen active RL workstation checkout):

\`\`\`bash
python -m pytest -q \
  tests/test_cst_row_chain.py \
  tests/test_cst_chain_campaign.py \
  tests/test_cst_cpu_report.py

python -m fpgai_research.experiments.cst_chain_campaign \
  --output build/compiler_ir/chain_v1 \
  --max-cases 12 --host-cases 4 --full-trace-cases 2

python -m fpgai_research.experiments.cst_cpu_report \
  --array-study build/compiler_ir/array_v1 \
  --row-study build/compiler_ir/row_lifetime_v1 \
  --chain-study build/compiler_ir/chain_v1 \
  --output build/compiler_ir/cpu_paper_table_v1
\`\`\`

The new \`row_chain\` policy is available via the opt-in \`compiler/cst_pipeline.py::compile_cst\`; its implementation supports a strictly sequential chain of 2–8 independent MatMul operators, each with its own Phi configuration, a static input and constant weights. A single row of each *internal* tensor is held in scratch until the next operator consumes it. Fan-out, unrelated operators, arbitrary casts, automatic FIFOs, and overlapped multi-clock/dataflow are **not** supported. The independent host trace verifies all per-edge row-ready/consumed events and the existing executed-\`Phi\` checker checks the original iteration assignments.

The [compiler-IR GitHub CPU workflow](../../.github/workflows/compiler-ir-cpu.yml) runs create-only array, row-lifetime and multi-operator-chain campaigns, generates a source-pinned E0/E1 paper table, prepares 12 original HLS sources **without running Vitis**, and uploads raw cases and reports for independent review. The snapshot SHA inside each report is the source identity: never merge different SHAs into one results table.


## Native HLS workflow from GitHub (explicit manual execution only)

The repository now contains [compiler-ir-native-vitis.yml](../../.github/workflows/compiler-ir-native-vitis.yml), deliberately **workflow_dispatch-only**. In GitHub Actions, select the manual workflow, a source revision, and 3/6/9/12 HLS cases. It runs on a dedicated **self-hosted, linux, x64, vitis-hls** labelled runner. Configure a separately licensed Vitis HLS environment before enrolling a runner. **Never use the source-pinned active L0 RL experiment checkout/runner**. The workflow refuses missing vendor tools, executes a CPU confidence gate, creates a new source-pinned matched HLS root, explicitly runs bounded original native calls, and archives original source/testbench/Tcl/logs/receipts and the independent auditor outputs.

**No runner is configured or claimed by this document**. Without a labelled, Vitis-equipped runner, the job cannot run. Local workstation commands earlier in this runbook are the alternative, using a distinct clean compiler checkout and preserving the frozen RL experiment. No tool results from this session establish that the native HLS experiment has been executed.

Latest verified compiler CPU milestone and go/no-go: [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md). Full technical paper draft: [MANUSCRIPT_DRAFT.md](MANUSCRIPT_DRAFT.md).

## Independent reference, scaling and trained-network coverage

The new source-pinned CPU research runs (no Vitis):

```bash
python -m pytest -q tests/test_cst_independent_schedule_oracle.py \
  tests/test_cst_research_oracle_scaling.py \
  tests/test_cst_trained_model_preflight.py

python -m fpgai_research.experiments.cst_oracle_campaign \
  --output build/compiler_ir/oracle_v1 --domains 5

python -m fpgai_research.experiments.cst_scaling_campaign \
  --output build/compiler_ir/checker_scaling_v1 \
  --max-graphs 15 --repetitions 3
```

Only from a machine with the *actual trained* MNIST study directory, exact metadata and supported optional ONNX dependencies:

```bash
python -m fpgai_research.experiments.cst_trained_model_preflight \
  --trained-directory /path/to/real/trained/mnist_mlp \
  --kind mlp --output build/compiler_ir/trained_mlp_admission_v1
```

This last command audits the real model's complete operator list and immutable hashes; it **does not** run CST HLS generation or numerical inference. A trained MLP with Flatten/Gemm/Relu must be recorded as not supported by the current strict MatMul-only row-chain path, **not transformed by deleting those operators**. If a real source directory is absent, record the missing workload rather than manufacturing training weights. Do not re-use this experimental evidence as original fixed-point CSim or paper accuracy.


## S3 nonlinear E1 and real trained MNIST experiments

Synthetic, deterministic, explicitly **not trained** network with intact Flatten → Gemm(bias) → Relu → Gemm(bias) semantics:

```bash
python -m pytest -q tests/test_cst_nonlinear_row_chain.py \
  tests/test_cst_nonlinear_campaign.py \
  tests/test_cst_trained_mlp_host.py \
  tests/test_cst_trained_mlp_native_prepare.py
python -m fpgai_research.experiments.cst_nonlinear_campaign \
  --output build/compiler_ir/nonlinear_e1_v1 --host-cases 6 --trace-cases 3
```

**Only if you have the same original, provenance-checked trained MNIST MLP assets** as the research study (ONNX, NumPy sample/label files and metadata; do not copy them into public GitHub):

```bash
python -m fpgai_research.experiments.cst_trained_model_preflight \
  --trained-directory /path/to/original/mnist_mlp \
  --kind mlp --output build/compiler_ir/trained_mlp_admission_v1

python -m fpgai_research.experiments.cst_trained_mlp_host \
  --trained-directory /path/to/original/mnist_mlp \
  --output build/compiler_ir/trained_mlp_e1_v1 \
  --host-samples 2 --atol 0.001 --rtol 0.001

python -m fpgai_research.experiments.cst_trained_mlp_native_prepare \
  --trained-directory /path/to/original/mnist_mlp \
  --output build/compiler_ir/trained_mlp_native_source_v1 \
  --sample-count 1 --part xck26-sfvc784-2LV-c \
  --clock-ns 32 --clock-uncertainty-ns 8.64 \
  --numeric-tolerance 0.05
```

The final command only prepares the **source and CSim testbench**. Execution needs original AMD Vitis HLS in a separate checkout/runner after freezing exact original weights, constraints and full source hashes. A failed CSim or infeasible synthesis must be reported, not hidden. Batch-one MNIST cannot show row-block source-memory benefit. The current nonlinear compiler does not support Conv2D or branching; never strip these from real ONNX models to force admission.
