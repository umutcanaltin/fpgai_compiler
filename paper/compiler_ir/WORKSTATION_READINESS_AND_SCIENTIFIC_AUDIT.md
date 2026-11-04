# Compiler-paper experiment readiness and scientific-value audit

**Audit date:** 2026-11-02. **Code:** `umutcanaltin/fpgai_research`, main. **Goal:** decide whether to run *the experiments* on a separate Linux/Vitis PC, not whether the paper is ready for submission.

## Go/no-go boundaries

| Question | Audit decision | Evidence / condition |
| --- | --- | --- |
| Is the existing CPU/host/ONNX pipeline executable? | YES, for a dedicated clean checkout | Both quick and full profiles previously passed at source 1c8db270 (GitHub Actions run 41372708180); the expanded mixed-size profile must pass its own source-pinned CI before a new run is green-lit |
| Can the user's installed Vitis/toolchain/license/part be verified from GitHub? | NO | Run the local `--doctor --with-native` and check the tool output yourself; this is binary detection, not a license or bitstream proof |
| Are native hardware measurements already available for CST? | NO | 2026 source-only HLS packages and empty native scorecards must not be promoted to E2/E3 evidence |
| Are original trained MNIST model assets in the public repo? | NO | The trained study checks source-hashed private ONNX, inputs, labels and metadata. It does not silently retrain a substitute |
| Is the current CST path general for CNNs and branched graphs? | NO | Sequential MatMul/Gemm/Relu + leading Flatten is tested; Conv2D, multi-reader edge lifetimes and arbitrary topology require separate research |
| Is the IR established as novel relative to related work? | NO | The publications already include composable type-safe schedules and spatial/temporal reuse; direct pinned comparison is missing |
| Is an exploratory small/medium source-matched hardware study worthwhile? | YES, conditional | Run a separate original licensed Vitis version after CPU and source preparation qualification, record all failures and native tool/clock configuration |

## Scientific questions and actual observables

**H1 — representation.** Does a *fixed* Functional IR + effective per-operator Phi admit more than one legal tensor availability/lifetime schedule, with actual changed emitted source and independent numerical agreement? Use the complete case-level source hashes, legality outcomes, GCC and ONNX reference traces; **distinct hashes alone are not a hardware result**.

**H2 — legality.** Does a constructive, separate bounded schedule-language oracle agree on the entire finite candidate universe? Existing evidence: 3,600/3,600 schedule permutations (150 legal, 3,450 illegal), with zero observed disagreements. This establishes agreement only for five six-event test domains sharing an atomic-stage specification; it does not prove arbitrary RTL correctness or detect realistic defects at a known rate.

**H3 — lowering fidelity.** Does actual generated C++ implement complete nonlinear operators, bias and transposed Gemm correctly? The synthetic-ONNX/GCC tests and 16-case nonlinear experiment support the restricted tested chain. Original trained weights must be evaluated as a separate data and numerical provenance experiment.

**H4 — hardware effect.** Measure actual original Vitis CSim, source-attributed bounded worst-case CSynth cycles, LUT, FF, DSP, BRAM, clock/timing report, tool version and *all* failed or missing cases. Compare matched baseline/variant only within the same graph, precision, part, requested clock/uncertainty and tool version. A source scratch ratio is not a BRAM measurement, and `worst_cycles × requested_clock_ns` is not routed latency or achieved Fmax.

**H5 — novelty.** Allo (PLDI 2024) already provides composable schedules, memory-layout/interface typing and buffer sizing; ST-Flow (PACT 2026) already supports spatial/temporal/hybrid mapping, cross-kernel hardware reuse and native Vitis validation. The only defensible unverified hypothesis is that the *specific coupled operator-Phi / element-lifetime relation and independent candidate certificate* offers a distinct expressiveness or verification capability. A per-operator-only ablation is not a published-compiler baseline.

## Prespecified benchmark population

The `quick` profile uses the older tiny pilot HLS source triplets to verify environment and source attribution. The `paper` profile now **separately labels** a source-only `paper` shape catalog with four original tensor-layout groups (12 cases) and four row-block groups (12 cases). Both families include two small controls and two medium-size graph groups. Medium workloads are for **exploratory hardware feasibility**, not broad statistical generalization. No observed native results were used to pick the shapes.

The native Vitis defaults remain a KV260-family part, 32 ns requested clock, 8.64 ns explicit uncertainty and 0.05 absolute CSim tolerance. These are provisional study settings and may be too relaxed for a performance paper; choose, log and justify any changes **before** beginning the source-pinned run. Actual Vitis 2023.2 and ST-Flow's validated 2024.2 are different toolchains: cross-tool numerical or resource comparisons require alignment or an explicit version limitation.

## Workstation launch sequence and release gate

Run from a **new standalone compiler-only clone**, never from the source-pinned RL L0 worktree. First bootstrap dependencies in the new checkout; the code does not mutate RL/Gym.

```bash
git clone https://github.com/umutcanaltin/fpgai_research.git fpgai_research_compiler_paper
cd fpgai_research_compiler_paper
bash experiments/compiler_ir/run_all.sh --bootstrap --profile paper --doctor
bash experiments/compiler_ir/run_all.sh --profile paper --plan
bash experiments/compiler_ir/run_all.sh --profile quick
bash experiments/compiler_ir/run_all.sh --profile paper
```

To check the **local** Vitis installation once its AMD environment is sourced, run `bash experiments/compiler_ir/run_all.sh --doctor --with-native`. If Vitis is available, first attempt source-pinned native **quick** cases on the isolated worktree with `--with-native --confirm-isolated-native`, inspect CSim and CSynth receipts, and only then run the full native `paper` profile using a **new output directory**. Do not edit source/config or pull changes during a run.

## Minimum requirements before claiming the paper contribution

- Independently check and preserve all original CSim numerical passes and CSynth bounded latencies; a green command only means orchestration executed, *not* that native pairs qualified.
- Produce paired resource/latency tables including failures, actual Vitis version, model/weights/input hashes, clock/uncertainty and tool settings; do not discard infeasible points.
- Demonstrate a stable cross-operator architectural effect on medium or larger graphs; the selected 2–16-row source cases are a **pilot** with limited external validity.
- Execute original trained MLP E1 on source-verified assets, not synthetic substitutes; include explicit operator inventory and classification error/accuracy denominators.
- Implement and evaluate additional non-chain semantics if claiming a general neural-network Model IR (Conv2D, fan-out, residual edges).
- Directly assess the exact related-work artifacts and distinguish our verification semantics from Allo, ST-Flow and StreamTensor. If the capability is equivalent, **do not claim firstness**.

## Related-work primary sources

- [Allo, PLDI 2024](https://arxiv.org/abs/2404.04815): independent kernel customizations, typed composition, memory/stream communication.
- [ST-Flow, PACT 2026 artifact](https://github.com/UCLA-VAST/ST-Flow): spatial-temporal accelerator compiler; `PACT26` artifact; authors validate with Vitis HLS 2024.2.

**Final distinction:** The automatic experiment harness is useful and CPU-verified. The scientific contribution remains a **testable research hypothesis**, not an accepted novelty or hardware advantage. Start an exploratory original native run only after green CI of the expanded benchmark source and local preflight.
