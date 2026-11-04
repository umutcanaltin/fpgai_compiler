**Next gated milestone — Native Gym RL 100×10 learning revision:** Source changes described in [SPRINT_RL_LEARNING_REDESIGN.md](SPRINT_RL_LEARNING_REDESIGN.md) are pending the CPU suite and a an audited fresh smoke-case diagnosis (old native folders deleted to free space); then execute a source-pinned 22-native-call smoke and only after an independent audit consider the 109-episode/1090-maximum-call study. Track full state/action/feasibility/loss curves and matched heldout controls. Do not claim improved learning until verified by comparative evidence.

# Roadmap — canonical Gym-based graph RL research

**31 October 2026; design locked. Workstation verified 24 focused and 651 full tests at `dde0ba6`; actual Vitis 2023.2 XML parsed successfully. Subsequently corrected strict timing feasibility to `EstimatedClockPeriod ≤ TargetClockPeriod − ClockUncertainty` after a real 10/7.664/2.7 ns sample exposed an overly permissive comparison. Strict margin tests passed on workstation; new floating-clock actor tests pending. Original new native Gym qualification remains pending.**
Primary contract: [RESEARCH_QUESTION.md](RESEARCH_QUESTION.md). No external pretrained DRL, no historical experiment tracks as current work.

## Engineering sequence

| Stage | Work | Exit gate |
|---|---|---|
| GYM-00 | Canonicalize specs, clean legacy experiments, preserve source-pinned evidence | Clean main checkout and no conflicting current objectives |
| GYM-01 | Define `Gymnasium.Env` task/reset/step, typed variable graph observations, legal action/observation spaces and reproducible seeds | CPU-only Gymnasium environment checker; no vendor tool on reset |
| GYM-02 | Shared **trainable message-passing GNN**, node/global embeddings, constraints encoder | Same weights process networks with differing depth, width, operator type; backward gradients update GNN |
| GYM-03 | Complete variable-length graph-conditioned architecture decoder with global requested HLS clock period | Default clock-free CLI and learned architecture-conditioned bounded Beta **floating-clock** actor implemented with 0.001 ns action resolution; verify gradients, no fabricated HLS feedback, native receipt and replay. Legal candidate proof, action identity and duplicate rejection, same action space for controls |
| GYM-04 | Original Vitis timing parser, exact source/part/clock binding, HLS timing-closure feasibility, numeric/resource constraints | Fail closed on missing/ambiguous estimated clock OR reported uncertainty and on negative effective scheduling margin; trusted 10/7.664/2.7 ns fixture |
| GYM-05 | **Latency-only reward**: measured HLS whole-top worst cycles × **requested feasible period ns**, no resource penalty | Unit tests include same-cycles/different-clock and different-cycles/different-clock reversals |
| GYM-06 | Separate **from-scratch** online RL trainer: shared GNN+policy, continual across training episodes, checkpoint and optimizer state | **100 training episodes selected**; cap **30 HLS attempts each**, append-only progress and private checkpoints every 10; new source needs CPU verification and first audited native smoke |
| GYM-07 | Minimal real native smoke with independent source/numeric/timing audit on new source-pinned revision | Original Vitis 1-network then multi-network passing evidence; preserve older cases |
| GYM-08 | Frozen unseen-network validation, separate same-checkpoint online-adaptation treatment | Matched original calls, no heldout leakage and documented seed/split integrity |
| GYM-09 | Random/local/evolutionary/Bayesian baselines, GNN/non-GNN ablations | Fair same source-defined floating clock bounds/resolution, constraints, native call budgets; cost measured |
| GYM-10 | Multi-seed study and deeper genuine compiler-supported network families | HLS-estimated ns learning curves, paired confidence intervals, honest negative results |
| GYM-11 | Scientific paper and release gates | Full evidence replay audit, provenance, approved IP/license/history scan before publishing |

## Scientific invariants

One episode = fixed real network, numerical samples, FPGA part and constraints; the environment generates a bounded exploratory clock-action domain internally without user clock inputs. One step = one complete architecture plus policy-selected requested clock and one fresh real Vitis HLS invocation. Only objective is feasible estimated latency in ns. Resources/precision/timing are hard constraints. The same randomly initialized, trainable shared GNN and policy persist across all training episodes. No Vivado, no proxy rewards, no claimed physical Fmax.

## Existing code to reuse

- Real FPGAI Functional/Architectural/Lowered Model IR and HLS emission under `src/fpgai_research/`.
- The previous original Vitis CSim/CSynth source provenance, immutable native result rules and independent HLS auditor.
- Compiler and source-attributed HLS modules plus `tasks.py`/`post_hls_env.py` were extended with legal clock choices and HLS-ns qualification, and passed the reported workstation CPU suite at `01c7f87` (21 focused/648 full); subsequent effective-timing-margin edits still require tests. Legacy `agent.py`/`study.py` remain retired for native research.

## Do not silently substitute

- A handcrafted flat feature MLP for a trainable shared GNN.
- A fixed clock for the globally selectable requested HLS period.
- Requested HLS period for measured physical Fmax.
- HLS cycles alone for qualified estimated ns.
- Saving model weights at checkpoints for external pretrained models.
- Earlier tests/smoke native evidence for a qualified new clock-aware HLS result.

## Engineering verification protocol

1. Source tree stable on `main`, original evidence untouched. New Gym modules passed 15 focused and 642 full CPU tests at `f83e5ab`; later schema-probe code is unverified. Native research remains blocked on installed Vitis report schema and fresh source-pinned audit.
2. Focused unit and negative tests for every new gate.
3. Full active test suite in an isolated clean worktree.
4. Explicit, source-pinned small foreground Vitis smoke; independently replay audit all original cases.
5. Hold out network families and training seeds before large native study.
6. Update [SPRINTS.md](SPRINTS.md), [HANDOFF.md](HANDOFF.md), and [CHANGELOG.md](CHANGELOG.md) after each actual completed stage.

## Open source

The repository is PRIVATE. Do not choose a license or publish without institutional/third-party IP approval, entire reachable Git history security review, original weights/reports confidentiality review and user sign-off.
