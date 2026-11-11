# Hardware-stage comparison: matched storage after Vitis HLS and after Vivado

**6 November 2026; original matched Vitis HLS + Vivado 12ns evidence now independently SHA-verified on the author's workstation (read-only), after 40/40 targeted tests passed.** This report reconciles **already-executed original, SHA-locked** R4k/R4l Vitis CSIM/CSynth results with separately executed original R4p/R4q/R4r Vivado OOC routed results. This is NOT the same population as the recent 54-profile C1 E1 GCC study. **No new vendor runs were performed by writing this report or analyzer.**

## Actual controlled comparison and evidence tiers

The original R4k paired HLS source was constructed with the **same source functional model, 32-bit/6-integer-bit numeric format, same operator-local Φ/source event order, part and numerical testbench**, changing only full-versus-row source scratch storage mapping/capacity. R4l Vitis HLS and R4p Vivado post-route were executed separately; audited original file source, RTL and ROM hashes plus tool invocation receipts are required before combining any measurements.

| Original verified measurement | `full_storage` | `row_storage` | Row – full | Correct stage/scope |
|---|---:|---:|---:|---|
| HLS estimated worst-case cycles | 22,954 | 22,602 | −352 | Vitis E3 cycle estimate, **NOT FPGA latency** |
| HLS estimated clock period | 9.857 ns | 9.500 ns | −0.357 ns | Vitis E3 estimated period, **NOT routed clock** |
| HLS cycles × HLS estimated period | **226.258 μs** | **214.719 μs** | **−11.539 μs** | Derived **E3-only latency proxy**, not achieved inference runtime |
| HLS cycles × requested 12ns | 275.448 μs | 271.224 μs | −4.224 μs | Hypothetical same-clock source cycle proxy; **12ns timing failed** after route |
| Vitis estimated BRAM18K | 13 | 13 | 0 | HLS CSynth estimate |
| Vitis estimated DSP | 21 | 19 | −2 | HLS CSynth estimate |
| Vitis estimated LUT | 5,095 | 4,755 | −340 | HLS CSynth estimate |
| Vitis estimated FF | 2,120 | 1,647 | −473 | HLS CSynth estimate |
| Vivado routed BRAM blocks | 7 | 4 | −3 | Source-attributed *descriptive mapped* utilization |
| Vivado routed DSP | 24 | 21 | −3 | Source-attributed descriptive |
| Vivado routed LUT | 1,282 | 1,710 | **+428** | **Adverse source row trade-off** |
| Vivado routed FF | 883 | 1,022 | **+139** | **Adverse source row trade-off** |
| Vivado routed WNS at requested 12ns | **−2.170 ns** | **−3.718 ns** | **−1.548 ns worse** | Both timing FAIL, do **not** compute qualified Fmax from this |
| Vivado routed DRC warnings/violations | 74 | 71 | −3 | Both DRC NONCLEAN; inspect rule-level original evidence |
| Strict routed timing + clean DRC E4 | **NULL** | **NULL** | **Not established** | Neither arm is acceptable as timing-qualified physical implementation |
| Physically realizable inference/board latency | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | No physically qualified cycle-level timing and no measured board input/output throughput |

**The apparent contradiction is the scientific outcome:** source/HLS row retention reduces some costs and HLS estimated latency, and Vivado maps fewer BRAM/DSP resources, but **mapped LUT, FF and timing slack worsen**. No claim that the row implementation is faster on FPGA is justified. Do not compare Vitis estimated BRAM 13 with Vivado mapped BRAM 4 as a percent improvement across stages: they are different tool-stage estimates/representations and only *within-stage matched-arm* contrasts are meaningful.

## 1. Source-only integrated audit — 40-test workstation check and original SHA-attested R4v review PASSED

`src/fpgai_research/experiments/cst_paper_r4v_vitis_vivado_joint_compare.py` reuses the original `r4l.audit_native()` qualified Vitis E1/E2/E3 source reports and `r4r.summarize(r4q.inspect(...))` signed R4p routed/source forensic evidence. It fails closed if **both** Vitis arms are not CSIM/CSynth qualified, HLS source SHA or original revision differs, either routed report loses original SHA/DRC/WNS attestation, any stage resource is absent, or an arm/paired delta is inconsistent. It outputs a single machine-readable Vitis-vs-Vivado source-stage report, including source SHA, HLS cycles/clock and resources, derived **E3-only** latency proxy, routed BRAM/DSP/LUT/FF/URAM, WNS, DRC, blocking conditions and explicit `strict_physical_E4:null`/`board_qualified:false`. No program compilation or vendor tool runs occur in analysis.

New synthetic-source regression: `tests/test_cst_paper_r4v_vitis_vivado_joint_compare.py`, with positive exact-style resource/latency deltas, missing/mismatched source/certified stage/timing/DRC/report digests and without mutation. Synthetic fixtures are **not original experiment evidence**. The original R4 result must be shown only after the read-only audit successfully runs on the author's saved source files.

```bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_paper_c1_original_e1_readonly_descriptives.py \
  tests/test_cst_paper_r4v_vitis_vivado_joint_compare.py \
  tests/test_cst_r4l_schedule_matched_storage_native.py \
  tests/test_cst_r4r_original_routed_tradeoff_ledger.py

# Read and SHA-audit the already-generated Vitis and Vivado artifacts;
# do not repeat the original native tools.
make research-hls-vivado-review
```

## 2. What a publishable physically qualified resource/latency experiment requires

**The R4 original 12ns routed comparison is deliberately a negative descriptive result.** Before claiming real hardware benefit, run a **new study** under a new source SHA and new create-only result roots; do **not** edit the frozen R4k/R4l/R4o/R4p/R4u original outputs, source report hashes, native attempt locks, ROM or checkpoint files. There is no permission to reinterpret failing 12ns timing by merely changing an existing DCP's clock and calling that original study qualified.

**Predefine hypotheses and controlled arms:**

- Same graph, weights, input/output, datatype/precision 32/6, original source Φ event order, synthesis top/part `xck26-sfvc784-2LV-c`, fixed external memory model/BRAM init, same HLS directives and generated testbench per paired arms. One knob at a time: full versus row retained storage. Confirm source code and generated RTL/ROM hashes. Include a strong third alternative, e.g., feasible source banking, only in a separately preregistered follow-up; do not retroactively add it to original pair.
- Stage E2 Vitis CSIM: equality to independent numerical oracle with max error, source SHA and failure counts, for **both** implementations. Stage E3 Vitis HLS: report estimated worst/typical cycles, estimated II if available, estimated period and BRAM18K/DSP/LUT/FF/URAM for **both**, retaining timing failures and missing values. Distinguish latency `cycles × estimated period` from the HLS clock constraint and from actual board latency.
- Stage E4 Vivado: compile/reroute fresh immutable RTL/ROM from these matched source arms at **predeclared feasible clock points**, identical constraints and synthesis/placement seed/configurations per pair, reporting actual resource utilization, routed WNS/TNS and **complete DRC severities/rules**, critical paths and achieved feasible period. Each failed run counts in the denominator. **Clock relaxation does not add missing DSP48 input/output pipeline stages or resolve other DRC rules. The observed DPIP/DPOP rules are DSP pipelining recommendations, not evidence of RAM port collisions.**
- **Timing qualification:** Positive WNS and strict clean DRC must be met for BOTH arms at the same tested clock before making a direct feasible PPA comparison. If no common feasible clock/clean DRC exists, record `UNKNOWN` for relative achievable latency, and publish a negative feasibility outcome. OOC passing cannot establish on-board DMA/inference throughput or power.
- **Latency experiment:** When source-cycle behavior and deployment constraints are qualified, compute `cycles × one common *feasible requested* period` explicitly as a bound/proxy, not as a measurement. To claim **real latency**, run board-resident timing around the same workload, separating core cycles/II, invocation overhead, DMA/host transfer, and end-to-end inference/throughput. Same bitstreams, clocks, inputs, independent correctness checks, repeated trials, first/warm setup and data paths. A routed WNS number alone cannot be used to calculate actual inference time.
- Predefine ≥3 independent place/route configurations per clock if studying tool variance (fixed seed and report the configuration). Keep all original successes/failures, resources, tool/runtime receipts, raw reports, hashes, warm/cold context and scheduling assumptions. Do not cherry-pick only the best routed seed, omit unfavorable RAM/logic utilization or silently recalculate vendor receipts.

**Critical study decision:** Because the original 12ns design has negative WNS and numerous DPIP/DPOP **DSP48 pipelining** warnings (and RTSTAT-10), a valid-design follow-up must investigate registered DSP operators and real routed critical paths, not assume RAM address collisions. A relaxed timing constraint alone does not change those DSP pipeline structures or resolve the existing warning count; the warnings may be performance advisories rather than hardware functional errors. The [new strict-valid timing/latency protocol](R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md) requires complete route, matched clock and clean DRC to claim strict feasibility, and separately records the repository's preexisting narrow warning-only diagnostic without treating it as clean E4 or board acceptance. Any DSP retiming/source memory change is a **new** experimental arm and must restart matched E2/E3/E4 validation.

## Source/file connectivity and change audit

| File | Ownership/dependency and effect | Execution status |
|---|---|---|
| `src/fpgai_research/experiments/cst_paper_r4v_vitis_vivado_joint_compare.py` | NEW read-only source-attested cross-stage comparison, imports existing R4l/R4q/R4r auditors; original source and vendor reports untouched | New analyzer unverified on user's workstation |
| `tests/test_cst_paper_r4v_vitis_vivado_joint_compare.py` | NEW controlled signed-source receipt fixture, source/clock/DRC/resource/latency negative controls; no vendor calls | New tests unverified on user's workstation |
| `paper/compiler_ir/R4V_VITIS_VIVADO_STAGE_PARETO_AND_REAL_LATENCY_GATE_20261106.md` | NEW paper methodology/provenance table, prevents E3 cycle-based physical performance and E4 qualification claims | No new hardware experiment |
| `Makefile` (`research-hls-vivado-review`) | One read-only predeclared original R4k/R4l/R4p source stage analysis command | New Make target unverified |

**Paper integration:** The source 54-profile C1 E1 GCC validation and this separate R4k fixed-graph Vitis/Vivado study are different populations. Do NOT attribute 7→4 routed BRAM or 22,954→22,602 HLS cycles to all 54 C1 profiles. Relative source-slot savings from E1 are not a replacement for this controlled routed hardware comparison.
