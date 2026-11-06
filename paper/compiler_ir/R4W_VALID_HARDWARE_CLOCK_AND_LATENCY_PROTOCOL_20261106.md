# Timing-qualified matched hardware designs and real inference latency: R4w/R4x

**6 November 2026 | All six new Vivado runs completed; author then passed 32 focused tests and verified all six source-attested WNS/DRC reports. ZERO jointly strict clean-DRC OOC pairs, ZERO measured board latency.** The following preregistration remains as historical method; the timestamped author postrun checkpoint at the end of this document governs current results. Scientific goal: compare real hardware behavior **only for valid designs**. Original after-Vitis/after-Vivado R4 observations are descriptive because both 12ns routed arms have negative WNS; no physical latency or throughput winner can be inferred from them. See [R4v stage table](R4V_VITIS_VIVADO_STAGE_PARETO_AND_REAL_LATENCY_GATE_20261106.md).

## 1. Three different performance quantities MUST be separated

1. **HLS estimated compute latency** = `CSynth worst_latency_cycles × HLS estimated_clock_ns`. This is only an E3 estimate in time; cannot be compared to routed resource counts as if it were achieved hardware performance.
2. **Estimated core response time at a jointly routing- and DRC-qualified clock** = `fixed_HLS_cycles × same feasible clock_period_ns`. This is a **source/RTL-schedule-based analytical proxy**, not a board measurement. It requires preserved exact compiled RTL/ROM and independently qualified HLS cycles, both implementations routed at the SAME clock and all mandatory constraints complete. In the new R4x report this value is **NULL unless both cases strictly pass** WNS>=0, full routing and zero DRC violations. It remains NULL for unmeasured physical deployment.
3. **Actually measured implementation latency** must come from a verified board/RTL timing trace: accept input/start to final output-valid/ready in same clock domain; include any observable stalls and reset/warm-up semantics; `measured_core_latency_ns = observed_cycles × measured_board_clock_period_ns`. Report *end-to-end wall latency* separately from input DMA initiation through output DMA completion and required CPU synchronization. For sustained throughput, use actual measured II/transactions per second, not single-inference latency. Until board timing is captured, every measured latency/p95/II field remains `UNKNOWN`.

The Vitis 2023.2 user guide reports top-level function latency (cycles and time) and II as different concepts. Pipelining can **increase** latency cycles while **improving** clock period or II; a winner cannot be selected using cycle count or WNS alone. Official references: https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/Pipelining-Paradigm and https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/syn.directive.pipeline.

## 2. Mandatory pair acceptance rules

- Same source and numerical graph, weights, correctness testbench, precision (`32/6` for this legacy pair), exact requested source Φ event order, arithmetic/RTL function and source part `xck26-sfvc784-2LV-c`; only full versus retained storage layout differs **within a comparison**. HLS source SHA/RTL/ROM/Tcl SHA and invocation logs must link each vendor stage. No generated HDL changed post hoc.
- Every arm must pass E2 Vitis CSim against the independent oracle and E3 HLS source/cycles/resource qualifications. A fresh one-time Vivado OOC run must complete routing, pass a *shared* predeclared clock constraint with WNS>=0, and report all DSP/LUT/FF/BRAM/URAM, DRC rules/severities, critical paths and source SHA. Equal source clocks alone are not equal achieved internal timing unless constraints actually cover the relevant paths.
- **Primary strict comparison** requires **zero DRC violations for both** as well as complete routing and WNS>=0. The repository already has `toolchains/vivado/advisory_observation.py`: R4w now reuses its **narrow, non-promoting warning-only classifier** for exactly DPIP-2/DPOP-3/DPOP-4/RTSTAT-10 when Vitis CSim, source SHA and native vendor receipts are accepted. It records `advisory_only_timing_met_not_strict`, but does **not waive any rule, set clean DRC, provide board signoff or qualify latency winner**. Such cases require documented independent DRC interpretation before a separately defined acceptance tier. Do not silently waive rules to make a failed original study pass.
- If the original or newly staged route fails one arm, one DRC check, or source/ROM SHA, **exclude BOTH from the qualified pair**, preserve its report and include the failure in the denominator. The original R4p native attempt lock remains immutable.
- When equal-clock pairs are valid, compare **latency in ns/µs**, resource vector, maximum setup slack, power only if measured with switching evidence, and HLS/RTL II if independently available. A clock-sweep result with no strict pairs is a scientifically valid negative feasibility finding but NOT a valid-design speedup experiment.
- Neither OOC routing nor a clean DRC report establishes KV260 board bitstream readiness, external PS/DDR/DMA timing or real inference latency. Actual deployment is a separate qualification gate.

## 3. Why initial Vivado DSP warnings matter (corrected diagnosis)

The original 12ns R4 route/DRC totals were **74 vs 71**, chiefly `DPIP-2`, `DPOP-3`, `DPOP-4`, plus `RTSTAT-10`. These **DPIP/DPOP rules concern DSP48 input/output pipelining**, not RAM read/write port-collision correctness. The source experiment must **not** claim that these counts prove unsafe memory ports. Their performance significance and whether any additional warnings are dangerous need the exact original `drc.rpt` and `critical_paths.rpt` (not just aggregate counts). See examples of Vivado messages at https://discourse.grlib.community/t/multiple-warnings-for-noel-v-32-bit/483 . The original still has **negative WNS** in both arms and is not qualified at 12ns, independently of the warning classification.

**A DSP pipeline is a new source/architecture intervention, not a trivial Vivado waiver.** If source performance is the main paper claim, preregister a **2 × 2 controlled factorial:** storage `full`/`row` × DSP-operator/loop pipeline `original`/`proposed`, with exactly the same graph/weights/precision, independent CSIM oracle, target part and timing policies. Use AMD Vitis HLS 2023.2's supported operator-latency and pipeline directives only after inspecting the exact `model.cpp` arithmetic variables, and report actual post-HLS II/cycles plus generated RTL/ROM hashes. Pipeline stages may add cycles yet improve routed clock, or may increase RAM pressure and fail. **Do not modify the frozen R4k/R4l/R4p originals** to create new pipeline variants.

## 4. The first guarded feasibility probe — source-prepared, NOT executed

**R4w** checks whether the exact previously SHA-attested R4k/R4l generated RTL can route at *predeclared slower clocks* **16, 20, 24 ns**, for both storage arms, six fresh native cases. Target, clock uncertainty (1.5ns), RTL, weights and timing policy are matched. 12ns originals are untouched; unlike changing an existing routed DCP, this **creates and routes distinct new implementations**. All candidate outcomes, failures and source/report hashes are retained. Three clocks × one nominal place/route configuration are **a feasibility pilot, not a variance/repeated-seed PPA study**.

**If at least one clock has strict timing and DRC pass for both arms**, R4x reports a pairwise core-latency-in-ns **estimate** using the original E3 cycle counts at that identical period. If no clock qualifies, R4x explicitly reports `NO_JOINTLY_QUALIFIED_PAIR_NO_HARDWARE_LATENCY_COMPARISON`; this is the expected possibility given the original DSP pipelining warnings, not an error to hide. In that situation perform the 2×2 pipelining refactor as a *new* E2/E3/E4 study, not another isolated clock relaxation or DRC suppression.

**Tool execution is never implicit.** R4w preparation copies and SHA-verifies original source; R4w native execution is separately double-opted-in, one-time, and preserves its attempt lock even if a vendor run fails. R4w audit and R4x comparison are read-only. These NEW files and their synthetic fixtures have **not yet run on the author's workstation**.

### A. Local source-only tests and original-stage review

```bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_paper_c1_original_e1_readonly_descriptives.py \
  tests/test_cst_paper_r4v_vitis_vivado_joint_compare.py \
  tests/test_cst_paper_r4w_feasible_clock_sweep.py \
  tests/test_cst_paper_r4x_feasible_only_core_latency.py
make research-c1-e1-review
make research-hls-vivado-review
```

### B. Prepare six new clock candidates WITHOUT launching Vivado

```bash
cd ~/Desktop/github_projects/fpgai_compiler
OUT=build/r4w_valid_clock_sweep_20261106_01
test ! -e "$OUT"
python -m fpgai_research.experiments.cst_paper_r4w_feasible_clock_sweep \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --prepare-output "$OUT"
```

### C. One-time actual native pilot, only after source-only tests/preflight

**This command launches UP TO SIX NEW Vivado 2023.2 implementations and should be run only if you elect to spend compute on this pilot.** Existing completed R4p and R4u experiments will not be rerun. A process failure stops subsequent cases and permanently preserves the R4w attempt lock; never delete it to retry the same experimental attempt.

```bash
cd ~/Desktop/github_projects/fpgai_compiler
python -m fpgai_research.experiments.cst_paper_r4w_feasible_clock_sweep \
  --study build/r4w_valid_clock_sweep_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --vivado /tools/Xilinx/Vivado/2023.2/bin/vivado \
  --execute-native --confirm-isolated-new-native
```

### D. Read-only postrun acceptance and valid-only latency estimator

```bash
make research-r4w-audit
make research-r4x-feasible-latency
```

**Expected truth:** the R4x time number is `cycles × same feasible clock` *only* when both designs strictly qualify at a clock. It is still **not measured core/board latency**. For a peer-reviewed real latency claim, build actual board-qualified bitstreams and collect per-input hardware cycle counters/time stamps and host/DMA end-to-end latency as described below.

## 5. Predeclared later actual board-latency benchmark (not yet implemented or run)

- Target: the same KRIA KV260 board and supported PS/PL clocking/DDR/AXI/DMA contracts per design; include bitstream SHA, generated IP/RTL source SHA, clock report and validation of the actual measured clock. Do not represent OOC `xck26` resource data as a fully qualified board integration.
- For each identical known-good input, instrument: **PL start accepted cycle**, **PL output committed/valid cycle**, first/last output transfer and host/DMA submit/receive monotonic timestamps. If the data path has buffering or an initial warm-up, record it separately and consistently. Check numerical outputs per sample.
- Measure **single-inference core latency** as observed PL output minus accepted start in clock cycles and ns, **end-to-end latency** including PL+DMA+host synchronization, and **sustained II/throughput** for back-to-back inputs. They are different metrics and must be separate columns in the paper.
- Predeclare ≥3 tool placement configurations and >=30 independent warmed input trials per qualified implementation and report median/p95 and all failures; report board/host load, warm/cold first-call overhead, sample distribution, input/output shape, source/evidence version and any CPU-side computation. No percentile or performance variance claims are made until actual traces exist.
- Pareto objective: **minimize measured latency and resource/power budget while maintaining identical numerical accuracy and timing feasibility**; if no design dominates, show the resource/latency trade-off. Different physical clock points require explicit equal-clock comparisons or a separate 'best individually feasible' secondary comparison, not a mixture.

## 6. Repository change impact and dependencies

| File | Role | State |
|---|---|---|
| `src/fpgai_research/experiments/cst_paper_r4v_vitis_vivado_joint_compare.py` | Read-only join of original R4l source/E3 and R4p/R4q/R4r mapped Vivado reports | Implemented; new targeted tests not workstation-run |
| `tests/test_cst_paper_r4v_vitis_vivado_joint_compare.py` | Source SHA/clock/resource negative controls, strict E4 null | New synthetic tests not run |
| `src/fpgai_research/experiments/cst_paper_r4w_feasible_clock_sweep.py` | New create-only 6-case Vivado feasibility study, exact staged RTL/ROM signatures, one-time native opt-in and source/DRC/clock audit | Source code prepared; native vendor calls **zero so far** |
| `tests/test_cst_paper_r4w_feasible_clock_sweep.py` | Synthetic original RTL+ROM verification, Tcl/clock drift, safe dry-run and native lock checks | New synthetic tests not run |
| `src/fpgai_research/experiments/cst_paper_r4x_feasible_only_core_latency.py` | Rejects invalid pairs, compares ns estimates at common strictly feasible clocks only; real board time stays null | Source-only; new tests not run |
| `tests/test_cst_paper_r4x_feasible_only_core_latency.py` | Validity gate, single-arm failure and exact time-proxy checks | New synthetic tests not run |
| `paper/compiler_ir/R4W_VALID_HARDWARE_CLOCK_AND_LATENCY_PROTOCOL_20261106.md` | This protocol: exact source/native/real-latency distinction and commands | No original user vendor experiments run in this change |
| `Makefile` | `research-hls-vivado-review`, `research-r4w-audit`, `research-r4x-feasible-latency` and targeted source-only checks | New commands unverified locally |

The existing source compiler, R4k emitters, frozen R4l/HLS source files, R4p original RTL/ROM and original vendor/clock reports remain unchanged. The PR is still draft. Any future DSP pipeline variant is a new source experiment and must be tracked file by file in the maintainer impact report.

## Author original post-run checkpoint — 6 November 2026, 22:02 Europe/Amsterdam

**Actual author workstation, no inferred completion.** After pulling branch revision `a0d4db117247f08b0c85849f3fbd5d8c0cdac34a`, the author ran 4 focused test modules: **40/40 tests passed in 0.73s**. The read-only original C1 E1 SHA-pinned analysis then completed on the previously executed immutable full 54-profile/108-GCC result (`build/paper_c1_e1_cpp_bridge_full_20261106_01`, old E1 manifest SHA `a79e700208e5d7cada93dd1f488809b6622f52cf6b14fa7b7d7b05a8d624d83e`). **Overall C1 original source-slot effects:** 54/54 positive, median **15.5** saved source slots, per-profile range **4–45**; median source-slot reduction **50.0%**, per-profile percent range **38.461538–55.555556%**; maximum final absolute numerical error **2.3781921365539915e-7**, maximum intermediate absolute error **2.408715820134688e-7**; original 54 distinct source designs sum **1999 disjoint units vs 1009 reused units** (990 total savings ACROSS separate models, **NOT** a single design or mapped BRAM). Each layout logged **6625 actual source address-access witnesses**. No new compiler/GCC/vendor calls occurred during analysis.

**Original 6 reader/mapping source-only strata (9 profiles each):**

| Reader count | Φ mapping | Median saved source slots | Median reduction % | Min/max saved slots |
|---:|---|---:|---:|---|
| 2 | scalar | 14 | 45.454545 | 4–17 |
| 2 | mixed_phi | 12 | 50.000000 | 4–23 |
| 3 | scalar | 14 | 50.000000 | 7–27 |
| 3 | mixed_phi | 18 | 47.058824 | 8–33 |
| 4 | scalar | 24 | 50.000000 | 8–45 |
| 4 | mixed_phi | 32 | 50.000000 | 13–45 |

**Original Vitis+Vivado read-only joint source audit succeeded:** After original source E1/E2/E3 qualification, same fixed original R4k/R4l source revision `776e3a7a4b599fd114718f6a03ab73e06be17f5c`, 12ns routed OOC original failed both timing and clean DRC. The source-attested after-Vitis stage was CSIM max absolute error **3.7956994241e-06** per arm; full/row HLS cycles **22954/22602**, estimated periods **9.857/9.500ns**, pure HLS latency proxy **226.257578/214.719 µs**. Original routed BRAM **7/4**, DSP **24/21**, LUT **1282/1710**, FF **883/1022**, routed WNS **−2.170/−3.718ns**, DRC counts **74/71**. Vitis and Vivado compiled observations are distinct stages; **strict E4 = NULL**, achieved physical/core/board latency = **UNKNOWN**.

**New R4w one-time native feasibility pilot HAS EXECUTED:** A *new*, create-only 6-case Vivado 2023.2 study was prepared at `build/r4w_valid_clock_sweep_20261106_01` from immutable original R4o/R4k RTL/ROM source SHA. Six new native Vivado implementations were run, one full/row pair at each **16ns, 20ns, 24ns**, each process returned **exit=0**, native runner's printed `actual_new_vivado_calls=6`, `process_completed=true` in all six. No original R4p/R4u native attempts were replayed, no original source/report modified. **That is only vendor process completion, NOT routed WNS/DRC qualification.** Both subsequent original-data read-only `make research-r4w-audit` and `make research-r4x-feasible-latency` **FAILED BEFORE reading per-arm timing/DRC** with `ValueError: R4w unexpected RTL/ROM files under source root` inside `audit_prepared` at the postrun RTL directory inventory.

**New audit-only fix, not yet workstation tested:** Vivado's generated Tcl explicitly `cd`'s into the new `vivado/rtl` working directory. The old `audit_prepared` accepted no additional post-run artifacts there; normal `.Xil` and journal/usage files therefore can block analysis despite intact original RTL/ROM SHA. `src/fpgai_research/experiments/cst_paper_r4w_feasible_clock_sweep.py` now permits **non-source sidecars only inside the native Vivado working directory after a valid immutable one-time native-attempt lock**, while preserving SHA-verification of the complete original staged RTL/ROM/Tcl and original full manifest. New HDL/ROM/constraints/IP files, symlinked entries, extra source files in canonical HLS RTL, missing original file, or invalid lock still fail. Added four targeted synthetic regression tests. The actual *names* of the author's generated sidecars have not been independently listed; this patch addresses the observed postrun class and needs the author's `pytest` and read-only audits to confirm.

**Do NOT rerun Vivado or delete the R4w one-time attempt lock.** Only rerun the new tests, `make research-r4w-audit`, and `make research-r4x-feasible-latency` after pulling. If any postrun SHA/tool receipt or new Vivado DRC fails, report `UNKNOWN`/unqualified with the exact failing gate, preserve all six original new outputs and follow the separate storage×DSP-pipeline E2/E3/E4 study plan. No clock/time/board latency winner has been observed yet. GPU/GCC source slot percentages are not physical FPGA BRAM percentages.

## Post-run read-only scientific checkpoint — 6 November 2026

The author pulled revision `5bbea74`, passed **32/32 focused** R4w/R4x/R4v regression tests, and ran `make research-r4w-audit` and `make research-r4x-feasible-latency`. The first confirmed **six routed source-attested cases** and the original permanent native attempt lock, with **no new vendor executions during audit**. The earlier non-design work-directory sidecar blocker is resolved.

| Original requested clock | Full routed WNS | Row routed WNS | Both timing-met? | Strict clean-DRC pair? |
|---|---:|---:|---|---|
| 16 ns | +0.241 ns | −0.975 ns | No | No |
| 20 ns | +0.433 ns | +0.538 ns | Yes | No |
| 24 ns | +1.249 ns | +2.555 ns | Yes | No |

Full storage has **74** and row storage **71** source-attested, recognized warning-only DRC findings per clock. The native classifier allows a separate observation label for these DSP-pipelining and no-routable-load advisories; it does **not** waive them. Strict pair count is **zero** and strict or actual hardware latency remains **NULL**.

### Non-promoting advisory-only HLS schedule × requested clock sensitivity

The new R4x analyzer reports a **separate, explicitly non-strict diagnostic** only when both arms meet routed timing at an identical predeclared clock and both match the existing narrow DSP advisory classifier, original native attempt lock, unmodified SHA-attested source/report evidence and valid resources. It does NOT relax the strict gate, claim actual RTL execution cycles, or calculate measured board latency. Original HLS worst-case cycle estimates are full **22,954**, row **22,602**.

| Common timing-met clock with nonzero warning-only DRC | Full: HLS cycles × requested period | Row: HLS cycles × requested period | Row − full |
|---|---:|---:|---:|
| 20 ns | 459.080 µs | 452.040 µs | −7.040 µs |
| 24 ns | 550.896 µs | 542.448 µs | −8.448 µs |

These results are **arithmetical sensitivity estimates, NOT verified core/FPGA latency or hardware speedups**. The 16 ns case cannot be fairly paired because row storage fails timing. Routed descriptive resources are **full: BRAM 7 / DSP 24 / LUT 1283 / FF 883** and **row: BRAM 4 / DSP 21 / LUT 1709 / FF 1022**; reduced BRAM/DSP is accompanied by increased LUT/FF. No strict timing-and-clean-DRC Pareto winner has been established.

**Next independent physical study, only if a hardware performance claim is essential:** preregister a NEW 2×2 full/row storage × original/DSP-pipelined source experiment, new source and vendor receipts, preserved original locks, E2 numerical oracle, E3 schedule and II measurements, E4 matched timing/DRC plus verified board core and DMA latency. Do not re-execute original R4w, alter original DCPs or infer board output time from WNS. The existing GitHub R4x extension and new synthetic tests remain pending the author's local execution; the earlier 32-pass result predates those changes.
