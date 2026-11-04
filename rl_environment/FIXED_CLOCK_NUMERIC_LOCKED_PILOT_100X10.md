# Easier FPGAI Gym milestone: fixed 32 ns + locked 16/6 numeric format

**1 November 2026 — implementation committed; prospective experiment only. The new native experiment has NOT run.**

## Why the objective is easier

An initial GNN learning milestone should isolate *hardware architecture search* rather than simultaneously solving fixed-point precision, clock timing and architecture search.

- **One requested HLS clock:** 32.000 ns (slowest / least timing-aggressive period **inside the current allowed 2–32 ns envelope**). This is **NOT verified physically safe**, does not imply 31.25 MHz achieved Fmax, and cannot repair missing original HLS XML timing evidence.
- **One numeric format:** global `precision_bits=16`, `integer_bits=6` locked to the compiler's original baseline. The actor cannot select either of these fields; all candidate mutations and backend authorizations are filtered. Source tasks have a new opt-in hash and the independent audit refuses precision drift. The HLS emitter still uses actual 16-bit fixed point, not substituted software simulation.
- **Architectural variables remain searchable:** per-operator parallelism, PEs, SIMD, reuse/serialization, tile and pipeline II, each subject to compiler legality and real Vitis.
- **One shared GNN trained from scratch:** 100 variable one-/two-operator networks × up to 10 original Vitis HLS attempts per training episode; 1 validation + 8 independent test/reference arms, upper bound 1090 HLS calls. No pretrained RL or synthetic QoR.
- **Unchanged scientific reward**: infeasible `-0.05`; feasible first/no improvement `-0.005`; improving feasible `log(previous_best_ns/updated_best_ns)-0.005`. Since clock is fixed, minimum whole-top HLS ns is equivalent to minimum qualified HLS worst-case cycles, still subject to numerical, resource, provenance and effective scheduling timing gates.

## CRITICAL SOURCE-EVIDENCE STOP GATE

The independently audited old interrupted `build/gym_100_train_10_hls_seed101_learning_v3` recorded 1009 cases: 850 `ORIGINAL_HLS_TIMING_REPORT_INVALID`, 65 numeric/CSim/source failures and only 9 verified scheduling-timing failures. **Clock relaxation alone is not demonstrated to solve the main failure source.**

The earlier **24 ns smoke** produced 22 successfully exiting Vitis subprocesses, but vendor `exit=0` is NOT strict timing qualification. The user has not yet provided its independent audit or qualification funnel. **Before pulling new source**, audit that smoke while still checked out at its original source SHA; otherwise use the Git worktree command below:

```bash
cd ~/Desktop/github_projects/fpgai_research
SMOKE=build/gym_fixed24_arch_smoke_seed111_v1
git show HEAD:rl_environment/gym_study.py >/dev/null
python -m rl_environment.gym_audit --study "$SMOKE"
python -m rl_environment.qualification_funnel --study "$SMOKE"
```

If the original smoke SHA does not equal current HEAD, use its exact `campaign.json.source_revision` in a dedicated worktree, and point the audit to the absolute existing smoke directory (never edit study files). Do not call `gym_audit` with changed main source, as its source pin must fail.

Also use the new read-only failure classifier on the still-present 1009-case campaign:

```bash
python -m rl_environment.qualification_funnel \
  --study build/gym_100_train_10_hls_seed101_learning_v3
```

It reports `timing_exception_category_counts`, distinguishing missing `ClockUncertainty`, missing/malformed period fields, freshness/hashes, and other strict parser errors. If `missing_clock_uncertainty` dominates, repair/attest the real Vitis HLS 2023.2 effective timing-budget semantics with original report evidence; **never set uncertainty to zero or mark cases feasible by guess**. If clock only fails actual scheduling gate, 32 ns is a reasonable intentionally conservative HLS *target*, not a guarantee.

**Do not start 100×10 native experiment while XML failure cause remains unresolved.** First make a tiny new source-pinned smoke and demand real accepted evidence. Preserve old source-pinned studies and private checkpoints.

## Zero-Vitis preflight

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q rl_environment tests
python -m pytest -q
python -m rl_environment.gym_study \
  --train-episodes 100 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 10 \
  --fixed-clock-ablation-ns 32.000 --freeze-numeric-format --seed 111
```

The source plan must declare `fixed_clock_ablation_ns=32.0`, `freeze_numeric_format=true`, `fixed_numeric_precision_bits=16`, `fixed_numeric_integer_bits=6`, 109 episodes and a maximum of 1090 fresh HLS invocations. This is a *new* source-pinned protocol, not the old 24 ns ablation.

## Gate 1: small original-Vitis smoke, create-only

```bash
python -m rl_environment.gym_study \
  --train-episodes 2 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 2 \
  --fixed-clock-ablation-ns 32.000 --freeze-numeric-format \
  --seed 111 --output build/gym_fixed32_numeric16_6_smoke_seed111_v1 \
  --execute-native --max-total-vitis-calls 22 \
  --vitis-executable vitis_hls --timeout-seconds 7200
python -m rl_environment.gym_audit \
  --study build/gym_fixed32_numeric16_6_smoke_seed111_v1
python -m rl_environment.qualification_funnel \
  --study build/gym_fixed32_numeric16_6_smoke_seed111_v1
```

Required: independent original-native case replay passes, numerical source/testbench provenance is valid, all advertised actions use exactly 16/6 format and 32 ns, and sufficient cases have accepted numeric/resource/timing data to produce a meaningful latency signal. A Vitis exit code alone is NOT a passed gate.

## Gate 2: complete 100×10 only if gate 1 proves viable

```bash
python -m rl_environment.gym_study \
  --train-episodes 100 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 10 \
  --checkpoint-every-train-episodes 10 \
  --fixed-clock-ablation-ns 32.000 --freeze-numeric-format \
  --seed 111 --output build/gym_fixed32_numeric16_6_100x10_seed111_v1 \
  --execute-native --max-total-vitis-calls 1090 \
  --vitis-executable vitis_hls --timeout-seconds 7200
python -m rl_environment.gym_audit \
  --study build/gym_fixed32_numeric16_6_100x10_seed111_v1
python -m rl_environment.paper_evidence \
  --study build/gym_fixed32_numeric16_6_100x10_seed111_v1 \
  --output build/gym_fixed32_numeric16_6_100x10_paper_v1
python -m rl_environment.gym_report \
  --study build/gym_fixed32_numeric16_6_100x10_seed111_v1 \
  --output build/gym_fixed32_numeric16_6_100x10_report_v1
```

To show learning, the *trained* GNN must outperform the identically initialized, **never-trained GNN** and independent controls on the same untouched test network and same HLS budget, with strict qualification. Single seed/test task is a pilot, not proof of population-wide generalization. Compare first/last training feasibility, actual per-task latency improvements and source-derived diagnostic curves honestly; do not smooth a curve and relabel it as a different reward, and do not assert learning if comparison fails.

**Progressive curriculum after this result:** fixed numeric+fixed clock => fixed numeric+narrow continuous clock => unlocked numeric+clock; only then expand operator families and depth. Each stage has separate source-pinned cases and independently audited tests.
