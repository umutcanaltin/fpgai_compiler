> **POST-ORIGINAL READ-ONLY DIAGNOSTIC CORRECTION (2026-11-03).**
> The original note below was written *before* the detailed audit.
> The later diagnostic at `142ca73d569ec6944ee30db3382eac333e887905`
> passed **16/16** focused and **1418/1418** full tests, and
> revealed routed **BRAM 1.5→1.5 (tie)**, **LUT 649→684
> (+35)**, **FF 396→407 (+11)**, **DSP 7→7**, and
> **WNS +26.356→+27.304 ns**. There are 10 warnings in
> each case: DPIP-2×3, DPOP-3×3, DPOP-4×3, RTSTAT-10×1.
> The exact graph is **fanout3_3x3x4x4, scalar mapping**,
> not the earlier mistakenly transcribed last_reader_pes2.
> **0 strict E4-qualified cases** and **no routed BRAM
> improvement** are established. The earlier references
> below to unknown physical counts/DRC IDs describe the
> *historical pre-diagnostic status* and are superseded.
> See [complete source-pinned numerical addendum](R2J_PAIR20_POSTROUTE_NUMERIC_DIAGNOSTIC_20261103.md).
>
# CST R2j — Original source-attributed Vivado routed pilot: warning-only, NOT strict E4

**Date of original user workstation execution:** 2026-11-03.
**Original HLS compiler revision:** \`b2d5dcc4f18fad7cb1cd2c342101dd33185e817f\`.
**R2j physical packaging code at original run:** \`adaa373145ed2e3ecf508b78091db4620338e777\`.

This note transcribes the original user-provided workstation results
and does NOT claim to have rerun Vivado, read unseen routed reports,
or established physical BRAM improvement.

## Original preflight and complete denominator

- The new R2i E3 effects exporter ran read-only with **zero vendor
  calls**; original full regression **1,402/1,402 passed** after
  **8/8** new focused tests.
- Original R2i native cases: **72/72 CSim E2**,
  **48/72 CSynth E3**, **24/36 both-E3 qualified matched pairs**.
- Of those 24 original E3 pairs, all 24 improved **estimated**
  worst-case HLS cycle count. Delta minimum **13 cycles**,
  maximum **48**, mean **24.9583** cycles, median **23**.
- HLS-estimated BRAM outcomes across **all 36 prespecified pairs**:
  **6 reductions, 18 ties, 0 qualified regressions,
  12 E3-UNKNOWN** (\`latency_best = undef\`). Do not impute
  unknown results or remove those 12 from the denominator.
- Read-only R2j source/RTL/ROM preparation succeeded with
  **48 source-attributed original HLS RTL packages** and
  **24 blocked E2-only source cases** in an immutable
  **72-case/36-pair** prospective physical study.
  All original Vitis source and vendor artifacts remained untouched.

The newly created original physical study is
\`build/r2j_fanout_physical_36_pairs_v1\`,
and the first source-attributed read-only export is
\`build/r2j_fanout_physical_first_audit_v1\`.
**Both directories are immutable**, do not overwrite,
rerun their original vendor cases or change original results.

## Predeclared physical pilot: pair 20 (case_040 and case_041)

The native HLS comparison predicted **BRAM_18K 3 vs 2**
for this matched fan-out-3, \`scalar\` policy (corrected after original read-only source audit)
(see source-pinned R2i E3 report). The R2j physical pilot
was selected from the six independently qualified
HLS-estimated memory-reduction pairs.

Both original Vivado OOC runs exited successfully,
one after 75 seconds and the other after 73 seconds.
The original read-only report gave:

| Original physical field | disjoint: case_040 | optimal: case_041 |
|---|---:|---:|
| Vendor execution completed | Yes | Yes |
| Original RTL/ROM/report audit | Passed observation gates | Passed observation gates |
| Route observation valid | Yes | Yes |
| Routed timing met | Yes | Yes |
| Routed WNS | **+26.356 ns** | **+27.304 ns** |
| WNS difference optimal minus disjoint | — | **+0.948 ns**, descriptive only |
| Disclosed noncritical DRC warnings | **Yes** | **Yes** |
| Strict zero-DRC E4 qualified | **No** | **No** |
| Board bitstream qualified | No | No |

The original cohort-wide physical export confirmed
**2/72 original Vivado invocation receipts**,
**2 routed warning-only observations**, **1/36
matched observation pairs**, but **0/72 strict
E4 cases** and **0/36 strict E4 matched pairs**.
The other **35/36** pairs are not observed and
must remain UNKNOWN. Original HLS E2-only cases
remain blocked from this physical stage.

**Critical evidence boundary:** The pasted workstation
output did **not** include actual routed BRAM/LUT/FF/DSP
counts or DRC rule IDs. We CANNOT claim that a
routed BRAM saving occurred. It might be observable
from the existing saved physical report, but even
that would be descriptive OOC warning-only evidence,
not strictly DRC-qualified E4.

## First necessary diagnostic: identify exact DRC warnings and routed resources

Before running any new original Vivado cases,
use the NEW read-only diagnostic module against
the already exported immutable physical audit:

\`\`\`bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main

python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r2j_physical_diagnostics.py

python -m fpgai_research.experiments.cst_r2j_physical_diagnostics \
  --export-study build/r2j_fanout_physical_first_audit_v1 \
  --pair 20 \
  --output build/r2j_physical_pair20_readonly_diagnostic_v1.json
\`\`\`

The diagnostic does not run or alter Vivado and
rejects incomplete/forged source-pinned original
72-case denominators. It reports the two
source-attested physical utilization entries,
exact DRC rule IDs/severities/counts,
ROM evidence mode, WNS, descriptive resource
differences and **strict E4-qualified effects
only if both cases have independently clean DRC**.
The original case40/41 warning-only outcome
must remain unqualified.

## Next scientific decision — contingent on original DRC IDs

1. **Do not waive or relabel the warnings.** Inspect
   their actual IDs and precise net/pin violations
   in the already saved original report.
   The existing observation allowlist
   \`DPIP-2\`, \`DPOP-3\`, \`DPOP-4\`, \`RTSTAT-10\`
   is exploratory and does NOT establish E4.
   The original user output does not tell us which
   specific rule IDs appeared.
2. Investigate whether physical warnings are inherent
   in this isolated out-of-context top-level interface
   or in actual generated RTL. Any new constraints or
   wrapper design would define a **new, separately
   preregistered physical experiment** with matched
   control/treatment and new SHA provenance. Do not
   modify existing original physical studies or
   retroactively promote warning-only observations.
3. Decide whether to extend physical runs after
   reviewing the actual warning modes and routed
   resources of the single pilot. If it is meaningful,
   the remaining HLS-estimated BRAM-saving pairs
   are **23 (cases 046/047), 28 (056/057),
   31 (062/063), 32 (064/065) and 35 (070/071)**.
   These are the other five of six positive
   **HLS-only** predictions, not physical effects.
   Each new original Vivado case must be invoked
   only once and retained regardless of outcome.
4. Independently reproduce the result on an external
   compiler baseline and more diverse operators
   before any strong general firstness or
   all-DAG/system-IR performance claims.

## Publication wording

**Allowed now:** Source-pinned 72-case CSim and
48-case strict E3 HLS evidence; 6/36
HLS-estimated memory decreases with 18 ties
and 12 unknown. The first R2j physical pair
completed source-attested placement/routing,
met timing, with +0.948 ns **descriptive OOC WNS
difference**, but both had DRC warnings.
The physical study has **zero strict clean-DRC
E4 cases**.

**Not allowed:** A claim of a verified routed
BRAM improvement, achieved max frequency,
board deployment, zero-DRC physical closure,
measured throughput or scientific firstness.

**Research submission readiness remains NO**
until strict clean-DRC physical validation or
an explicitly justified separated evidence
tier, generality/external comparisons and
reproducibility complete. Existing source
and native proofs remain valuable regardless
of a negative physical result.
