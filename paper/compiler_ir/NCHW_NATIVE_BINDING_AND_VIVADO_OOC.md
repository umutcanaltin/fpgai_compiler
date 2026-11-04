# CST NCHW native HLS policy and routed FPGA implementation acceptance

Status: **software implementation only**, no Vivado execution or qualified E4 result has been demonstrated by this document. The immutable original Vitis HLS source cases live in user-controlled `build/nchw_native_v1` (explicit `bram_2p`, Git `8814bd0005e8...`) and `build/nchw_native_auto_v1` (automatic inference, Git `8fe0ef31f6ac...`). Do not copy, change, replace, or re-run existing vendor receipts.

## What the implementation does

The independent read-only `fpgai_research.experiments.cst_nchw_cross_policy` audits the two original HLS study directories, preserving **12 same-storage cross-policy source comparisons**, six four-choice graph/Phi groups and 24 planned HLS outcomes. It requires exact Functional model, original input, numerical samples, device part, HLS clock, explicit uncertainty, numerical tolerance, testbench, Tcl and architecture identities. Distinct Git revisions are permitted ONLY when the actual generated C++ is byte-identical except for **the one requested `#pragma HLS BIND_STORAGE variable=__cst_scratch type=ram_2p impl=bram`**. Different model architecture, clock, source code or missing original source hashes fail closed. HLS QoR stays null when a side has not qualified. Each four-option Pareto is only computed when all four original HLS cases are qualified, and is HLS-estimated, not post-route.

The create-only `fpgai_research.experiments.cst_nchw_physical` prepares an **isolated Vivado OOC research study** *only* from original cases whose native CSim/CSynth and uncertainty-adjusted timing checks pass and whose generated Verilog hashes match the original Vitis invocation receipt. It copies source and testbench, original manifest and generated RTL into NEW case folders, retains 12 full case statuses including blocked cases, generates canonical native Vivado Tcl for the original FPGA part and requested clock, and audits every copied byte. Original HLS-local .dat/.mem ROM initialization assets are supported: the copy is byte-identical, individually hashed, staged both next to the original Verilog and in the Vivado working directory, and registered as Vivado memory-initialization sources. Missing referenced files, path escapes, symlinks, unknown embedded IP files/scripts and unsupported auxiliary formats cause an explicit error, never a silent incomplete source copy. Historical Vitis vendor receipts predating ROM hashing are recorded as legacy provenance, not falsely upgraded.

The Vivado runner requires `--execute-native --confirm-isolated-native`, finite case count and timeout, and refuses to run over prior reports/logs. Native physical audit checks fresh original vendor receipt + exact generated Verilog and Tcl hash, complete routed nets, parsed utilization, routed WNS and timing met. Negative WNS or missing/unattributed reports cannot qualify. Successful OOC route is NOT KV260 board I/O timing, a deployed bitstream, measured on-board latency, measured power or automatically achieved maximum Fmax. No original Vitis/Vivado tool is invoked in CPU tests or by default.

## Finish existing AUTO study on the original source pin FIRST

On the workstation, **before pulling newer repository commits**, verify the current checkout is source revision `8fe0ef31f6acc69c25902fe3f7ab94757e6e42bd`. Original case 000/001 already ran; do not rerun them. To finish only the 10 unrun cases:

```bash
git rev-parse HEAD
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/nchw_native_auto_v1 --audit-native
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/nchw_native_auto_v1 \
  --execute-native --confirm-isolated-native \
  --start-pair 1 --max-pairs 5
python -m fpgai_research.experiments.cst_nchw_native_export \
  --study build/nchw_native_auto_v1 \
  --output build/nchw_native_auto_v1_complete_report
```

The last export is create-only: select a *fresh* output directory if this name already exists. If original Vitis cases are partially attempted/failed, do not rerun or overwrite them; export their failures and ask for a source-pinned recovery protocol.

## Use NEW code for read-only policy comparison and separate physical preparations

Only **after** preserving and finishing the original source-pinned auto study:

```bash
git pull --ff-only origin main
python -m pip install --no-deps -e .

python -m fpgai_research.experiments.cst_nchw_cross_policy \
  --auto-study build/nchw_native_auto_v1 \
  --bram-study build/nchw_native_v1 \
  --output build/nchw_auto_vs_bram_original_v1

# Source-only: no Vivado calls. Does NOT mutate original Vitis directories.
python -m fpgai_research.experiments.cst_nchw_physical \
  --native-study build/nchw_native_v1 \
  --prepare-output build/nchw_bram_vivado_ooc_v1

python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_ooc_v1 --audit

# Preview first case only, no native tool:
python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_ooc_v1 --start-case 0 --max-cases 1

# Only with original Vivado installed in an isolated experiment checkout:
python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_ooc_v1 \
  --execute-native --confirm-isolated-native \
  --start-case 0 --max-cases 1

python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_ooc_v1 --audit
```

Create a **separate** routed study for auto rather than mixing source revisions or memory policies. Run source-verified, matched OOC physical results across both policies; do not infer physical BRAM savings or timing from HLS or from source C++ scratch element reductions.

## Scientific outcome and open gates

Original user-supplied Vitis transcripts show all 12 explicit-binding original cases qualified at E3, while the automatic inference study currently has only two source-attested E3 cases (0 inferred BRAM_18K and 975 cycles each). Neither original study has attached all vendor XML/RTL to this chat; software code can inspect original workstation files only when run there.

The CST Model IR research hypothesis must distinguish **semantic correctness**, **source event legality**, **vendor directives requested**, **actual vendor HLS timing/resources**, and **routed physical feasibility**. We must not claim an optimal memory architecture, novel-firstness or a deployable design from the current tiny bounded NCHW subset. Full matched trained-model/peer-artifact experiments and complete E4 physical data are required for publication, and RL training remains frozen.

## 2026-11-02 real Vitis-generated ROM .dat packaging fix

The user-provided original workstation transcript reports that the first explicit BRAM physical-study preparation stopped with:

```text
ValueError: UNSUPPORTED_HLS_MEMORY_INIT_ASSETS:
fpgai_model_fpgai_model_Pipeline_VITIS_LOOP_39_10_tensor_1_ROM_AUTO_1R.dat
```

This was an intentional **source-incomplete** fail-closed error, **not** a Vitis CSim/CSynth or Vivado place-and-route failure. The subsequent `Physical study root unavailable` only occurred because preparation had not produced a study directory.

The new source-pinned OOC bridge supports original local Vitis `.dat` and `.mem` ROM images via `toolchains/vivado/rtl_inputs.py`, the original Vitis/physical launch receipts and a Vivado Tcl memory-file registry. It preserves original bytes, SHA-256 in the physical manifest, staged bytes adjacent to Verilog and under Vivado's cwd for `$readmemh`/`$readmemb`, and rejects altered/missing/path-traversing references or IP setup scripts. Only an **actual original Vivado run** can establish the real tool accepts the package and routed quality; no such run is claimed here. AMD UG901 2023.2 documents `$readmemh`/`$readmemb` ROM init; AMD UG1399 2023.2 warns standalone Verilog is inadequate where additional vendor IP setup is required.

After updating the repository code, use a **new** output directory (or verify the old attempted output does not exist) and do not overwrite original source-pinned Vitis folders:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m pip install --no-deps -e .

python -m fpgai_research.experiments.cst_nchw_physical \
  --native-study build/nchw_native_v1 \
  --prepare-output build/nchw_bram_vivado_rom_v2

python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_rom_v2 --audit

python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_rom_v2 \
  --start-case 0 --max-cases 1

# Explicit optional REAL Vivado, one bounded case and preserved raw reports:
python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_rom_v2 \
  --execute-native --confirm-isolated-native \
  --start-case 0 --max-cases 1

# Full denominator, even while most physical cases are incomplete:
python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_rom_v2 \
  --export-output build/nchw_bram_vivado_rom_v2_report
```

Do NOT run Vitis again on the completed original `bram_2p` cases. The remaining ten `auto` HLS cases require a separately isolated checkout of their exact original source revision `8fe0ef31...`; the newer `main` intentionally rejects original-case native execution at a mismatched Git revision. Cross-policy read-only auditing on newer `main` is allowed because it checks every original source pair's bytes.

**Gate:** CPU regression and original experiment-series CI must pass on the new code revision. Vendor Vivado and ROM initialization success are unverified until executed on the workstation.



## 2026-11-02 follow-up: original Vitis `./ROM.dat` literal repaired

The next original BRAM preparation was blocked by
`UNSAFE_OR_UNSUPPORTED_RTL_READMEM_PATH: ./fpgai_model_..._ROM_AUTO_1R.dat`.
This is a **single local current-directory prefix** emitted by Vitis, not
an absolute path or traversal. The source inventory now accepts exactly
`name.dat` or `./name.dat` (likewise `.mem`) and canonicalizes both to
the same source-directory basename. It still refuses `../`, `./../`,
`././`, repeated separators, nested directories, backslashes, symlinks
and non-ROM files. Original Verilog bytes are NOT rewritten; their SHA-256
and the original ROM payload SHA-256 remain independently checked.
New regression tests exercise real-literal form, exact staged bytes,
audit/dry-run and negative traversal/malformed paths. This change does
**not** claim original Vivado execution or physical qualification.

The previous `build/nchw_bram_vivado_rom_v2` was not created according
to the supplied traceback; verify locally with `test -e` before reusing
a name. For a clean, create-only next attempt use:

```bash
git pull --ff-only origin main
python -m pip install --no-deps -e .
python -m pytest -q tests/test_cst_nchw_physical.py

python -m fpgai_research.experiments.cst_nchw_physical \\
  --native-study build/nchw_native_v1 \\
  --prepare-output build/nchw_bram_vivado_rom_v3

python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v3 --audit

python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v3 --start-case 0 --max-cases 1
```

These commands perform zero Vivado calls; explicit original Vivado
execution remains an isolated, separately authorized foreground step.
Leave the source-pinned `build/nchw_native_v1` HLS cases untouched.


## 2026-11-02 native Vivado case 000: exit 1, immutable failure diagnostics

The first original Vivado 2023.2 OOC invocation for
`build/nchw_bram_vivado_rom_v3/case_000` exited with **return code 1
after about seven seconds**. The read-only E4 audit correctly reported
`REJECTED_VENDOR_PROVENANCE`, one original invocation receipt, zero
routed-qualified cases, and 11 prepared but unattempted cases. This is
**not evidence of timing, DRC, resource, Verilog or ROM failure** until
the original Vivado logs identify the precise tool error. The original
case_000 vendor receipt, logs, ROM bytes and Tcl must not be overwritten.
The original `v3` directory remains source-pinned and protected.

New *read-only* command for that immutable failed run:

```bash
python -m fpgai_research.experiments.cst_nchw_vivado_diagnose \\
  --study build/nchw_bram_vivado_rom_v3 --case 0
```

The diagnostic runs **zero vendor processes**, writes **zero files**,
and prints original exit code, stderr/stdout log paths, SHA-256 matches
against the original invocation receipt, and at most 16 recent matching
error lines and 12 last lines per hash-matched log. If a log was
modified after invocation, its contents are deliberately suppressed and
its hash mismatch reported instead. The generated diagnostic is not E4
physical evidence and cannot qualify a case. Original compiler source
and the old immutable `v3` evidence are not modified.

**After inspecting the precise Tcl/Vivado error**, fix source-only Tcl
integration, add negative tests, and prepare a *fresh* versioned physical
study directory from `build/nchw_native_v1`; do not rerun the failed
case inside `v3`, copy a failed vendor receipt as a new run, or weaken
hash-based qualification. A source-fixed new study must receive its
own manifest and original Vivado invocation. All remaining 11 prepared
cases stay unattempted for now.


## 2026-11-02 Vivado 2023.2 `filemgmt 20-1707` Tcl repair

The **original, immutable** `build/nchw_bram_vivado_rom_v3/case_000`
Vivado invocation failed after seven seconds, exit 1. Independently
SHA-256-verified original stderr identifies the *exact* failure:
`ERROR: [filemgmt 20-1707] The file type 'Data Files' is not user changeable.`
The source-generated Tcl had successfully registered the original HLS
`.dat` file with `add_files`, then unsuccessfully attempted
`set_property file_type {Memory Initialization Files}`.

**Source-only repair:** keep the byte-identical `.dat/.mem` staging
beside the original RTL and in Vivado's working directory; keep the
`add_files` and `get_files` single-file registration checks; stop
trying to change Vivado's native `FILE_TYPE` of `Data Files`.
The original `$readmemh/$readmemb` literals remain verbatim and resolve
to original hash-checked local files. This removes the demonstrated Tcl
error but DOES NOT yet prove that synthesis, ROM initialization, route,
DRC, timing or report qualification succeeds. Original `v3` source,
logs, Tcl and failed receipt are never modified. Standalone Vivado still
requires an original tool run, not a CPU fixture.

Since the research package is already editable-installed in the active
`.venv`, do **not** reinstall for every source-only `git pull`.
Only reinstall if packaging/dependencies or the environment change.

Use a **new** create-only `v4` physical study (never overwrite `v3`):
```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m pytest -q tests/test_cst_nchw_physical.py tests/test_vendor_tools.py

python -m fpgai_research.experiments.cst_nchw_physical \\
  --native-study build/nchw_native_v1 \\
  --prepare-output build/nchw_bram_vivado_rom_v4

python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v4 --audit

python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v4 \\
  --start-case 0 --max-cases 1

# After CPU/audit gates: EXPLICIT bounded original Vivado, case 000 only.
python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v4 \\
  --execute-native --confirm-isolated-native \\
  --start-case 0 --max-cases 1

python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v4 --audit
```

Any further failure must be reported from the *new* original Vivado logs;
do not equate a clean CPU check, successful vendor exit or a read-only
diagnostic with routed E4 evidence. The source-pinned Vitis study is not
re-executed.


## 2026-11-02 v4 original native outcome and ROM-provenance remediation

The real workstation `build/nchw_bram_vivado_rom_v4/case_000`
finished Vivado **exit 0 in about 98 seconds**, after staging
all 12 qualified original HLS source packages. The next independent
audit raised `Physical study ROM bytes missing, altered or duplicated`
before reporting physical resources, WNS or DRC. **No E4 case or
physical pair qualified.** This mismatch does not yet establish whether
an original ROM was changed/missing or Vivado created an additional
`.dat/.mem` asset; both require explicit reporting.

The source-only remediation (after v4's original vendor execution):

- `toolchains/vivado/rtl_inputs.py::compare_staged_roms` checks
  exact original filename/hash and records missing, modified, invalid
  and *unattributed extra* ROM files separately without changing them.
- `experiments/cst_nchw_physical.py::audit_routed_study` now returns
  a **per-case rejected status** with the actual file inventory and
  original invocation presence, rather than raising away useful
  evidence. No extra file is automatically trusted or treated as
  original HLS input.
- `toolchains/runner.py::run_tool` now independently snapshots the
  **Vivado working-directory** ROM before and after execution, not
  only the RTL source-directory ROM. The new native receipt includes
  both, and E4 rejects missing execution-boundary attestations. This
  is prospective only; it does not retroactively rewrite v4 receipts.
- CPU negative tests use fake vendor shell processes and are **not**
  original Vivado/HLS measurements.

The correct immediate action is **read-only** diagnosis of the original
v4 folder. Do **not** delete, rename, regenerate, re-run or overwrite
any original vendor case just because a new audit version was committed:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m pytest -q tests/test_cst_nchw_physical.py tests/test_vendor_tools.py

python -m fpgai_research.experiments.cst_nchw_vivado_diagnose \\
  --study build/nchw_bram_vivado_rom_v4 --case 0

python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v4 --audit
```

The output must distinguish `modified_original_roms`,
`missing_original_roms`, `unattributed_rom_files` and
`invalid_rom_files`. The **original** v4 receipt may lack the new
per-working-directory execution hash fields, so no E4 promotion is
permitted from source-only software changes. A subsequent toolchain
repair, if genuinely needed after the exact inventory is known,
requires a newly prepared create-only source-pinned v5 (or later)
directory. No more automatic Vivado runs until the v4 diagnosis
identifies the specific cause.

Paper-level hypotheses, independent oracles and actual peer-system
artifact comparisons are governed by `SCIENTIFIC_STUDY_PROTOCOL.md`,
not by these toolchain fixes.


## 2026-11-02 v4 workstation failure closure and v5 prospective gate

The workstation re-ran exact CPU tests at the initial v4-fix
revision: **4 failed, 34 passed focused; 4 failed, 1145 passed full**.
Three failures were a Python `SyntaxError` in the new **read-only
diagnostic import** (a literal `\\n` escaped sequence, not a newline).
One was a CPU-only fake Vivado shell fixture whose shebang contained
literal `\\n` sequences and therefore could not execute. These are
confirmed new-source/test bugs, unrelated to the original native RTL.
Both malformed-text fixes were committed; do **not** report tests
passing until the revised source runs.

The original `v4/case_000` physical audit now explicitly finds
`missing_original_roms=[fpgai_model_fpgai_model_Pipeline_VITIS_LOOP_39_10_tensor_1_ROM_AUTO_1R.dat]`
in the Vivado **working directory**. Its original source-directory
hash remains present in the original case package; 11 other staged
case working directories still match their expected ROMs.
The completed `v4` invocation is **unqualified** and immutable; do not
restore the missing working ROM inside v4, post-inject new receipt fields
or re-run the case. Its original vendor exit 0 and physical reports
can be analyzed descriptively only, never as qualified E4 data.

The v4 Tcl registered the same working-directory ROM using `add_files`.
The revised **prospective** Tcl registers the separately staged
**source-directory** ROM instead (whose hash the runner checks), while
independently verifying the original local `$readmemh` copy exists
before synthesis. Original ROM bytes and $readmemh literal are not
modified. Whether this prevents disappearance is **not proven** by
static tests; the first original Vivado v5 call is the empirical gate.
The new receipt records both source and cwd ROM inventories before and
after the run, and the independent auditor refuses any missing or
modified ROM evidence. This approach must be revised if v5 reveals
that Vivado legitimately consumes the local file or still removes it;
do not silently waive post-execution input integrity.

Suggested gated commands from the existing editable virtualenv:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_nchw_physical.py tests/test_vendor_tools.py
python -m pytest -q

# Only after CPU tests pass: prepare NEW create-only v5 package, no vendor calls
python -m fpgai_research.experiments.cst_nchw_physical \\
  --native-study build/nchw_native_v1 \\
  --prepare-output build/nchw_bram_vivado_rom_v5
python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v5 --audit
```

The new v5 original native Vivado execution, case 000 only, must be a
separate deliberate opt-in after this preflight. Do not run all 12 in a
batch before observing the original first case's RAM and report
integrity. Only if the first qualified case passes all E4 gates should
we attempt its matched case_001, then other pairs. Physical effects
must remain pairwise and source-matched.

**Scientific note:** this is a research experiment infrastructure
repair, not yet evidence for CST mapping-conditioned lifetime novelty.
That thesis requires the independent certificate discrimination,
Phi ablations, executable source refinement, external peer artifacts
and matched E3/E4 evidence in `SCIENTIFIC_STUDY_PROTOCOL.md`.


## 2026-11-02 actual v5 route — accepted input, rejected physical claim

The user's original workstation log at the **v5** package shows
`python -m compileall -q src/fpgai_research tests` completed without
diagnostics, **40/40 focused tests passed** and **1151/1151 full
repository tests passed**. A fresh 12-case original RTL source-pinned
Vivado OOC v5 package was prepared/audited with 0 vendor receipts and
12 `PREPARED_NOT_RUN` cases.

The user explicitly executed only `v5/case_000`. Original Vivado
2023.2 completed with returncode **0 after about 95 seconds**.
Nevertheless the original staged working-directory ROM file
`fpgai_model_fpgai_model_Pipeline_VITIS_LOOP_39_10_tensor_1_ROM_AUTO_1R.dat`
was missing *again*. Its expected SHA-256 was
`6544d2b4ab5503a4ad273984aee8c9e1993f3963cb98eb943b3b61a98f6e55d8`.
The runner correctly reports
`original_vendor_process_success=false`; audit reports
`REJECTED_ORIGINAL_STAGED_ROM_MODIFIED_OR_MISSING`, one original vendor
receipt, **0/12 E4-qualified cases** and 11 `PREPARED_NOT_RUN`.
Switching `add_files` registration from the cwd to the original RTL
directory did **NOT** prevent the disappearance. The mechanism is
**unknown**: Vivado source management, Tcl script side effect or vendor
processing cannot be distinguished from the reported output alone.
No timing, BRAM, DRC or scientific performance claim is warranted.

Neither v4 nor v5 may be modified, reconstructed, deleted, filled with
replacement ROM bytes, or rerun. Their vendor receipts and audit failures
are important negative reproducibility evidence. Before a **v6**
experiment, inspect only the original v5 SHA-256-attested Vivado logs and
any actual project-internal ROM copies:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_nchw_physical.py tests/test_vendor_tools.py

# ORIGINAL v5 evidence read-only; no vendor calls, no files written:
python -m fpgai_research.experiments.cst_nchw_vivado_diagnose \\
  --study build/nchw_bram_vivado_rom_v5 --case 0
```

The extended read-only diagnostic displays `rom_activity_lines` only
when original stdout/stderr hashes match the original receipt and
enumerates bounded `research_implementation/` project ROM copies with
their SHA-256 matches. These copies are merely potential clues: their
presence cannot retroactively qualify a missing cwd input or prove
what data synthesis used. Do not create v6 yet until the observed
original vendor log/ROM behavior yields a bounded, testable Tcl repair.

**Publication scope:** Software E0/E1 checks and qualified original Vitis
E3 can be discussed with exact primary receipts, but physical E4
comparison remains zero; v5's successful Vivado **process** is not a
qualified circuit. The central CST-IR scientific novelty still requires
a Phi-dependent element-level counterexample with independent original
source refinement and direct prior-art artifact comparison.


## 2026-11-02 v5 original Vivado log — causal triage outcome

The real hash-matched `v5/case_000` stdout contains **Vivado
`Synth 8-3876`: `$readmem data file './fpgai_model_fpgai_model_Pipeline_VITIS_LOOP_39_10_tensor_1_ROM_AUTO_1R.dat' is read successfully`**.
The same original output lists the 32×8 ROM as **LUT**, and reports
generation of `reports/routed_design.dcp` before Vivado exited 0 in
approximately 95 s. The original stderr is an empty file with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
A read-only project-file search found no matching `.dat` under the
`research_implementation/` project directory. **No evidence identifies
which Vivado stage removed the cwd file.** Neither the original
`v4` nor `v5` receipt is rewritten or E4-promoted.

### New v6-only original-read witness protocol

A source-level research protocol update makes future physical studies
**prospectively, explicitly** eligible for a *different evidence regime*:
`ATTESTED_VENDOR_READ_POST_RUN_ABSENCE_V1`. The prospective study
manifest carries the protocol version; its newly generated original
Vivado Tcl carries `ROM_READ_ATTESTATION_PROTOCOL_V1`. The frozen
`v4/v5` studies lack both and retain their historical rejected status.

**This is NOT a toolchain fix, proof of vendor cleanup behavior, or a
waiver of source integrity.** A new original vendor receipt can attest
post-run ROM absence only when ALL of the following hold:

1. Before the process, every cwd ROM was a regular, correctly named,
   original SHA-256-matching file. Original source ROM/RTL also has
   source-pinned hashes before and after the process.
2. After the process, no original cwd ROM is modified and no
   unattributed/unsafe ROM appears; the only difference is that some
   original, **literal $readmem input** files are absent.
3. The original captured Vivado stdout is SHA-256-matched to the
   process receipt. For every absent file, Vivado's **exact**
   `INFO: [Synth 8-3876] $readmem data file './<name>' is read
   successfully` message must appear, matching the original
   independently parsed readmem basename. Reports without this
   message and indirect/parameterized ROM filenames do not qualify.
4. The vendor process really exited 0, the original Tcl and all
   project/hash checks pass, and the auditor independently recomputes
   the post-run witness from read-only bytes. A fabricated or
   subsequently modified stdout, extra ROM file, changed ROM byte,
   unknown name or missing pre-run receipt remains rejected.
5. **Separate hard E4 gates still apply:** fresh route/timing/
   utilization/DRC reports with original hashes; complete routing,
   verified clean DRC, nonnegative WNS and complete device
   utilization. A successful `readmem` witness alone merely advances
   the *provenance* stage; it does not itself qualify physical QoR.

The run report labels `rom_input_evidence_mode` as
`PRE_RUN_HASH_AND_VENDOR_REPORTED_READ_POST_ABSENCE`, distinct from
`ORIGINAL_SOURCE_AND_WORKING_ROM_STABLE`. This provenance distinction
must be included in the final paper tables and limitations. It does
NOT prove the internal synthesized ROM content by a second independent
gate-level memory-content check. The original hash before execution
and Vivado's own declared read success are the extent of this
evidence. Original source remains untouched.

**Gated create-only v6 study:**

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_vivado_postrun_rom_attestation.py \\
  tests/test_cst_nchw_physical.py tests/test_vendor_tools.py
python -m pytest -q

# ZERO vendor calls until explicit execution:
python -m fpgai_research.experiments.cst_nchw_physical \\
  --native-study build/nchw_native_v1 \\
  --prepare-output build/nchw_bram_vivado_rom_v6
python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v6 --audit

# Explicit ONE original Vivado call only after preflight succeeds:
python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v6 \\
  --execute-native --confirm-isolated-native \\
  --start-case 0 --max-cases 1

python -m fpgai_research.experiments.cst_nchw_physical \\
  --study build/nchw_bram_vivado_rom_v6 --audit
```

The strict E4 study never counts a warning-only case as E4. Under the
new prospective **exploratory warning-observation** policy, case_001 may
be considered only if case_000 is independently either strict E4 or
`OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS`, its original
ROM input mode is verified and it has no unmatched or unexpected DRC
rules. All matched exploratory cases retain their exact warning lists.
No case may be rerun or repaired in place.
If the first case fails timing/DRC/resource/report parsing, preserve
its failures and repair the prospective protocol in a **new** study;
do not rerun a historic physical case.

Only six fully qualified matched pairs (12 cases) can establish the
prespecified physical comparison; unmatched or rejected pairs remain
UNKNOWN rather than contributing zeros. For a publishable novelty
claim, the separate Phi-conditioned certificate/source-refinement and
peer-artifact experiments remain essential.


### Read-only v5 report triage before running a new study

The `cst_nchw_vivado_diagnose` command now includes
`physical_report_preview` **only if every original report SHA-256
matches the original Vivado invocation receipt**. A matched preview
contains the real routed-report parser's routing, WNS, LUT/FF/DSP/BRAM
and DRC fields with `e4_qualified=false`; it is explicitly
*descriptive*, not a promotion of the source-rejected original v5.
Failed report parsing emits `UNQUALIFIED_REPORTS_UNPARSEABLE` and its
specific error; a stale report emits
`UNQUALIFIED_REPORT_HASH_MISMATCH` with no numerical values.

Run this read-only gate and inspect its output **before** another
original Vivado invocation:

```bash
python -m fpgai_research.experiments.cst_nchw_vivado_diagnose \\
  --study build/nchw_bram_vivado_rom_v5 --case 0
```

If v5 already reveals route/DRC/timing/parser problems, address the
independent source/report issue before starting prospective v6.

## 2026-11-02 v5 physical report triage — original hash-attested numbers

The workstation reran **57/57 focused and 1168/1168 full CPU tests**.
The `v5/case_000` read-only physical preview confirmed that every
original route/timing/utilization/DRC report SHA-256 matches the original
Vivado invocation receipt. Vivado reports: **fully routed routable
nets**, `WNS=+20.724 ns` at the requested `32.0 ns` clock,
`CLB LUTs=4995`, `FF=2783`, `DSP=25`, `Block RAM Tile=2.5`,
`URAM=0`. These are original *observations*, NOT qualified E4 or
measured power/board-level Fmax. The original `v5` ROM disappeared
from cwd after a successful source-reported `$readmem` read, so
the original `v5` remains provenance-rejected.

More importantly, original DRC was **not clean**: `55` warnings,
`0` reported critical warnings/errors, distributed as
`DPIP-2=16`, `DPOP-3=19`, `DPOP-4=19`,
`RTSTAT-10=1`. The vendor-reported route-counter proof and a
`RTSTAT-10` warning must both be retained; the latter may mean
non-routable-load nets, **not** necessarily a conflict with complete
routing for all *routable* nets. Do not infer whether a warning is
harmless or fixable merely from its rule name. The `DPIP/DPOP`
family generally concerns DSP input/output pipelining and
performance trade-offs, but this case's exact original DRC text is
the authoritative evidence.

The `cst_nchw_vivado_diagnose` read-only tool now emits the first
two original, SHA-256-attested detail examples for each actual DRC
rule under `original_drc_rule_details` when the report hashes
match. **Do not suppress, downgrade or waive these warnings in
a new toolchain script to manufacture a clean DRC.** Do not edit
`v5`. Inspect the warning details and determine a source-matched
prospective physical qualification policy, without changing
architecture/numerical schedule in only one member of a pair.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_nchw_physical.py \\
  tests/test_vivado_postrun_rom_attestation.py \\
  tests/test_vendor_tools.py
python -m pytest -q
python -m fpgai_research.experiments.cst_nchw_vivado_diagnose \\
  --study build/nchw_bram_vivado_rom_v5 --case 0
```

**Next gate:** source-dependent DRC warnings require scientific
attribution and an explicit treatment; the v6 ROM witness alone will
not make case_000 E4-qualified because the strict zero-warning DRC
gate would still reject. If the goal is reproducible routed timing/
utilization under disclosed *warning-only* DRC, a separate
preregistered **descriptive, warning-labeled tier** could be
studied, but it must not be silently called
`QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4` or retroactively
count historical pairs. Alternatively redesign the source-matched
HLS architecture and produce a genuinely new paired native
Vitis campaign to close DRC. The original v5 result contains
only one observed member; **no paired performance effect** exists.

## 2026-11-02 v5 original DRC detail closure, exploratory physical policy

The latest original workstation gate passed **59/59 focused tests** and
**1170/1170 full tests** at the code revision before this new
prospective protocol. The hash-verified DRC report explains all
**55 warnings**:

- **16 DPIP-2**: non-pipelined DSP48 input; vendor suggests input
  pipelining for potential frequency improvement
- **19 DPOP-3**: `PREG=0` DSP48 output; vendor suggests output
  pipelining and notes possible power benefits
- **19 DPOP-4**: `MREG=0` DSP multiplier stage; vendor suggests
  pipelining for performance and potential power
- **1 RTSTAT-10**: `30` nets with no routable loads, including
  `ap_idle` and `ap_ready` source control nets

All 55 are Vendor-severity **Warning**, with zero Critical Warnings
and zero Errors. The original Vivado route status independently
reported complete routing of routable nets and +20.724ns WNS.
**Neither timing closure nor the Warning severity proves functional
correctness, board deployment, power improvement or DRC cleanliness.**
Pipelining original HLS-generated DSPs now would change experiment
architecture/latency and violate matched comparisons. We preserve
the original RTL and these findings rather than masking them.

### Newly implemented experimental measurement distinction

New create-only physical study manifests include
`DISCLOSED_OOC_NONCRITICAL_WARNING_OBSERVATION_V1` and the exact
frozen, **pilot-informed** allowlist `DPIP-2, DPOP-3, DPOP-4,
RTSTAT-10`. New original Tcl has
`DRC_WARNING_OBSERVATION_PROTOCOL_V1`. No warning is waived
or its severity changed.

- **Strict primary E4** `QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4`:
  source/ROM/receipt/report hashes all original; routable nets
  completely routed; nonnegative WNS; all target resource fields;
  DRC summary verified, **zero violations**. Unchanged.
- **Exploratory observation only**
  `OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS`:
  same source, provenance, routing, hash, timing and resource gates,
  but nonzero DRC violations. Every violation must be a positive-count
  **Warning** in the four allowlisted classes; no Critical Warning,
  Error, unconstrained I/O, conflicting summary or other rule. The
  exact count and rule list is reported. `routed_qualified=false`
  remains explicit; `routed_warning_observation=true`.
- **Failed or unknown**: unexpected warning, severity, count,
  source hash, ROM witness, report hash, missing/unrouted net,
  negative WNS or missing resources results in neither tier.
  `v4/v5` do not carry the prospective policy and **are never
  retrospectively promoted**.

The create-only export now writes `cases.csv`,
`matched_pairs.json`, `pairs.csv`, original audit and summary
with separate, non-overlapping *strict E4 pair differences* and
*descriptive warning-labeled paired differences*. Missing cases
remain NULL and all 12 cases/six pairs remain in denominators.
The allowlist was derived **after seeing the original v5 pilot**,
so the exploratory tier is **not** a blinded independent test or a
primary preregistered effect claim. Report it as
pilot-informed secondary analysis.

### Next experiment gate

Only after the new focused/full CPU tests pass on the workstation,
prepare a **new** `build/nchw_bram_vivado_rom_v6` (do not reuse if
it already exists). Its first native Vivado invocation is case_000
only, and run the independent audit afterward. The expected
strict E4 result may remain unqualified because vendor DSP warnings
are architecture-dependent; a valid, warning-disclosed observation
is still useful for the *exploratory*, not primary E4, result.

Any case_001 execution must follow a separate review of the original
case_000 audit, hash/report/DRC counters and classification.
Only then continue a matched pair. Every physical run remains
opt-in, bounded, foreground and never reuses previous original
study paths.

The genuine paper novelty remains mapping-dependent `Phi` per-element
availability/reduction/last-use contracts and independently reproduced
certificate discrimination, **not** a favorable routed WNS or a
vendor warning exemption.
