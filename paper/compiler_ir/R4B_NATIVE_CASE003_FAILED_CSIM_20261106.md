# R4b — original case 003 native CSim failure (preserved negative result)

**Evidence:** user's workstation terminal output, 6 November 2026. Source revision of R4a original `fe99d627a5a16ce75e5d4df4542bb0f0f93e6ab0`. Original raw files live under `build/r4a_row_retained_paper_20261106_01/case_003` on the user's workstation. We **have not inspected the raw vendor logs here**; we must not invent numerical failure counts, max errors or causes.

## What ran

- R4a original full paper E1: 5/5 real GCC matched pairs completed. This remains qualified at host-source level only.
- R4b read-only audit: 5 original source pairs, 10 SHA-checked HLS packages, no prior vendor invocation receipts, zero E2/E3 qualified cases, all native outcomes unknown before this new pilot.
- R4b dry run: selected `case_003` only, two prepared Vitis arms, no calls.
- R4b opt-in: **two original Vitis HLS 2023.2 processes ran, both exited 0**, at approximately 16 s and 15 s respectively.
- **Both original arms failed E2 numerical CSim** according to the source-pinned qualification:
  - `case_003/full_H`: `FAILED_ORIGINAL_CSIM`, E2 false, E3 false;
  - `case_003/row_H`: `FAILED_ORIGINAL_CSIM`, E2 false, E3 false.
- Subsequent read-only audit: **two** native invocation receipts (both in case 003), **zero** E2-accepted or E3-qualified arms, **zero** qualified HLS pairs. Other four pairs were *not run*, not failures.

**No BRAM18K, LUT, FF, DSP, cycle or physical FPGA comparison is scientifically qualified.** Vendor exit 0 is not numerical success. A failure in both arms does **not** isolate a row-retention regression.

## Numerical hypothesis to test, not an asserted cause

The original package uses the canonical `ap_fixed<16,6,AP_RND,AP_SAT>` model (10 fractional bits), and its tolerance is `0.02` against a floating-point FunctionalModel/NumPy reference. The original MatMul fixture's coefficients increase in magnitude with index. Even if values never saturate at ±32, fixed-point quantization/rounding can cause a larger output difference than the tolerance, especially for the (m=16,k=4,n=16,q=8) case. **Do not increase the acceptance tolerance or change precision retroactively in the frozen source-pinned case.** The original logs must establish whether the actual CSim metric is excessive, testbench or source failed, or something else.

## Read-only extraction command (no tool launch, no original file edits)

Run from the existing compiler checkout. The exact original case files are used; do not copy and modify them.

```bash
cd ~/Desktop/github_projects/fpgai_compiler
python - <<'PY'
import json
from pathlib import Path
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.report import discover_csim_success
root = Path("build/r4a_row_retained_paper_20261106_01/case_003")
for arm in ("full_H", "row_H"):
    case = root / arm / "vitis_source"
    meta = json.loads((case/"manifest.json").read_text())
    result = summarize_case(str(case))
    csim = discover_csim_success(case)
    print("\n====", arm, "====")
    print(json.dumps({
        "source_sha256": result["source_sha256"],
        "precision_bits": meta.get("precision_bits"),
        "integer_bits": meta.get("integer_bits"),
        "numerical_tolerance": meta.get("numerical_tolerance"),
        "csim_passed": result["csim_passed"],
        "csim_receipt": result.get("csim_receipt"),
        "csim_parser": csim,
        "warnings": result.get("warnings"),
        "csynth_report_exists": (
            case/"fpgai_hls_project/solution1/syn/report/fpgai_model_csynth.xml"
        ).is_file(),
    }, indent=2, sort_keys=True))
    log = case/"logs/vitis.stdout.log"
    if log.is_file():
        selected = [line for line in log.read_text(errors="replace").splitlines()
                    if any(token in line for token in (
                        "FPGAI_METRICS", "total_failures",
                        "FPGAI_CSIM_STATUS", "CSim done", "ERROR:", "Error:"
                    ))]
        print("Selected original stdout lines:")
        print("\n".join(selected[-25:]))
PY
```

If the original CSim metrics show positive numerical `failures`, record their **actual** counts/max error/RMSE and compare the two arms before choosing any correction. If metrics or markers are missing, diagnose the original tool/testbench log and do not call it a precision failure. If both arms fail due to common numerical precision, any follow-up accuracy/bitwidth study must be a **new preregistered study with new output path and source revision**, preserving all case 003 files. Even if a later case passes, keep case 003 in the original denominator.

## Next paper decision

A successful 26/26 source test suite and 5/5 E1 GCC source study are still valid. The native experiment has revealed an **E1-to-E2 numerical qualification gap**. Confirm the mechanism first. Do not run cases 000–002/004 or rewrite old testbench tolerance merely to obtain qualified HLS resource deltas.
