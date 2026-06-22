# FPGAI Evidence Directory

This directory contains reviewer-facing evidence derived from experiment outputs.

## Current contents

```text
evidence/reproducibility/
```

The reproducibility folder contains the claim support matrix.

## Generate reproducibility evidence

```bash
python scripts/collect_claim_support_v2.py \
  --out evidence/reproducibility
```

## Evidence status convention

```text
READY   = at least one supporting artifact path exists
MISSING = no supporting artifact path exists; do not use this claim yet
```

## Evidence-first manuscript rule

Only use claims that are supported by generated code, experiment logs, HLS/Vivado output, or collected evidence tables.

Do not overclaim planned features.
