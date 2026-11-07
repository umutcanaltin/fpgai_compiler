---
name: Compiler bug report
about: Report a reproducible compiler or documentation defect
title: "[Bug] "
labels: ["bug"]
assignees: []
---

### Summary
Describe the observed compiler behavior without including private models, credentials, vendor logs or datasets.

### Reproduction
Provide the exact FPGAI Git commit, Python version, installed optional extras, minimal **synthetic** model/source input, and a CPU-only command if possible.

### Expected and actual behavior
Explain the expected result and observed status/exception, with relevant non-sensitive excerpts.

### Evidence level
Mark the highest actual level you verified: E0 source contract, E1 generated/host C++, E2 Vitis C simulation, E3 HLS estimate, or E4 routed physical (only if timing/DRC-qualified). Do not label an HLS estimate or negative WNS as board performance.

### Scope and safety
- [ ] Minimal reproduction uses publishable, synthetic data
- [ ] No private weights, non-redistributable vendor assets, secrets, licenses or personal data
- [ ] No original frozen experiment folder was modified
- [ ] Any native tool invocation was deliberate and locally authorized

For suspected security issues, follow [SECURITY.md](../../SECURITY.md) instead of opening a public issue.
