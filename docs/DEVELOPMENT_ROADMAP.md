# FPGAI development roadmap

FPGAI is developed as a reusable compiler and research platform rather than as a sequence of public iteration labels. New work should extend existing owners, preserve stable contracts, and remain selectable through explicit configuration where architectural alternatives exist.

## Mixed-backend composition

1. Validate grouped multi-input and multi-output VHDL contracts in real mixed-language DAGs.
2. Generalize DAG composition beyond the maintained fork/join topology.
3. Add multi-port HLS physical interfaces without guessing ambiguous AXI-stream groups.
4. Add independent-port handshake policies alongside grouped transactions.
5. Characterize buffering, latency, resource, and timing effects of each bridge policy.

## Model and operator coverage

1. Expand maintained CNN and residual-network coverage.
2. Add attention and transformer IR/operator foundations.
3. Extend operator compatibility reports with implementation/backend coverage.
4. Add maintained example models and configuration files for each supported architecture class.

## Training

1. Extend external implementation contracts with training-forward, backward-input, parameter-gradient, bias-gradient, and optimizer-update capabilities.
2. Validate external training implementations numerically against Python/reference execution.
3. Carry training tensors, gradients, optimizer state, memory residency, and transport through the same physical reporting model used for inference.

## Backends and boards

1. Continue VHDL as a first-class backend alongside C++/HLS.
2. Expand board contracts and toolchain adapters through registries rather than central compiler edits.
3. Add additional FPGA-vendor backends behind stable IR, implementation, build, and report contracts.

## Research characterization

1. Keep compiler estimates, HLS synthesis results, Vivado implementation results, and board measurements separate.
2. Use YAML-selectable architecture mechanisms for comparative experiments.
3. Preserve complete provenance for package, implementation, toolchain, clock, precision, memory, and generated artifacts.
