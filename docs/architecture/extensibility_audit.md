# FPGAI Extensibility Audit Architecture

## Goal

The first ecosystem sprint records where contributors currently need to edit FPGAI core files and defines the migration order toward external research packages.

The audit is read-only. It must not import plugin modules, execute package code, alter compiler behavior, or create build artifacts.

## Questions answered

For every extension family, the audit records:

- current owner files;
- current mechanism;
- whether an external contribution is possible without a core edit;
- inference and training coverage;
- central-dispatch indicators;
- recommended registry or contract;
- migration priority;
- FPGAI research ownership;
- Morfics production ownership;
- readiness for deterministic Morfics backend invocation.

## Current architectural findings

The repository already contains useful foundations, especially `fpgai/layers/registry.py`, structured runtime-package generation, board metadata, and explicit inference/training contracts. These should be evolved rather than replaced.

The main contributor barriers are distributed dispatch and ownership across:

- ONNX import and canonicalization;
- layer capability metadata;
- inference HLS generation;
- training HLS generation;
- training numeric reference execution;
- resource and architecture estimation;
- optimizer and loss handling;
- boards, backends, memory policies, and transports.

Adding a new ONNX operator can therefore require changes to several central modules. The target is one package declaration plus clearly separated optional contracts for import, IR semantics, numeric reference, inference, training, implementation, estimation, and validation.

## Research-use boundary

Community implementations compiled by FPGAI are research and benchmarking assets. They may be simulated, synthesized, implemented, tested on hardware, compared numerically, and published reproducibly. Commercial deployment and production operation are Morfics responsibilities.

## Running the audit

```bash
python -m fpgai.devtools.extensibility_audit
```

Generated reports are written under `dev_audits/extensibility/` by default and should not be committed as source files.

## Next approval gate

The audit does not create registries. After its findings are reviewed, the next proposed sprint must define `fpgai.package/v1`, package identity, capabilities, validation levels, and safe metadata-only discovery. Implementation begins only after approval.
