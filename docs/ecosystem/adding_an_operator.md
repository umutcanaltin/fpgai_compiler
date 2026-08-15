# Adding an FPGAI Research Operator

This document describes the target contributor path. E3A establishes the contract model; external execution and importer registration are implemented in later work.

A research operator contribution will eventually provide:

```text
my_operator/
├── fpgai.yaml
├── README.md
├── LICENSE
├── python/
│   ├── operator.py
│   ├── onnx_import.py
│   ├── shape_inference.py
│   └── reference.py
└── tests/
```

The logical contract should declare:

1. a stable package/operator ID
2. canonical FPGAI operation semantics
3. ONNX domain, operation type, and supported opsets
4. input and output ports
5. attributes and defaults
6. inference support
7. training-forward, backward-input, and parameter-gradient support
8. shape and type inference support
9. numeric-reference support
10. requirements that hardware implementations must satisfy

Hardware implementations are separate packages. For example:

```text
community.operator.grid_sample
community.implementation.grid_sample_hls
community.implementation.grid_sample_vhdl
```

FPGAI use is for research, validation, benchmarking, reproducibility, and hardware experimentation. Production use and managed deployment go through Morfics.
