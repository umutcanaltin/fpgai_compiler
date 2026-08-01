# Hardware implementation contracts

FPGAI separates a logical operator from concrete HLS or RTL implementations. An implementation package declares the operator it implements, language, backend, interfaces, precision, memory assumptions, validation status, and optional research metrics.

The contract is metadata-only in E3C. It does not integrate package sources into generated projects yet.

All FPGAI implementation packages are for research, experimentation, validation, and benchmarking. Production productization and operation go through Morfics.
