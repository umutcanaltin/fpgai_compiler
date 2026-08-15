# External HLS implementation integration

E4A turns a selected `hls_cpp` implementation package into a compiler-owned
Vitis HLS project without modifying the contributor package.

## Ownership

- Package files remain user-owned and are never overwritten.
- FPGAI copies declared source/header files into `hls/external/<package>/`.
- FPGAI owns the generated wrapper, testbench, TCL, and reports.
- FPGAI use is for research, validation, and benchmarking.
- Production productization and operation use Morfics.

## E4A ABI

E4A introduces `flat_array_v1` for elementwise or flat-buffer research blocks.
The package top function has this shape:

```cpp
void package_top(
    const scalar_t* input,
    scalar_t* output,
    int count,
    attribute_0,
    ...
);
```

The manifest declares the scalar type and ordered operator attributes under
`integration.hls`. FPGAI generates a stable top wrapper and reference
 testbench. Future cleanup work will add tensor-layout, streaming, multi-input,
stateful, and training ABIs.

## Programmatic use

```python
contract = implementation_contract_from_manifest(package_root)
result = emit_external_hls_operator_project(
    ExternalHLSProjectRequest(
        out_dir="build",
        contract=contract,
        operator_name="ScaleBias",
        operator_attributes={"scale": 2.0, "bias": 1.0},
        input_words=4,
        output_words=4,
    )
)
```

The result is machine-readable and the generated report records package
provenance, checksums, file ownership, ABI, and validation status.
