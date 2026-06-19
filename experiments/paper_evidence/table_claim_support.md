# Claim-support status

| claim | status | evidence |
|---|---|---|
| Embedded/on-chip weight strategy passes end-to-end | supported | passing benchmark for on-chip/BRAM baseline design |
| Streamed runtime weights pass end-to-end | supported | weight_stream interface plus passing benchmark |
| External-DDR runtime weights pass end-to-end | supported | weights_mem m_axi interface plus passing benchmark |
| BRAM storage binding is generated and Vitis-valid | supported | BIND_STORAGE impl=bram plus passing benchmark |
| URAM storage binding is generated and Vitis-valid | supported | BIND_STORAGE impl=uram plus passing benchmark |
| Parallel policies materialize distinct HLS parameters | supported | policy-specific PE/SIMD/unroll/partition evidence comments plus passing benchmarks |
| General automatic optimal hardware search | not claimed / not supported by this evidence | current evidence supports materialized sweeps and validation, not global optimality |
| Training accelerator support | not supported by this evidence | these tables cover inference/memory/parallel HLS paths only |
