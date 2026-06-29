# Memory semantics table

This table reports generated memory/data-movement semantics. It must not be interpreted as a pure BRAM-vs-DDR-vs-URAM storage-only comparison.

| design | paper label | classifier | LUT | FF | DSP | BRAM18 | URAM | latency | runtime payload | weights_mem | full W/B arrays | URAM bind | tile buffer | interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `kv260_memory_bram` | Embedded constants | `embedded_constants` | 5056 | 3531 | 24 | 5 | 0 | 90 | required=False, present=False, words=0 | False | False | False | False | Weights are compiled into the HLS design. No runtime weight payload or weights_mem AXI port is used. |
| `kv260_memory_ddr` | DDR preload full | `ddr_preload_full` | 11750 | 6932 | 34 | 9 | 0 | 190 | required=True, present=True, words=46 | True | True | False | False | Weights are loaded from weights_mem through m_axi into full local W/B arrays. This is not scalable DDR-tiled execution. |
| `kv260_memory_ddr_new_schema` | DDR preload full, new schema | `ddr_preload_full` | 11750 | 6932 | 34 | 9 | 0 | 190 | required=True, present=True, words=46 | True | True | False | False | Same generated memory behavior as DDR preload full, using the normalized data_movement.weights schema. |
| `kv260_memory_uram` | URAM preload full | `uram_preload_full` | 12653 | 6905 | 34 | 9 | 18 | 197 | required=True, present=True, words=46 | True | True | True | False | Weights are loaded from weights_mem and stored in full local URAM-bound W/B arrays. BRAM remains available for activations/scratch. |
