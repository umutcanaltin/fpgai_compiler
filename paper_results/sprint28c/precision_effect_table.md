# Precision effect table

Precision rows are interpreted as arithmetic-resource effects. They should not be used to claim BRAM reduction for this small model.

| design | precision | type claim | LUT | FF | DSP | BRAM18 | URAM | latency | memory semantics | claim boundary |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `kv260_precision_fx8_3` | fx8_3 | ap_fixed<8,3> activations/weights | 4119 | 1914 | 8 | 5 | 0 | 72 | `embedded_constants` | Precision changes arithmetic resources; BRAM18 remains constant in this small model. |
| `kv260_precision_fx12_4` | fx12_4 | ap_fixed<12,4> activations/weights | 4876 | 3159 | 20 | 5 | 0 | 85 | `embedded_constants` | Precision changes arithmetic resources; BRAM18 remains constant in this small model. |
| `kv260_precision_fx16_6` | fx16_6 | ap_fixed<16,6> activations/weights | 5056 | 3531 | 24 | 5 | 0 | 90 | `embedded_constants` | Precision changes arithmetic resources; BRAM18 remains constant in this small model. |
